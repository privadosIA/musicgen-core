using MusicGen.Core.Layers;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public class RotaryEmbeddingTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private static Dictionary<string, Tensor> Reference() => SafeTensors.LoadFile(CoreFixtures.PathOf("rope_hf.safetensors"));

    public static IEnumerable<object[]> Cases() { yield return new object[] { "a" }; yield return new object[] { "b" }; }

    private static double MaxAbs(Tensor got, Tensor expected) => (got - expected).abs().max().item<float>();

    [Theory]
    [MemberData(nameof(Cases))]
    public void HalfRotation_PerCallFloat32_MatchesHuggingFaceWithin1e6(string name)
    {
        var r = Reference();
        int headDim = (int)r[$"{name}.params"][0].item<float>();
        double theta = r[$"{name}.params"][1].item<float>();
        var rope = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(theta), RopeTable.InputDtypePerCall, headDim);

        using var y = rope.Apply(r[$"{name}.x"], r[$"{name}.pos"], seqDim: 2);

        var err = MaxAbs(y, r[$"{name}.out"]);
        output.WriteLine($"case {name}: max abs error vs HuggingFace {err:E3}");
        Assert.True(err < 1e-6, $"max abs error {err:E3}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void HalfRotation_CachedFloat32Table_MatchesHuggingFaceWithin1e6(string name)
    {
        var r = Reference();
        int headDim = (int)r[$"{name}.params"][0].item<float>();
        double theta = r[$"{name}.params"][1].item<float>();
        var rope = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(theta), RopeTable.Float32Cached, headDim, maxSeqLen: 8192);

        using var y = rope.Apply(r[$"{name}.x"], r[$"{name}.pos"], seqDim: 2);

        var err = MaxAbs(y, r[$"{name}.out"]);
        output.WriteLine($"case {name}: max abs error vs HuggingFace {err:E3}");
        Assert.True(err < 1e-6, $"max abs error {err:E3}");
    }

    [Fact]
    public void HalfRotation_SeqDim1_EqualsSeqDim2()
    {
        var r = Reference();
        var rope = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(10000), RopeTable.Float32Cached, 64, maxSeqLen: 64);
        using var a = rope.Apply(r["a.x"], r["a.pos"], 2);
        using var b = rope.Apply(r["a.x"].transpose(1, 2), r["a.pos"], 1);
        Assert.True(a.Equals(b.transpose(1, 2)));
    }

    [Fact]
    public void TheTwoLayouts_AreNotInterchangeable()
    {
        var r = Reference();
        var half = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, 64);
        var pairs = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, 64);
        using var h = half.Apply(r["a.x"], r["a.pos"], 2);
        using var p = pairs.Apply(r["a.x"], r["a.pos"], 2);
        Assert.True(MaxAbs(h, p) > 0.1, "the layouts must give different outputs");
        Assert.True(MaxAbs(pairs.Apply(r["a.x"], r["a.pos"], 2), h) > 0.1);
    }

    [Fact]
    public void PartialRotation_LeavesTheTrailingChannelsUntouched()
    {
        var r = Reference();
        var rope = new RotaryEmbedding(RopeLayout.HalfRotation, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, 64, ropeDim: 32);
        using var y = rope.Apply(r["a.x"], r["a.pos"], 2);
        Assert.True(y[.., .., .., 32..].Equals(r["a.x"][.., .., .., 32..]));
        Assert.False(y[.., .., .., ..32].Equals(r["a.x"][.., .., .., ..32]));
    }

    [Fact]
    public void Llama3Scaling_ChangesOnlyTheLowFrequencies()
    {
        var plain = RopeFrequencies.Plain(500_000).InverseFrequencies(128, ScalarType.Float32, CPU);
        var scaled = RopeFrequencies.Llama3Scaled(500_000, 32).InverseFrequencies(128, ScalarType.Float32, CPU);
        Assert.True(plain[0].Equals(scaled[0]), "the highest frequency is not scaled");
        Assert.True(Math.Abs(scaled[63].item<float>() - plain[63].item<float>() / 32) < 1e-12, "the lowest frequency is divided by the scale factor");
    }

    [Fact]
    public void TheCachedTable_RequiresPositions_AndAMaxSeqLen()
    {
        Assert.Throws<ArgumentException>(() => new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.Float32Cached, 8));
        var rope = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.Float32Cached, 8, maxSeqLen: 4);
        Assert.Throws<ArgumentException>(() => rope.Apply(zeros(1, 1, 2, 8), null, 2));
    }

    // The layout is a required argument, with no default. Creating the class without one does not compile:
    //     new RotaryEmbedding(RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, 64);   // error CS1503: cannot convert RopeFrequencies to RopeLayout
    //     new RotaryEmbedding();                                                                // error CS7036: no argument given for 'layout'
    // This test is the executable counterpart: no parameterless constructor, and the layout parameter has no default value.
    [Fact]
    public void TheLayout_IsARequiredConstructorArgument()
    {
        var ctors = typeof(RotaryEmbedding).GetConstructors();
        Assert.DoesNotContain(ctors, c => c.GetParameters().Length == 0);
        var ctor = Assert.Single(ctors);
        var layout = ctor.GetParameters()[0];
        Assert.Equal(typeof(RopeLayout), layout.ParameterType);
        Assert.False(layout.HasDefaultValue);
        Assert.DoesNotContain(ctors.SelectMany(c => c.GetParameters()), p => p.ParameterType == typeof(RopeLayout) && p.HasDefaultValue);
    }
}
