using MusicGen.Core.Layers;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

internal static class LayerKit
{
    /// <summary>Deterministic parameter values (the layers allocate some with uninitialised or library-random memory).</summary>
    public static void Fill(nn.Module m, int seed, double scale = 0.3)
    {
        using var _ = no_grad();
        int i = 0;
        foreach (var (_, p) in m.named_parameters().OrderBy(x => x.name, StringComparer.Ordinal))
            p.copy_(randn(p.shape, generator: new Generator((ulong)(seed * 1000 + i++))) * scale);
    }

    public static Tensor Param(nn.Module m, string name) => m.named_parameters().Single(p => p.name == name).parameter;

    public static double MaxAbs(Tensor a, Tensor b) => (a.to(ScalarType.Float64) - b.to(ScalarType.Float64)).abs().max().item<double>();
}

public class FlowMlpTests
{
    [Theory]
    [InlineData(8, 256)]
    [InlineData(1024, 2816)]
    public void TheDefaultHiddenSize_IsTheDerivedOne(int dim, int expectedHidden)
    {
        var mlp = new FlowMlp(dim);
        Assert.Equal(expectedHidden, FlowMlp.HiddenSize(dim));
        Assert.Equal(new long[] { expectedHidden, dim }, LayerKit.Param(mlp, "gate.weight").shape);
    }

    [Fact]
    public void ACustomHiddenSize_ChangesTheParameterShapes_AndNothingElse()
    {
        var mlp = new FlowMlp(8, hiddenSize: 20);
        Assert.Equal(new[] { "down.weight", "gate.weight", "up.weight" }, mlp.named_parameters().Select(p => p.name).Order().ToArray());
        Assert.Equal(new long[] { 20, 8 }, LayerKit.Param(mlp, "gate.weight").shape);
        Assert.Equal(new long[] { 20, 8 }, LayerKit.Param(mlp, "up.weight").shape);
        Assert.Equal(new long[] { 8, 20 }, LayerKit.Param(mlp, "down.weight").shape);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(20)]
    public void TheOutput_IsDownOfSiluGateTimesUp(int? hidden)
    {
        var mlp = new FlowMlp(8, hidden);
        LayerKit.Fill(mlp, 1);
        using var x = randn(new long[] { 2, 5, 8 }, generator: new Generator(3));

        using var y = mlp.forward(x);

        var xd = x.to(ScalarType.Float64);
        var g = xd.matmul(LayerKit.Param(mlp, "gate.weight").to(ScalarType.Float64).t());
        var u = xd.matmul(LayerKit.Param(mlp, "up.weight").to(ScalarType.Float64).t());
        var expected = (g * g.sigmoid() * u).matmul(LayerKit.Param(mlp, "down.weight").to(ScalarType.Float64).t());
        Assert.True(LayerKit.MaxAbs(expected, y) < 1e-5);
    }

    [Fact]
    public void ANonPositiveHiddenSize_IsRejected() => Assert.Throws<ArgumentException>(() => new FlowMlp(8, hiddenSize: 0));
}

public class FlowAttentionTests
{
    private const int Dim = 16, Heads = 2, HeadDim = 8;

    private static RotaryEmbedding Rope() =>
        new(RopeLayout.AdjacentPairs, RopeFrequencies.Plain(10000), RopeTable.InputDtypePerCall, HeadDim);

    private static FlowAttention Attn(int seed = 1)
    {
        var a = new FlowAttention(Dim, Heads, HeadDim, Rope());
        LayerKit.Fill(a, seed);
        return a;
    }

    private static Tensor X(int b, int t, int seed) => randn(new long[] { b, t, Dim }, generator: new Generator((ulong)seed));

    /// <summary>Plain float64 attention written out with tensor maths, no SDPA.</summary>
    private static Tensor Naive(FlowAttention a, Tensor x, Tensor? kv, Tensor? mask, bool rope)
    {
        var xd = x.to(ScalarType.Float64);
        var src = (kv ?? x).to(ScalarType.Float64);
        Tensor W(string n) => LayerKit.Param(a, n).to(ScalarType.Float64).t();
        long b = xd.shape[0], t = xd.shape[1], tk = src.shape[1];
        Tensor Split(Tensor y, long len) => y.view(b, len, Heads, HeadDim).transpose(1, 2);
        var q = Split(xd.matmul(W("q_proj.weight")), t);
        var k = Split(src.matmul(W("k_proj.weight")), tk);
        var v = Split(src.matmul(W("v_proj.weight")), tk);
        if (rope)
        {
            var r = Rope();
            q = r.Apply(q, null, 2); k = r.Apply(k, null, 2);
        }
        var scores = q.matmul(k.transpose(-2, -1)) / Math.Sqrt(HeadDim);
        if (mask is not null) scores = scores.masked_fill(mask.logical_not(), double.NegativeInfinity);
        var o = scores.softmax(-1).matmul(v).transpose(1, 2).contiguous().view(b, t, Heads * HeadDim);
        return o.matmul(W("o_proj.weight"));
    }

    [Fact]
    public void SelfAttentionWithoutMask_MatchesTheNaiveFloat64Reference()
    {
        var a = Attn();
        using var x = X(2, 6, 1);
        using var y = a.forward(x);
        Assert.True(LayerKit.MaxAbs(Naive(a, x, null, null, rope: true), y) < 1e-5);
    }

    [Fact]
    public void OmittingMaskAndKvInput_IsTheSameAsTheOriginalForward()
    {
        var a = Attn();
        using var x = X(2, 6, 2);
        using var viaForward = a.forward(x);
        using var viaExtended = a.Forward(x, null, null);
        Assert.True(viaForward.Equals(viaExtended), "bit-for-bit");
        Assert.Equal(new[] { "k_proj.weight", "o_proj.weight", "q_proj.weight", "v_proj.weight" }, a.named_parameters().Select(p => p.name).Order().ToArray());
    }

    [Theory]
    [InlineData(7, 3)]
    [InlineData(5, 1)]
    public void AMaskThatHidesTheLastPositions_EqualsTruncatingTheSequence(int t, int hidden)
    {
        var a = Attn();
        int visible = t - hidden;
        using var x = X(2, t, 3);
        var keep = arange(t) < visible;                                          // [T]
        using var mask = keep.reshape(1, 1, 1, t).expand(2, 1, t, t).contiguous();

        using var masked = a.Forward(x, mask);
        using var truncated = a.Forward(x[.., ..visible]);

        Assert.True(LayerKit.MaxAbs(masked[.., ..visible], truncated) < 1e-5, "visible positions must not see the masked ones");
        Assert.True(LayerKit.MaxAbs(Naive(a, x, null, mask, rope: true), masked) < 1e-5);
    }

    [Fact]
    public void ADifferentMaskPerBatchRow_IsHonoured()
    {
        var a = Attn();
        using var x = X(2, 7, 4);
        var lengths = new[] { 7, 4 };
        var rows = lengths.Select(l => (arange(7) < l).reshape(1, 1, 1, 7).expand(1, 1, 7, 7)).ToArray();
        using var mask = cat(rows, 0).contiguous();

        using var y = a.Forward(x, mask);

        for (int i = 0; i < 2; i++)
        {
            using var single = a.Forward(x[i..(i + 1), ..lengths[i]]);
            Assert.True(LayerKit.MaxAbs(y[i..(i + 1), ..lengths[i]], single) < 1e-5, $"row {i}");
        }
    }

    [Fact]
    public void CrossAttention_UsesTheOtherSequenceForKeysAndValues_WithoutRotaryEmbedding()
    {
        var a = Attn();
        using var x = X(2, 4, 5);
        using var kv = X(2, 9, 6);

        using var y = a.Forward(x, null, kv);

        Assert.Equal(new long[] { 2, 4, Dim }, y.shape);
        Assert.True(LayerKit.MaxAbs(Naive(a, x, kv, null, rope: false), y) < 1e-5);
        using var self = a.Forward(x);
        Assert.True(LayerKit.MaxAbs(self, y) > 1e-3, "cross-attention must differ from self-attention");
    }

    [Fact]
    public void CrossAttentionWithAMask_HidesTheMaskedKeys()
    {
        var a = Attn();
        using var x = X(1, 3, 7);
        using var kv = X(1, 8, 8);
        using var mask = (arange(8) < 5).reshape(1, 1, 1, 8).expand(1, 1, 3, 8).contiguous();

        using var masked = a.Forward(x, mask, kv);
        using var truncated = a.Forward(x, null, kv[.., ..5]);

        Assert.True(LayerKit.MaxAbs(masked, truncated) < 1e-5);
    }
}
