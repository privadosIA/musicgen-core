using System.Text.Json;
using MusicGen.Core.Layers;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public sealed class StrictLoaderTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("strict-loader-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private const string Prefix = "The test checkpoint does not match the model:";

    private sealed class Net : nn.Module<Tensor, Tensor>
    {
        private readonly nn.Module<Tensor, Tensor> fc1;
        public Net(bool withBuffer = false) : base(nameof(Net))
        {
            fc1 = nn.Linear(2, 3);
            if (withBuffer) register_buffer("stat", zeros(3));
            RegisterComponents();
        }
        public override Tensor forward(Tensor x) => fc1.forward(x);
    }

    private sealed class ConvNet : nn.Module<Tensor, Tensor>
    {
        private readonly Conv1dLayer conv = new(2, 4, 3);
        public ConvNet() : base(nameof(ConvNet)) => RegisterComponents();
        public override Tensor forward(Tensor x) => conv.forward(x);
    }

    private sealed class Wrapper : nn.Module<Tensor, Tensor>
    {
        private readonly Net decoder = new();
        public Wrapper() : base(nameof(Wrapper)) => RegisterComponents();
        public override Tensor forward(Tensor x) => decoder.forward(x);
    }

    private static Dictionary<string, Tensor> Values(int seed) => new()
    {
        ["fc1.weight"] = randn(new long[] { 3, 2 }, generator: new Generator((ulong)seed)),
        ["fc1.bias"] = randn(new long[] { 3 }, generator: new Generator((ulong)seed + 100)),
    };

    private CheckpointIndex Write(Dictionary<string, Tensor> tensors, string name = "model.safetensors")
    {
        SafeTensorsWriter.Save(Path.Combine(_dir, name), tensors);
        return CheckpointIndex.Open(Path.Combine(_dir, name));
    }

    private static StrictLoadOptions Options() => new(Prefix);

    private static Dictionary<string, Tensor> Snapshot(nn.Module m) => m.named_parameters().ToDictionary(p => p.name, p => p.parameter.detach().clone());

    [Fact]
    public void ACompleteCheckpoint_IsCopiedExactly_AndCounted()
    {
        var net = new Net();
        var values = Values(1);
        var result = StrictLoader.Load(net, Write(values), CPU, Options());

        foreach (var (name, p) in net.named_parameters()) Assert.True(p.Equals(values[name]), name);
        Assert.Equal(2, result.TensorsRead);
        Assert.Equal(2, result.ModelTensors);
        Assert.Equal(0, result.PairsFolded);
        Assert.Equal(0, result.OutsidePrefix);
        Assert.Equal((6 + 3) * 4, result.BytesRead);
    }

    [Fact]
    public void MissingExtraAndWrongShape_AreReportedTogether_BeforeAnythingIsRead()
    {
        var net = new Net();
        var before = Snapshot(net);
        var values = Values(1);
        values.Remove("fc1.bias");                                   // missing
        values["fc1.weight"] = zeros(new long[] { 3, 5 });           // wrong shape
        values["extra.w"] = zeros(new long[] { 1 });                 // extra

        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(net, Write(values), CPU, Options() with { UnexpectedLabel = "decoder keys in the checkpoint" }));

        Assert.StartsWith(Prefix + "\n  ", ex.Message);
        Assert.Contains("1 model keys are missing from the checkpoint: fc1.bias", ex.Message);
        Assert.Contains("1 decoder keys in the checkpoint are not in the model: extra.w", ex.Message);
        Assert.Contains("fc1.weight: the model has [3, 2], the checkpoint [3, 5]", ex.Message);
        foreach (var (name, p) in net.named_parameters()) Assert.True(p.Equals(before[name]), $"{name} was modified by a failed load");
    }

    [Fact]
    public void ThePrefixSelectsASubset_AndTheRestIsNotLookedAt()
    {
        var net = new Net();
        var values = Values(2).ToDictionary(kv => "m." + kv.Key, kv => kv.Value);
        values["other.huge"] = zeros(new long[] { 4 });

        var result = StrictLoader.Load(net, Write(values), CPU, Options() with { Prefix = "m." });

        Assert.Equal(1, result.OutsidePrefix);
        Assert.True(net.named_parameters().Single(p => p.name == "fc1.bias").parameter.Equals(values["m.fc1.bias"]));
    }

    [Fact]
    public void TheStripPrefixCanBeShorterThanTheFilterPrefix()
    {
        var model = new Wrapper();
        var values = Values(3).ToDictionary(kv => "outer.decoder." + kv.Key, kv => kv.Value);
        values["outer.encoder.w"] = zeros(new long[] { 2 });

        var result = StrictLoader.Load(model, Write(values), CPU, Options() with { Prefix = "outer.decoder.", StripPrefix = "outer." });

        Assert.Equal(1, result.OutsidePrefix);
        Assert.True(model.named_parameters().Single(p => p.name == "decoder.fc1.weight").parameter.Equals(values["outer.decoder.fc1.weight"]));
    }

    [Fact]
    public void TheAllowList_SkipsKeysTheModelDoesNotHave_ButOnlyThose()
    {
        var values = Values(4);
        values["junk.cluster_size"] = zeros(new long[] { 2 });
        var index = Write(values);

        var strict = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Net(), index, CPU, Options()));
        Assert.Contains("junk.cluster_size", strict.Message);

        var result = StrictLoader.Load(new Net(), index, CPU, Options() with { Skip = k => k.StartsWith("junk.") });
        Assert.Equal(1, result.SkippedByAllowList);
        Assert.Equal(2, result.TensorsRead);

        values["other.w"] = zeros(new long[] { 2 });
        var stillStrict = Assert.Throws<InvalidDataException>(() =>
            StrictLoader.Load(new Net(), Write(values, "b.safetensors"), CPU, Options() with { Skip = k => k.StartsWith("junk.") }));
        Assert.Contains("other.w", stillStrict.Message);
        Assert.DoesNotContain("junk.cluster_size", stillStrict.Message);
    }

    private static (Tensor g, Tensor v) Pair(int seed) =>
        (rand(new long[] { 4, 1, 1 }, generator: new Generator((ulong)seed)) + 0.5, randn(new long[] { 4, 2, 3 }, generator: new Generator((ulong)seed + 1)));

    [Fact]
    public void AWeightNormPair_IsFoldedIntoThePlainWeight()
    {
        var (g, v) = Pair(5);
        var bias = randn(new long[] { 4 }, generator: new Generator(9));
        var values = new Dictionary<string, Tensor>
        {
            ["conv.parametrizations.weight.original0"] = g,
            ["conv.parametrizations.weight.original1"] = v,
            ["conv.bias"] = bias,
        };
        var net = new ConvNet();

        var result = StrictLoader.Load(net, Write(values), CPU, Options() with { FoldWeightNorm = true });

        Assert.Equal(1, result.PairsFolded);
        Assert.Equal(3, result.TensorsRead);
        Assert.True(net.named_parameters().Single(p => p.name == "conv.weight").parameter.Equals(WeightNorm.Fold(g, v)));
        Assert.True(net.named_parameters().Single(p => p.name == "conv.bias").parameter.Equals(bias));
    }

    [Theory]
    [InlineData("original0", "has original0 but not original1")]
    [InlineData("original1", "has original1 but not original0")]
    public void AnIncompleteWeightNormPair_IsReported(string present, string expected)
    {
        var (g, v) = Pair(6);
        var values = new Dictionary<string, Tensor> { ["conv.bias"] = zeros(new long[] { 4 }) };
        values[$"conv.parametrizations.weight.{present}"] = present == "original0" ? g : v;

        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new ConvNet(), Write(values), CPU, Options() with { FoldWeightNorm = true }));

        Assert.Contains($"incomplete weight_norm pair in conv: it {expected}", ex.Message);
    }

    [Fact]
    public void ACompletePair_WhoseScaleIsNotPerChannel_IsReported()
    {
        var (_, v) = Pair(7);
        var values = new Dictionary<string, Tensor>
        {
            ["conv.parametrizations.weight.original0"] = ones(new long[] { 4, 2, 1 }),
            ["conv.parametrizations.weight.original1"] = v,
            ["conv.bias"] = zeros(new long[] { 4 }),
        };
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new ConvNet(), Write(values), CPU, Options() with { FoldWeightNorm = true }));
        Assert.Contains("is not a per-channel scale", ex.Message);
    }

    [Fact]
    public void WithoutFolding_APairIsJustUnexpectedKeys()
    {
        var (g, v) = Pair(8);
        var values = new Dictionary<string, Tensor>
        {
            ["conv.parametrizations.weight.original0"] = g,
            ["conv.parametrizations.weight.original1"] = v,
            ["conv.bias"] = zeros(new long[] { 4 }),
        };
        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new ConvNet(), Write(values), CPU, Options()));
        Assert.Contains("conv.weight", ex.Message);
        Assert.Contains("original0", ex.Message);
    }

    [Fact]
    public void Tensors_AreReadByFileAndOffset_NotByKey()
    {
        // fc1.weight lives in the file that sorts first and fc1.bias in the one that sorts last: key order would read bias first.
        var net = new Net();
        var values = Values(10);
        SafeTensorsWriter.Save(Path.Combine(_dir, "s0.safetensors"), new Dictionary<string, Tensor> { ["fc1.weight"] = values["fc1.weight"] });
        SafeTensorsWriter.Save(Path.Combine(_dir, "s1.safetensors"), new Dictionary<string, Tensor> { ["fc1.bias"] = values["fc1.bias"] });
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new { weight_map = new Dictionary<string, string> { ["fc1.weight"] = "s0.safetensors", ["fc1.bias"] = "s1.safetensors" } }));

        var result = StrictLoader.Load(net, CheckpointIndex.Open(_dir), CPU, Options());

        Assert.Equal(new[] { "fc1.weight", "fc1.bias" }, result.ReadOrder);
        Assert.True(net.named_parameters().Single(p => p.name == "fc1.weight").parameter.Equals(values["fc1.weight"]));
    }

    [Fact]
    public void WithinOneFile_TensorsAreReadByOffset_NotByKey()
    {
        // Hand-made file: the data of fc1.weight comes BEFORE the data of fc1.bias, the opposite of key order.
        var bias = new float[] { 1, 2, 3 };
        var weight = new float[] { 4, 5, 6, 7, 8, 9 };
        var header = JsonSerializer.SerializeToUtf8Bytes(new Dictionary<string, object>
        {
            ["fc1.bias"] = new { dtype = "F32", shape = new[] { 3 }, data_offsets = new[] { 24, 36 } },
            ["fc1.weight"] = new { dtype = "F32", shape = new[] { 3, 2 }, data_offsets = new[] { 0, 24 } },
        });
        var padded = header.Concat(Enumerable.Repeat((byte)' ', (8 - header.Length % 8) % 8)).ToArray();
        var path = Path.Combine(_dir, "manual.safetensors");
        using (var fs = File.Create(path))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write((long)padded.Length);
            bw.Write(padded);
            foreach (var f in weight.Concat(bias)) bw.Write(f);
        }
        var net = new Net();

        var result = StrictLoader.Load(net, CheckpointIndex.Open(path), CPU, Options());

        Assert.Equal(new[] { "fc1.weight", "fc1.bias" }, result.ReadOrder);   // offset order
        Assert.Equal(weight, net.named_parameters().Single(p => p.name == "fc1.weight").parameter.data<float>().ToArray());
        Assert.Equal(bias, net.named_parameters().Single(p => p.name == "fc1.bias").parameter.data<float>().ToArray());
    }

    [Fact]
    public void TheReadDtypeAndTheModelDtype_AreApplied()
    {
        var net = new Net();
        var values = Values(11);
        StrictLoader.Load(net, Write(values), CPU, Options() with { ReadDtype = ScalarType.BFloat16, ModelDtype = ScalarType.BFloat16 });

        foreach (var (name, p) in net.named_parameters())
        {
            Assert.Equal(ScalarType.BFloat16, p.dtype);
            Assert.True(p.to(ScalarType.Float32).Equals(values[name].to(ScalarType.BFloat16).to(ScalarType.Float32)), name);
        }
    }

    [Fact]
    public void Buffers_AreLoadedOnlyWhenAsked()
    {
        var values = Values(12);
        values["stat"] = tensor(new[] { 1f, 2f, 3f });
        var index = Write(values);

        var withBuffers = new Net(withBuffer: true);
        StrictLoader.Load(withBuffers, index, CPU, Options() with { IncludeBuffers = true });
        Assert.True(withBuffers.named_buffers().Single().buffer.Equals(values["stat"]));

        var ex = Assert.Throws<InvalidDataException>(() => StrictLoader.Load(new Net(withBuffer: true), index, CPU, Options()));
        Assert.Contains("stat", ex.Message);
    }
}
