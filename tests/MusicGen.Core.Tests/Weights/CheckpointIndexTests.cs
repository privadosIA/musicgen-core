using MusicGen.Core.Testing;
using System.Text.Json;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public sealed class CheckpointIndexTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ckpt-index-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static Tensor T(params float[] v) => tensor(v);

    private void WriteShard(string name, params (string key, Tensor t)[] tensors) =>
        SafeTensorsWriter.Save(Path.Combine(_dir, name), tensors.ToDictionary(x => x.key, x => x.t));

    private void WriteIndex(params (string key, string shard)[] map) =>
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new { metadata = new { total_size = 0 }, weight_map = map.ToDictionary(x => x.key, x => x.shard) }));

    [Fact]
    public void Sharded_ExposesExactlyTheIndexKeys()
    {
        WriteShard("s1.safetensors", ("a.weight", T(1, 2, 3)), ("b.weight", arange(6, dtype: ScalarType.Float32).reshape(2, 3)));
        WriteShard("s2.safetensors", ("c.scale", T(4)));
        WriteIndex(("a.weight", "s1.safetensors"), ("b.weight", "s1.safetensors"), ("c.scale", "s2.safetensors"));

        var index = CheckpointIndex.Open(_dir);

        Assert.Equal(new[] { "a.weight", "b.weight", "c.scale" }, index.Tensors.Keys.Order());
        var b = index.Tensors["b.weight"];
        Assert.Equal(Path.Combine(_dir, "s1.safetensors"), b.File);
        Assert.Equal("F32", b.Dtype);
        Assert.Equal(new long[] { 2, 3 }, b.Shape);
        Assert.Equal(24, b.Length);
        Assert.Equal("s2.safetensors", Path.GetFileName(index.Tensors["c.scale"].File));
    }

    [Fact]
    public void Shard_MissingAnIndexedKey_FailsNamingShardAndKey()
    {
        WriteShard("s1.safetensors", ("a.weight", T(1)));
        WriteIndex(("a.weight", "s1.safetensors"), ("b.weight", "s1.safetensors"));

        var ex = Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(_dir));

        Assert.Contains("s1.safetensors", ex.Message);
        Assert.Contains("missing", ex.Message);
        Assert.Contains("b.weight", ex.Message);
    }

    [Fact]
    public void Shard_WithKeyNotInIndex_FailsNamingShardAndKey()
    {
        WriteShard("s1.safetensors", ("a.weight", T(1)), ("z.extra", T(2)));
        WriteIndex(("a.weight", "s1.safetensors"));

        var ex = Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(_dir));

        Assert.Contains("s1.safetensors", ex.Message);
        Assert.Contains("extra", ex.Message);
        Assert.Contains("z.extra", ex.Message);
    }

    [Fact]
    public void Inconsistencies_AcrossShards_AreReportedTogether()
    {
        WriteShard("s1.safetensors", ("a.weight", T(1)));
        WriteShard("s2.safetensors", ("c.scale", T(1)), ("q.extra", T(2)));
        WriteIndex(("a.weight", "s1.safetensors"), ("b.weight", "s1.safetensors"), ("c.scale", "s2.safetensors"));

        var ex = Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(_dir));

        Assert.Contains("shard s1.safetensors: missing 1 keys from the index: b.weight", ex.Message);
        Assert.Contains("shard s2.safetensors: 1 extra keys that the index does not list: q.extra", ex.Message);
    }

    [Fact]
    public void IndexReferencingAbsentShardFile_Fails()
    {
        WriteIndex(("a.weight", "nope.safetensors"));

        var ex = Assert.Throws<FileNotFoundException>(() => CheckpointIndex.Open(_dir));

        Assert.Contains("nope.safetensors", ex.Message);
    }

    [Fact]
    public void SingleFile_WithoutIndex_LoadsItsKeys()
    {
        WriteShard("model.safetensors", ("a.weight", T(1, 2)), ("b.weight", T(3)));

        var index = CheckpointIndex.Open(_dir);

        Assert.Equal(new[] { "a.weight", "b.weight" }, index.Tensors.Keys.Order());
        Assert.Equal(new long[] { 2 }, index.Tensors["a.weight"].Shape);
    }

    [Fact]
    public void UnsupportedDtype_FailsNamingTensorAndDtype()
    {
        // Hand-built header: one F8_E4M3 tensor, which MapDtype does not support.
        var header = """{"w.fp8":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}"""u8.ToArray();
        using (var fs = File.Create(Path.Combine(_dir, "model.safetensors")))
        {
            fs.Write(BitConverter.GetBytes((long)header.Length));
            fs.Write(header);
            fs.Write(new byte[2]);
        }

        var ex = Assert.Throws<NotSupportedException>(() => CheckpointIndex.Open(_dir));

        Assert.Contains("w.fp8", ex.Message);
        Assert.Contains("F8_E4M3", ex.Message);
    }

    [Theory]
    [InlineData(ScalarType.BFloat16)]
    [InlineData(ScalarType.Float32)]
    [InlineData(ScalarType.Int64)]
    public void Read_KeepsDtypeShapeAndBytesOfTheFile(ScalarType dtype)
    {
        var original = (arange(12, dtype: ScalarType.Float32) * 1.37f - 5f).reshape(3, 4).to(dtype);
        WriteShard("s1.safetensors", ("x", original), ("other", T(9, 9)));
        WriteShard("s2.safetensors", ("y", T(7)));
        WriteIndex(("x", "s1.safetensors"), ("other", "s1.safetensors"), ("y", "s2.safetensors"));

        var read = CheckpointIndex.Open(_dir).Read("x");

        Assert.Equal(dtype, read.dtype);
        Assert.Equal(new long[] { 3, 4 }, read.shape);
        Assert.Equal(original.bytes.ToArray(), read.bytes.ToArray());
    }

    [Fact]
    public void Read_UnknownKey_Fails()
    {
        WriteShard("model.safetensors", ("a", T(1)));

        var ex = Assert.Throws<KeyNotFoundException>(() => CheckpointIndex.Open(_dir).Read("nope"));

        Assert.Contains("nope", ex.Message);
    }

    private ushort[] ReadAsBf16Bits(float[] values)
    {
        WriteShard("model.safetensors", ("v", tensor(values)));
        using var bf = CheckpointIndex.Open(_dir).Read("v", ScalarType.BFloat16);
        Assert.Equal(ScalarType.BFloat16, bf.dtype);
        return System.Runtime.InteropServices.MemoryMarshal.Cast<byte, ushort>(bf.bytes).ToArray();
    }

    [Fact]
    public void Bf16Conversion_MatchesOracle_OnEdgeValues()
    {
        var bits = new uint[]
        {
            0x00000000, 0x80000000,                         // ±0
            0x7F800000, 0xFF800000,                         // ±Inf
            0x3F808000, 0x3F818000, 0x3F807FFF, 0x3F808001, // tie to even (rounds down), tie to even (rounds up), just below / above the tie
            0xBF808000, 0xBF818000,                         // same, negative
            0x00000001, 0x00008000, 0x00018000, 0x007FFFFF, // subnormals (including ties)
            0x80000001, 0x80018000, 0x807FFFFF,
            0x7F7FFFFF, 0x7F7F8000, 0xFF7FFFFF,             // overflow to Inf when rounding
            0x3F800000, 0x40490FDB,                         // 1.0 and pi
        };
        var values = bits.Select(BitConverter.UInt32BitsToSingle).ToArray();

        var got = ReadAsBf16Bits(values);

        Assert.Equal(values.Select(Bf16.RoundNearestEven), got);
    }

    [Fact]
    public void Bf16Conversion_MatchesOracle_OnOneMillionRandomValues()
    {
        var rng = new Random(12345);
        var values = new float[1_000_000];
        for (int i = 0; i < values.Length; i++)
        {
            float f;
            do f = BitConverter.UInt32BitsToSingle((uint)rng.NextInt64(0, 1L << 32));
            while (float.IsNaN(f));
            values[i] = f;
        }

        var got = ReadAsBf16Bits(values);

        var mismatches = Enumerable.Range(0, values.Length).Where(i => got[i] != Bf16.RoundNearestEven(values[i])).Take(5).ToList();
        Assert.True(mismatches.Count == 0,
            "first mismatches: " + string.Join(", ", mismatches.Select(i => $"0x{BitConverter.SingleToUInt32Bits(values[i]):X8}→got 0x{got[i]:X4} oracle 0x{Bf16.RoundNearestEven(values[i]):X4}")));
    }

    [Fact]
    public void EmptyDirectory_Fails() =>
        Assert.Throws<FileNotFoundException>(() => CheckpointIndex.Open(_dir));
}
