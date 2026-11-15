using System.Diagnostics;
using System.Text;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

/// <summary>
/// The checkpoint reader is the part of the project that parses files from outside, so it must fail cleanly and cheaply on hostile or corrupt input:
/// absurd header lengths, shapes or offsets (no giant allocation, no read past the end) and shard names that point outside the checkpoint directory.
/// </summary>
public sealed class SafeTensorsHardeningTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("safetensors-hardening-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string Write(string name, long headerLengthPrefix, string header, int dataBytes)
    {
        var path = Path.Combine(_dir, name);
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes(headerLengthPrefix));
        fs.Write(Encoding.UTF8.GetBytes(header));
        fs.Write(new byte[dataBytes]);
        return path;
    }

    private string WriteValid(string name, string header, int dataBytes) => Write(name, Encoding.UTF8.GetByteCount(header), header, dataBytes);

    [Theory]
    [InlineData(-1L)]
    [InlineData(0L)]
    [InlineData(100_000_001L)]                 // above the 100 MB limit
    [InlineData(int.MaxValue)]                 // would be a 2 GB allocation
    [InlineData(long.MaxValue)]
    [InlineData(1_000L)]                       // longer than the file
    public void Absurd_header_lengths_are_rejected_before_allocating(long prefix)
    {
        var path = Write("bad.safetensors", prefix, """{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4);
        var sw = Stopwatch.StartNew();

        Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(path));
        Assert.Throws<InvalidDataException>(() => SafeTensors.ReadMetadata(path));

        Assert.True(sw.ElapsedMilliseconds < 2000, "rejecting the header should be immediate");
    }

    [Fact]
    public void A_file_shorter_than_the_length_prefix_is_rejected()
    {
        var path = Path.Combine(_dir, "tiny.safetensors");
        File.WriteAllBytes(path, new byte[3]);
        Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(path));
    }

    [Fact]
    public void A_header_that_is_not_json_is_rejected_with_a_clear_error()
    {
        var path = WriteValid("notjson.safetensors", "this is not json", 0);
        var e = Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(path));
        Assert.Contains("not valid JSON", e.Message);
    }

    [Theory]
    [InlineData("""{"w":{"dtype":"F32","shape":[1099511627776],"data_offsets":[0,4]}}""")]            // 4 TB of floats claimed for 4 bytes
    [InlineData("""{"w":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,4]}}""")]    // the product overflows
    [InlineData("""{"w":{"dtype":"F32","shape":[-3],"data_offsets":[0,4]}}""")]                      // negative dimension
    [InlineData("""{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}""")]                       // 8 bytes expected, header says 4
    public void Shapes_that_do_not_match_their_bytes_are_rejected_without_allocating(string header)
    {
        var path = WriteValid("shape.safetensors", header, 4);
        var sw = Stopwatch.StartNew();
        Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(path));
        Assert.True(sw.ElapsedMilliseconds < 2000);
    }

    [Theory]
    [InlineData("""{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,40]}}""")]     // end beyond the data
    [InlineData("""{"w":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}""")]      // end before start
    [InlineData("""{"w":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}""")]     // negative start
    [InlineData("""{"w":{"dtype":"F32","shape":[1],"data_offsets":[0]}}""")]        // one offset
    [InlineData("""{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,2,4]}}""")]    // three offsets
    [InlineData("""{"w":{"dtype":"F32","shape":[1]}}""")]                           // no offsets at all
    [InlineData("""{"w":{"shape":[1],"data_offsets":[0,4]}}""")]                    // no dtype
    [InlineData("""{"w":[1,2,3]}""")]                                               // not an object
    public void Offsets_and_entries_outside_the_file_are_rejected(string header)
    {
        var path = WriteValid("offsets.safetensors", header, 4);
        Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(path));
    }

    [Fact]
    public void A_valid_file_still_loads_and_unsupported_dtypes_keep_their_own_error()
    {
        var ok = WriteValid("ok.safetensors", """{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"k":"v"}}""", 8);
        var index = CheckpointIndex.Open(ok);
        Assert.Equal(new long[] { 2 }, index.Tensors["w"].Shape);
        Assert.Equal("v", SafeTensors.ReadMetadata(ok)["k"]);

        var fp8 = WriteValid("fp8.safetensors", """{"w":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}""", 2);
        var e = Assert.Throws<NotSupportedException>(() => CheckpointIndex.Open(fp8));
        Assert.Contains("F8_E4M3", e.Message);
    }

    [Fact]
    public void Zero_sized_tensors_are_accepted()
    {
        var path = WriteValid("empty.safetensors", """{"e":{"dtype":"F32","shape":[0,5],"data_offsets":[0,0]}}""", 0);
        Assert.Equal(new long[] { 0, 5 }, CheckpointIndex.Open(path).Tensors["e"].Shape);
    }

    private void WriteIndex(string shardName)
    {
        WriteValid("real.safetensors", """{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4);
        var json = "{\"weight_map\":{\"w\":" + System.Text.Json.JsonSerializer.Serialize(shardName) + "}}";
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"), json);
    }

    [Theory]
    [InlineData("../real.safetensors")]
    [InlineData("sub/real.safetensors")]
    [InlineData("sub\\real.safetensors")]
    [InlineData("/etc/passwd")]
    [InlineData("..")]
    [InlineData("")]
    public void Shard_names_that_leave_the_checkpoint_directory_are_rejected(string shardName)
    {
        WriteIndex(shardName);
        var e = Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(_dir));
        Assert.Contains("invalid shard name", e.Message);
        Assert.Throws<InvalidDataException>(() => SafeTensors.LoadCheckpoint(_dir));
    }

    [Fact]
    public void A_plain_shard_name_works()
    {
        WriteIndex("real.safetensors");
        Assert.Contains("w", CheckpointIndex.Open(_dir).Tensors.Keys);
    }

    [Fact]
    public void A_non_string_shard_in_the_index_is_rejected()
    {
        WriteValid("real.safetensors", """{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}""", 4);
        File.WriteAllText(Path.Combine(_dir, "model.safetensors.index.json"), """{"weight_map":{"w":null}}""");
        Assert.Throws<InvalidDataException>(() => CheckpointIndex.Open(_dir));
    }
}
