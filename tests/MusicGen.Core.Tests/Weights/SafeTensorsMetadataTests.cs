using System.Text;
using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public sealed class SafeTensorsMetadataTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("st-meta-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);
    private string P(string name) => Path.Combine(_dir, name);

    private static Dictionary<string, Tensor> OneTensor() => new() { ["a"] = tensor(new[] { 1f, 2f, 3f }) };

    [Fact]
    public void WithoutMetadata_TheFileIsByteForByteTheExpectedOne()
    {
        SafeTensorsWriter.Save(P("plain.safetensors"), OneTensor());

        // Hand-written header: the pre-metadata format, padded with spaces up to a multiple of 8.
        var json = Encoding.UTF8.GetBytes("""{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}""");
        int pad = (8 - json.Length % 8) % 8;
        var expected = BitConverter.GetBytes((long)(json.Length + pad))
            .Concat(json).Concat(Enumerable.Repeat((byte)' ', pad))
            .Concat(new[] { 1f, 2f, 3f }.SelectMany(BitConverter.GetBytes)).ToArray();

        Assert.Equal(expected, File.ReadAllBytes(P("plain.safetensors")));
    }

    [Fact]
    public void WithMetadata_RoundTripsTheStrings_AndTensorsAreUntouched()
    {
        var meta = new Dictionary<string, string> { ["seed"] = "1234", ["device"] = "cpu", ["prompt"] = "ids:prompt_ids.json" };
        SafeTensorsWriter.Save(P("meta.safetensors"), OneTensor(), meta);

        Assert.Equal(meta.OrderBy(kv => kv.Key), SafeTensors.ReadMetadata(P("meta.safetensors")).OrderBy(kv => kv.Key));
        Assert.Equal(new[] { "a" }, SafeTensors.Describe(P("meta.safetensors")).Select(x => x.name));   // the metadata is not a tensor
        var loaded = SafeTensors.LoadFile(P("meta.safetensors"));
        Assert.Equal(new[] { "a" }, loaded.Keys);
        Assert.True(loaded["a"].Equals(tensor(new[] { 1f, 2f, 3f })));
    }

    [Fact]
    public void FileWithoutMetadata_HasAnEmptyMetadataDictionary()
    {
        SafeTensorsWriter.Save(P("plain.safetensors"), OneTensor());

        Assert.Empty(SafeTensors.ReadMetadata(P("plain.safetensors")));
    }

    [Fact]
    public void SameMetadataInAnyInsertionOrder_GivesTheSameBytes()
    {
        SafeTensorsWriter.Save(P("x.safetensors"), OneTensor(), new Dictionary<string, string> { ["b"] = "2", ["a"] = "1", ["c"] = "3" });
        SafeTensorsWriter.Save(P("y.safetensors"), OneTensor(), new Dictionary<string, string> { ["c"] = "3", ["a"] = "1", ["b"] = "2" });

        Assert.Equal(File.ReadAllBytes(P("x.safetensors")), File.ReadAllBytes(P("y.safetensors")));
    }
}
