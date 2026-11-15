using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public class SafeTensorsRoundTripTests
{
    [Fact]
    public void WriteThenLoad_PreservesDtypeAndValues()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.safetensors");
        try
        {
            var bf = arange(12, dtype: ScalarType.Float32).reshape(3, 4).to(ScalarType.BFloat16);
            var f32 = tensor(new[] { 1.5f, -2f, 3.25f });
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["a.bf16"] = bf, ["b.f32"] = f32 });

            var loaded = SafeTensors.LoadFile(path);

            Assert.Equal(ScalarType.BFloat16, loaded["a.bf16"].dtype);
            Assert.Equal(new long[] { 3, 4 }, loaded["a.bf16"].shape);
            Assert.True(loaded["a.bf16"].to(ScalarType.Float32).Equals(bf.to(ScalarType.Float32)));
            Assert.Equal(ScalarType.Float32, loaded["b.f32"].dtype);
            Assert.True(loaded["b.f32"].Equals(f32));
        }
        finally { File.Delete(path); }
    }
}
