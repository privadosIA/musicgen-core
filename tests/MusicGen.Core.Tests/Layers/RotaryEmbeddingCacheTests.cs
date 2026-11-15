using MusicGen.Core.Layers;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public class RotaryEmbeddingCacheTests
{
    [Fact]
    public void Apply_CalledRepeatedly_KeepsTheCacheAlive_AndIsDeterministic()
    {
        var rope = new RotaryEmbedding(RopeLayout.AdjacentPairs, RopeFrequencies.Llama3Scaled(10_000, 32), RopeTable.Float32Cached, 8, 16);
        var x = randn(new long[] { 1, 2, 2, 8 }, generator: new Generator(7));
        var pos = arange(2, dtype: ScalarType.Int64).reshape(1, 2);

        var first = rope.Apply(x, pos, 1);
        for (int call = 2; call <= 4; call++)
        {
            var again = rope.Apply(x, pos, 1);
            Assert.True(first.Equals(again), $"call {call} differs from the first");
        }
    }
}
