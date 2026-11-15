using static TorchSharp.torch;
using TorchSharp;

namespace MusicGen.Core.Testing;

public static class Rng
{
    /// <summary>Seeds the CPU and CUDA RNGs (the same function the generation commands use).</summary>
    public static void Seed(long seed) => Seeding.Apply(seed);

    /// <summary>Advances the global RNG the same as a real sampling step: one exponential_ over a bf16 [1, V].</summary>
    public static void Advance(long vocab)
    {
        using var noise = empty(new long[] { 1, vocab }, dtype: ScalarType.BFloat16).exponential_(1);
    }
}
