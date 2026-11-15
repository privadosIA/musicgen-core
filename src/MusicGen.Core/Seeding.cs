using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core;

/// <summary>
/// Seeds the RNG so a generation is reproducible. Unlike PyTorch in Python, TorchSharp's <c>torch.manual_seed</c>
/// only seeds the CPU generator: on GPU you must also seed <c>torch.cuda.manual_seed_all</c>. Without that, on CUDA
/// two generations with the same seed gave 126 of 128 different tokens from the first frame.
/// </summary>
public static class Seeding
{
    /// <summary>Seeds the CPU and, if a CUDA GPU is available, all GPUs.</summary>
    public static void Apply(long seed)
    {
        manual_seed(seed);
        if (cuda.is_available()) cuda.manual_seed_all(seed);
    }

    /// <summary>Non-negative random 32-bit seed (short, so it is easy to read and type).</summary>
    public static long NewSeed() => Random.Shared.NextInt64(0, 1L << 32);
}
