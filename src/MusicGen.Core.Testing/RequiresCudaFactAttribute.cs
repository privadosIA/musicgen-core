using Xunit;

namespace MusicGen.Core.Testing;

/// <summary>[Fact] that only makes sense with a CUDA GPU; it is skipped with an explicit reason if there is none (e.g. with the CPU backend).</summary>
public sealed class RequiresCudaFactAttribute : FactAttribute
{
    public RequiresCudaFactAttribute()
    {
        if (!TorchSharp.torch.cuda.is_available())
            Skip = "no CUDA GPU available (CPU backend or no GPU)";
    }
}
