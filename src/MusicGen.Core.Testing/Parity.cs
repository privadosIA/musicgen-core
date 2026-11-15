using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Testing;

/// <summary>Metrics and tolerances for parity with Python (model-parity spec).</summary>
public static class Parity
{
    public const double MinCosSim = 0.999;
    public const double MaxAbsLogits = 0.5;

    public readonly record struct Result(double CosSim, double MaxAbs)
    {
        public override string ToString() => $"cos_sim={CosSim:F6} max_abs={MaxAbs:E3}";
    }

    /// <summary>cos_sim and max_abs between two tensors with the same number of elements, computed on CPU in double precision.</summary>
    public static Result Compare(Tensor got, Tensor expected)
    {
        using var _ = no_grad();
        var g = got.detach().cpu().to(ScalarType.Float64).flatten();
        var e = expected.detach().cpu().to(ScalarType.Float64).flatten();
        if (g.shape[0] != e.shape[0])
            throw new InvalidOperationException($"shapes differ: {string.Join("x", got.shape)} vs {string.Join("x", expected.shape)}");
        var cos = ((g * e).sum() / (g.norm() * e.norm())).item<double>();
        var maxAbs = (g - e).abs().max().item<double>();
        return new Result(cos, maxAbs);
    }
}
