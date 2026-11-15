using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Weights;

/// <summary>
/// Folding of <c>weight_norm</c>. The HeartCodec checkpoint stores each weight as <c>parametrizations.weight.original0</c> (g, shape
/// <c>[C,1,1]</c>) and <c>original1</c> (v); the effective weight is <c>w = v · (g / ‖v‖)</c> with the norm over all dimensions except 0
/// (the default <c>weight_norm</c> dimension, also in <c>ConvTranspose1d</c>, whose weight is <c>[in, out, k]</c>). It is folded at load time,
/// in fp32, so that the model only has plain weights.
/// </summary>
public static class WeightNorm
{
    public static Tensor Fold(Tensor g, Tensor v)
    {
        using var scope = NewDisposeScope();
        if (g.dim() != v.dim() || g.shape[0] != v.shape[0] || g.numel() != g.shape[0])
            throw new ArgumentException($"g {Describe(g)} is not a per-channel scale of v {Describe(v)}");

        var gf = g.to(ScalarType.Float32);
        var vf = v.to(ScalarType.Float32);
        var dims = Enumerable.Range(1, (int)v.dim() - 1).Select(i => (long)i).ToArray();
        var norm = vf.pow(2).sum(dims, keepdim: true).sqrt();           // [C, 1, 1]
        return (vf * (gf / norm)).MoveToOuterDisposeScope();            // the same order as torch._weight_norm
    }

    private static string Describe(Tensor t) => "[" + string.Join(", ", t.shape) + "]";
}
