using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MusicGen.Core.Layers;

/// <summary>RMSNorm with a <c>weight</c> and eps 1e-6, computed in the input dtype (no cast to fp32), as in the reference.
/// Not interchangeable with <see cref="RmsNorm"/> (parameter <c>scale</c>, fp32 round trip).</summary>
public sealed class FlowRmsNorm : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly double eps;

    public FlowRmsNorm(int dim, double eps = 1e-6) : base(nameof(FlowRmsNorm))
    {
        this.eps = eps;
        weight = new Parameter(ones(dim), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var variance = x.pow(2).mean(new long[] { -1 }, keepdim: true);
        return (weight * (x * rsqrt(variance + eps))).MoveToOuterDisposeScope();
    }
}
