using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MusicGen.Core.Layers;

/// <summary>
/// torchtune's RMSNorm: computed in float32 and cast back to the input dtype; the parameter is called <c>scale</c>, not <c>weight</c>.
/// There are three RMSNorm flavors across the ports and they must not be swapped: this one (<c>scale</c>, fp32 round trip),
/// <see cref="FlowRmsNorm"/> (<c>weight</c>, input dtype) and HuggingFace's <c>LlamaRMSNorm</c> (<c>weight</c>, fp32; not here yet).
/// </summary>
public sealed class RmsNorm : Module<Tensor, Tensor>
{
    private readonly Parameter scale;
    private readonly double eps;

    public RmsNorm(int dim, double eps) : base(nameof(RmsNorm))
    {
        this.eps = eps;
        scale = new Parameter(torch.ones(dim));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // torchtune computes in float32 and goes back to the original dtype
        using var _ = NewDisposeScope();
        var xf = x.to(ScalarType.Float32);
        var normed = xf * torch.rsqrt(xf.pow(2).mean(new long[] { -1 }, keepdim: true) + eps);
        return (normed.to(x.dtype) * scale).MoveToOuterDisposeScope();
    }
}
