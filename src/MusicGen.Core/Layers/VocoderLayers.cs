using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace MusicGen.Core.Layers;

// HeartCodec vocoder layers (heartlib's sq_codec.py). They register `weight` and `bias` directly (without wrapping an nn.Conv1d, which would add
// a `conv.weight` level) so that the checkpoint key with the weight_norm already folded matches the model's with no mapping table.

/// <summary>
/// 1D convolution. Causal: padding only on the left of <c>dilation·(k−1)</c>, so each output sample only looks at the current and past input.
/// Non-causal: symmetric padding <c>(k·d−d)/2</c> (heartlib's get_padding). Weights <c>[out, in, k]</c>.
/// </summary>
public sealed class Conv1dLayer : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter bias;
    private readonly int dilation;
    private readonly int leftPad;     // causal only
    private readonly int symPad;      // non-causal only
    private readonly bool causal;

    public Conv1dLayer(int inChannels, int outChannels, int kernel, int dilation = 1, bool causal = false) : base(nameof(Conv1dLayer))
    {
        this.dilation = dilation;
        this.causal = causal;
        leftPad = causal ? dilation * (kernel - 1) : 0;
        symPad = causal ? 0 : (kernel * dilation - dilation) / 2;
        weight = new Parameter(empty(outChannels, inChannels, kernel), requires_grad: false);
        bias = new Parameter(zeros(outChannels), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        if (!causal) return F.conv1d(x, weight, bias, stride: 1, padding: symPad, dilation: dilation);
        using var padded = F.pad(x, new long[] { leftPad, 0 });
        return F.conv1d(padded, weight, bias, stride: 1, padding: 0, dilation: dilation);
    }
}

/// <summary>
/// 1D transposed convolution. Weights <c>[in, out, k]</c>.
/// Causal (default, HeartCodec): <c>kernel = 2·stride</c>, padding 0 and the last <c>stride</c> samples are trimmed, so the output has exactly
/// <c>T·stride</c> samples and each one only depends on the current and the previous input.
/// Non-causal (BigVGAN): symmetric padding <c>p = (kernel − stride) / 2</c>, output length <c>(T − 1)·stride + kernel − 2p</c>, which is
/// <c>T·stride</c> when <c>kernel − stride</c> is even.
/// </summary>
public sealed class ConvTranspose1dLayer : Module<Tensor, Tensor>
{
    private readonly Parameter weight;
    private readonly Parameter bias;
    private readonly int stride;
    private readonly int padding;
    private readonly bool causal;

    public ConvTranspose1dLayer(int inChannels, int outChannels, int kernel, int stride, bool causal = true) : base(nameof(ConvTranspose1dLayer))
    {
        if (causal && kernel != 2 * stride) throw new ArgumentException($"the causal transposed convolution requires kernel = 2·stride (kernel {kernel}, stride {stride})");
        if (!causal && (kernel < stride || (kernel - stride) % 2 != 0))
            throw new ArgumentException($"the non-causal transposed convolution requires kernel ≥ stride and kernel − stride even (kernel {kernel}, stride {stride})");
        this.stride = stride;
        this.causal = causal;
        padding = causal ? 0 : (kernel - stride) / 2;
        weight = new Parameter(empty(inChannels, outChannels, kernel), requires_grad: false);
        bias = new Parameter(zeros(outChannels), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var y = F.conv_transpose1d(x, weight, bias, stride: stride, padding: padding);
        if (!causal) return y.MoveToOuterDisposeScope();
        return y.narrow(2, 0, y.shape[2] - stride).contiguous().MoveToOuterDisposeScope();
    }
}

/// <summary><c>PReLU</c> with a single shared parameter (<c>weight</c> of shape <c>[1]</c>).</summary>
public sealed class PReLULayer : Module<Tensor, Tensor>
{
    private readonly Parameter weight;

    public PReLULayer() : base(nameof(PReLULayer))
    {
        weight = new Parameter(full(new long[] { 1 }, 0.25), requires_grad: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => F.prelu(x, weight);
}
