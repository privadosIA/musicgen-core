using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace MusicGen.Core.Layers;

/// <summary>
/// SwiGLU feed-forward: <c>down(silu(gate(x)) · up(x))</c> with bias-free <c>gate</c>, <c>up</c> and <c>down</c> projections (the checkpoint key names).
/// By default the hidden size is <c>int(2·4·dim/3)</c> rounded up to a multiple of 256 (HeartCodec); pass <paramref name="hiddenSize"/> for models that
/// state it directly (HuggingFace Llama: <c>intermediate_size</c>).
/// </summary>
public sealed class FlowMlp : Module<Tensor, Tensor>
{
    private readonly Linear gate;
    private readonly Linear up;
    private readonly Linear down;

    public static int HiddenSize(int dim, int multipleOf = 256)
    {
        int hidden = (int)(2 * (4 * dim) / 3.0);
        return multipleOf * ((hidden + multipleOf - 1) / multipleOf);
    }

    public FlowMlp(int dim, int? hiddenSize = null) : base(nameof(FlowMlp))
    {
        int hidden = hiddenSize ?? HiddenSize(dim);
        if (hidden <= 0) throw new ArgumentException($"hiddenSize must be positive (got {hidden})");
        gate = Linear(dim, hidden, hasBias: false);
        up = Linear(dim, hidden, hasBias: false);
        down = Linear(hidden, dim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        return down.forward(F.silu(gate.forward(x)) * up.forward(x)).MoveToOuterDisposeScope();
    }
}
