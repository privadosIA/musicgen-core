using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace MusicGen.Core.Layers;

/// <summary>
/// Multi-head attention with bias-free q/k/v/o projections (the checkpoint key names), a caller-chosen <see cref="RotaryEmbedding"/> over the full
/// head dimension and non-causal scaled-dot-product attention. Without a mask and without <c>kvInput</c> it is self-attention over all positions.
/// </summary>
public sealed class FlowAttention : Module<Tensor, Tensor>
{
    private readonly Linear q_proj;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear o_proj;
    private readonly int heads;
    private readonly int headDim;
    private readonly RotaryEmbedding rope;

    /// <param name="rope">Required: the rotary embedding is part of what a model is, so there is no default. It must have been built for <paramref name="headDim"/>.</param>
    public FlowAttention(int dim, int heads, int headDim, RotaryEmbedding rope) : base(nameof(FlowAttention))
    {
        this.heads = heads;
        this.headDim = headDim;
        this.rope = rope;
        int inner = heads * headDim;
        q_proj = Linear(dim, inner, hasBias: false);
        k_proj = Linear(dim, inner, hasBias: false);
        v_proj = Linear(dim, inner, hasBias: false);
        o_proj = Linear(inner, dim, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => Forward(x);

    /// <param name="x"><c>[B, T, dim]</c> (the queries).</param>
    /// <param name="mask">Optional boolean mask broadcastable to <c>[B, 1, T, Tk]</c>, true = may attend (as torchtune and torch SDPA).</param>
    /// <param name="kvInput">
    /// Optional <c>[B, Tk, dim]</c> that supplies the keys and values (cross-attention). Positions of two different sequences share no frame, so
    /// no rotary embedding is applied in that case.
    /// </param>
    public Tensor Forward(Tensor x, Tensor? mask = null, Tensor? kvInput = null)
    {
        using var scope = NewDisposeScope();
        long b = x.shape[0], t = x.shape[1];
        var kvSource = kvInput ?? x;
        long tk = kvSource.shape[1];
        Tensor Shape(Tensor y, long len) => y.view(b, len, heads, headDim).transpose(1, 2);
        var q = Shape(q_proj.forward(x), t);
        var k = Shape(k_proj.forward(kvSource), tk);
        var v = Shape(v_proj.forward(kvSource), tk);
        if (kvInput is null)
        {
            q = rope.Apply(q, null, seqDim: 2);
            k = rope.Apply(k, null, seqDim: 2);
        }
        var attn = mask is null ? F.scaled_dot_product_attention(q, k, v) : F.scaled_dot_product_attention(q, k, v, attn_mask: mask);
        return o_proj.forward(attn.transpose(1, 2).contiguous().view(b, t, heads * headDim)).MoveToOuterDisposeScope();
    }
}
