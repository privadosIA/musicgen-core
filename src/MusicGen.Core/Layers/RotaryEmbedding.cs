using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Layers;

/// <summary>How the channels of a head are paired for the rotation.</summary>
public enum RopeLayout
{
    /// <summary>Pairs <c>(x[2i], x[2i+1])</c>. torchtune (HeartMuLa, HeartCodec's flow transformer).</summary>
    AdjacentPairs,

    /// <summary>The two halves <c>(x[i], x[i + d/2])</c>, i.e. <c>x·cos + rotate_half(x)·sin</c>. HuggingFace Llama (DiffRhythm2).</summary>
    HalfRotation,
}

/// <summary>Where the cos/sin values come from and in which precision the rotation is computed.</summary>
public enum RopeTable
{
    /// <summary>
    /// A float32 table for <c>maxSeqLen</c> positions, built once per device (torchtune). The input is cast to float32, rotated, and cast back.
    /// Positions are gathered from the table, so they may be any values below <c>maxSeqLen</c>.
    /// NOTE: HuggingFace multiplies in the input dtype (its fp32 table is cast to the activation dtype first); with bf16 inputs this mode
    /// is therefore not bit-equal to HuggingFace. The two are equal for float32 inputs.
    /// </summary>
    Float32Cached,

    /// <summary>cos/sin computed on every call in the input dtype, and the rotation computed in that dtype (HeartCodec's flow transformer).</summary>
    InputDtypePerCall,
}

/// <summary>Llama 3.x frequency scaling parameters (torchtune's <c>Llama3ScaledRoPE</c>).</summary>
public sealed record Llama3Scaling(double ScaleFactor, double LowFreqFactor = 1, double HighFreqFactor = 4, double OldContextLen = 8192);

/// <summary>Inverse-frequency definition of a rotary embedding: <c>1 / base^(2i/d)</c>, optionally with Llama 3 scaling.</summary>
public sealed record RopeFrequencies(double Base, Llama3Scaling? Scaling = null)
{
    /// <summary>No scaling.</summary>
    public static RopeFrequencies Plain(double @base) => new(@base);

    /// <summary>Llama 3.x scaling.</summary>
    public static RopeFrequencies Llama3Scaled(double @base, double scaleFactor, double lowFreqFactor = 1, double highFreqFactor = 4, double oldContextLen = 8192) =>
        new(@base, new Llama3Scaling(scaleFactor, lowFreqFactor, highFreqFactor, oldContextLen));

    /// <summary>Inverse frequencies <c>[ropeDim/2]</c> in the given dtype.</summary>
    public Tensor InverseFrequencies(int ropeDim, ScalarType dtype, Device device)
    {
        var idx = arange(0, ropeDim, 2, dtype: dtype, device: device);
        var freqs = 1.0 / pow(Base, idx / ropeDim);
        return Scaling is { } s ? Scale(freqs, s) : freqs;
    }

    private static Tensor Scale(Tensor freqs, Llama3Scaling s)
    {
        double lowFreqWavelen = s.OldContextLen / s.LowFreqFactor;
        double highFreqWavelen = s.OldContextLen / s.HighFreqFactor;

        var wavelen = 2 * Math.PI / freqs;
        var smooth = (s.OldContextLen / wavelen - s.LowFreqFactor) / (s.HighFreqFactor - s.LowFreqFactor);
        var scaled = (1 - smooth) * freqs / s.ScaleFactor + smooth * freqs;

        return where(wavelen < highFreqWavelen, freqs,
               where(wavelen > lowFreqWavelen, freqs / s.ScaleFactor, scaled));
    }
}

/// <summary>
/// Rotary position embedding shared by the ports. Three choices are explicit because the models differ in each and mixing them up changes
/// the output without any error: the <see cref="RopeLayout"/> (required, no default), the <see cref="RopeFrequencies"/> and the
/// <see cref="RopeTable"/> precision. Optionally only the first <c>ropeDim</c> channels of each head are rotated.
/// The cos/sin cache is not part of any checkpoint (torchtune's <c>persistent=False</c>), so it is a plain field.
/// </summary>
public sealed class RotaryEmbedding
{
    private readonly RopeLayout layout;
    private readonly RopeFrequencies frequencies;
    private readonly RopeTable table;
    private readonly int headDim;
    private readonly int ropeDim;
    private readonly int maxSeqLen;
    private Tensor? cache;          // [maxSeqLen, ropeDim/2, 2] float32 (cos, sin); Float32Cached only
    private Device? cacheDevice;

    /// <param name="layout">Required: there is deliberately no default.</param>
    /// <param name="headDim">Channels per head of the tensors given to <see cref="Apply"/>.</param>
    /// <param name="maxSeqLen">Size of the table; required (&gt; 0) for <see cref="RopeTable.Float32Cached"/>.</param>
    /// <param name="ropeDim">Number of leading channels to rotate (even); all of them by default.</param>
    public RotaryEmbedding(RopeLayout layout, RopeFrequencies frequencies, RopeTable table, int headDim, int maxSeqLen = 0, int? ropeDim = null)
    {
        this.ropeDim = ropeDim ?? headDim;
        if (this.ropeDim % 2 != 0 || this.ropeDim <= 0 || this.ropeDim > headDim)
            throw new ArgumentException($"ropeDim must be even and within 1..headDim (ropeDim {this.ropeDim}, headDim {headDim})");
        if (table == RopeTable.Float32Cached && maxSeqLen <= 0)
            throw new ArgumentException("the cached float32 table needs maxSeqLen > 0");
        this.layout = layout;
        this.frequencies = frequencies;
        this.table = table;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
    }

    /// <param name="x"><c>[b, s, heads, headDim]</c> when <paramref name="seqDim"/> is 1, <c>[b, heads, s, headDim]</c> when it is 2.</param>
    /// <param name="positions">
    /// Absolute positions, <c>[b, s]</c> long. Required for <see cref="RopeTable.Float32Cached"/>. For <see cref="RopeTable.InputDtypePerCall"/>
    /// it is optional (positions <c>0..s−1</c> when null) and may also be <c>[s]</c>.
    /// </param>
    public Tensor Apply(Tensor x, Tensor? positions, int seqDim)
    {
        if (seqDim != 1 && seqDim != 2) throw new ArgumentException("seqDim must be 1 ([b, s, h, d]) or 2 ([b, h, s, d])");
        if (x.dim() != 4 || x.shape[3] != headDim) throw new ArgumentException($"x must be 4-D with headDim {headDim} (shape {string.Join("x", x.shape)})");
        using var scope = NewDisposeScope();
        long b = x.shape[0], s = x.shape[seqDim];
        int half = ropeDim / 2;

        Tensor cos, sin, work;
        if (table == RopeTable.Float32Cached)
        {
            if (positions is null) throw new ArgumentException("the cached float32 table needs explicit positions");
            var c = GetCache(x.device);
            var gathered = c.index_select(0, positions.reshape(-1)).view(b, s, half, 2);
            cos = Broadcastable(gathered.select(-1, 0), seqDim);
            sin = Broadcastable(gathered.select(-1, 1), seqDim);
            work = x.to(ScalarType.Float32);
        }
        else
        {
            (sin, cos) = PerCall(positions, s, x.device, x.dtype);
            cos = Broadcastable(cos, seqDim);
            sin = Broadcastable(sin, seqDim);
            work = x;
        }

        Tensor result;
        if (ropeDim == headDim)
        {
            result = Rotate(work, cos, sin);
        }
        else
        {
            var head = work[.., .., .., ..ropeDim];
            var tail = work[.., .., .., ropeDim..];
            result = cat(new[] { Rotate(head, cos, sin), tail }, dim: -1);
        }
        if (table == RopeTable.Float32Cached) result = result.to(x.dtype);
        return result.MoveToOuterDisposeScope();
    }

    /// <summary>[b or 1, s, half] → broadcastable against the 4-D tensor: [b, s, 1, half] or [b, 1, s, half].</summary>
    private static Tensor Broadcastable(Tensor t, int seqDim)
    {
        if (t.dim() == 2) t = t.unsqueeze(0);
        return seqDim == 1 ? t.unsqueeze(2) : t.unsqueeze(1);
    }

    private Tensor Rotate(Tensor v, Tensor cos, Tensor sin)
    {
        int half = ropeDim / 2;
        var lead = v.shape[..^1];
        if (layout == RopeLayout.AdjacentPairs)
        {
            var pairs = v.reshape(lead.Concat(new long[] { half, 2 }).ToArray());
            var x0 = pairs.select(-1, 0);
            var x1 = pairs.select(-1, 1);
            return stack(new[] { x0 * cos - x1 * sin, x1 * cos + x0 * sin }, dim: -1).flatten(3);
        }
        else
        {
            var x0 = v.narrow(-1, 0, half);
            var x1 = v.narrow(-1, half, half);
            return cat(new[] { x0 * cos - x1 * sin, x1 * cos + x0 * sin }, dim: -1);
        }
    }

    /// <summary>(sin, cos) <c>[s, half]</c> (or <c>[b, s, half]</c> with 2-D positions) in <paramref name="dtype"/>.</summary>
    private (Tensor sin, Tensor cos) PerCall(Tensor? positions, long s, Device device, ScalarType dtype)
    {
        using var scope = NewDisposeScope();
        var invFreq = frequencies.InverseFrequencies(ropeDim, dtype, device);
        Tensor freqs;
        if (positions is null)
        {
            freqs = einsum("i,j->ij", arange(s, dtype: dtype, device: device), invFreq);
        }
        else
        {
            var t = positions.to(dtype);
            freqs = t.dim() == 1 ? einsum("i,j->ij", t, invFreq) : einsum("bi,j->bij", t, invFreq);
        }
        return (freqs.sin().MoveToOuterDisposeScope(), freqs.cos().MoveToOuterDisposeScope());
    }

    private Tensor GetCache(Device device)
    {
        if (cache is not null && cacheDevice is not null && cacheDevice.type == device.type && cacheDevice.index == device.index)
            return cache;

        cache?.Dispose();
        using var scope = NewDisposeScope();
        var theta = frequencies.InverseFrequencies(ropeDim, ScalarType.Float32, device);
        var seqIdx = arange(maxSeqLen, dtype: ScalarType.Float32, device: device);
        var idxTheta = einsum("i,j->ij", seqIdx, theta);
        // Detach, not MoveToOuter: the outer scope is Apply's, which would free the cache when the first call returns.
        cache = stack(new[] { cos(idxTheta), sin(idxTheta) }, dim: -1).DetachFromDisposeScope();
        cacheDevice = device;
        return cache;
    }
}
