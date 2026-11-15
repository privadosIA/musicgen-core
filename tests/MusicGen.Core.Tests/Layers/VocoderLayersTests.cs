using MusicGen.Core.Layers;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MusicGen.Core.Tests;

/// <summary>
/// Vocoder layers against naive loops written here (independent of TorchSharp), in small cases:
/// causal and symmetric convolution with dilation, causal transposed convolution with trimming, and PReLU. It does not need a checkpoint.
/// </summary>
public class VocoderLayersTests
{
    private static float[] Rand(int n, int seed)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
    }

    private static void SetParam(Module m, string name, float[] data, params long[] shape)
    {
        using var _ = no_grad();
        var p = m.named_parameters().Single(x => x.name == name).parameter;
        p.copy_(tensor(data).reshape(shape));
    }

    private static float[] ToArray(Tensor t) => t.contiguous().data<float>().ToArray();

    // x: [C, T] flat; w: [O, C, K] flat. Returns [O, T] flat with `pad` padding on the left and `padRight` on the right (only for the length).
    private static float[] NaiveConv(float[] x, int c, int t, float[] w, int o, int k, float[] b, int dilation, int leftPad, int outLen)
    {
        var y = new float[o * outLen];
        for (int oc = 0; oc < o; oc++)
            for (int ti = 0; ti < outLen; ti++)
            {
                double acc = b[oc];
                for (int ci = 0; ci < c; ci++)
                    for (int j = 0; j < k; j++)
                    {
                        int src = ti - leftPad + j * dilation;
                        if (src >= 0 && src < t) acc += (double)w[(oc * c + ci) * k + j] * x[ci * t + src];
                    }
                y[oc * outLen + ti] = (float)acc;
            }
        return y;
    }

    [Theory]
    [InlineData(3, 1)]
    [InlineData(7, 1)]
    [InlineData(7, 3)]
    [InlineData(7, 9)]
    [InlineData(1, 1)]
    public void CausalConv_MatchesTheNaiveLoop_AndNeverLooksAhead(int k, int d)
    {
        const int c = 3, o = 4, t = 40;
        var layer = new Conv1dLayer(c, o, k, d, causal: true);
        var w = Rand(o * c * k, 1); var b = Rand(o, 2); var x = Rand(c * t, 3);
        SetParam(layer, "weight", w, o, c, k); SetParam(layer, "bias", b, o);

        using var y = layer.forward(tensor(x).reshape(1, c, t));

        Assert.Equal(new long[] { 1, o, t }, y.shape);                       // same length
        var expected = NaiveConv(x, c, t, w, o, k, b, d, leftPad: d * (k - 1), outLen: t);
        Assert.True(expected.Zip(ToArray(y), (e, g) => Math.Abs(e - g)).Max() < 1e-5f, "differs from the naive loop");

        // Causality: altering the input from instant s does not change the output before s.
        const int s = 25;
        var x2 = (float[])x.Clone();
        for (int ci = 0; ci < c; ci++) for (int ti = s; ti < t; ti++) x2[ci * t + ti] += 5f;
        using var y2 = layer.forward(tensor(x2).reshape(1, c, t));
        var a = ToArray(y); var a2 = ToArray(y2);
        for (int oc = 0; oc < o; oc++) for (int ti = 0; ti < s; ti++) Assert.Equal(a[oc * t + ti], a2[oc * t + ti]);
    }

    [Theory]
    [InlineData(5, 1)]
    [InlineData(7, 3)]
    public void SymmetricConv_MatchesTheNaiveLoop_AndLooksAheadHalfTheKernel(int k, int d)
    {
        const int c = 2, o = 3, t = 30;
        int pad = (k * d - d) / 2;                                            // heartlib's get_padding
        var layer = new Conv1dLayer(c, o, k, d, causal: false);
        var w = Rand(o * c * k, 4); var b = Rand(o, 5); var x = Rand(c * t, 6);
        SetParam(layer, "weight", w, o, c, k); SetParam(layer, "bias", b, o);

        using var y = layer.forward(tensor(x).reshape(1, c, t));

        Assert.Equal(new long[] { 1, o, t }, y.shape);
        var expected = NaiveConv(x, c, t, w, o, k, b, d, leftPad: pad, outLen: t);
        Assert.True(expected.Zip(ToArray(y), (e, g) => Math.Abs(e - g)).Max() < 1e-5f, "differs from the naive loop");

        // Looks `pad` instants ahead: altering from s changes the output from s - pad, and not before.
        const int s = 20;
        var x2 = (float[])x.Clone();
        for (int ci = 0; ci < c; ci++) x2[ci * t + s] += 5f;
        using var y2 = layer.forward(tensor(x2).reshape(1, c, t));
        var a = ToArray(y); var a2 = ToArray(y2);
        for (int ti = 0; ti < s - pad; ti++) Assert.Equal(a[ti], a2[ti]);
        Assert.NotEqual(a[s - pad], a2[s - pad]);
    }

    [Theory]
    [InlineData(5)]
    [InlineData(4)]
    [InlineData(3)]
    public void CausalTransposedConv_MatchesTheNaiveLoop_WithKernelTwiceTheStride(int stride)
    {
        const int c = 3, o = 2, t = 12;
        int k = 2 * stride;
        var layer = new ConvTranspose1dLayer(c, o, k, stride);
        var w = Rand(c * o * k, 7); var b = Rand(o, 8); var x = Rand(c * t, 9);
        SetParam(layer, "weight", w, c, o, k); SetParam(layer, "bias", b, o);

        using var y = layer.forward(tensor(x).reshape(1, c, t));

        Assert.Equal(new long[] { 1, o, t * stride }, y.shape);               // exactly T·stride after trimming `stride`
        var expected = new float[o * t * stride];                              // y[o,p] = b + Σ x[c,i]·w[c,o,j] with i·stride + j = p
        for (int oc = 0; oc < o; oc++)
            for (int p = 0; p < t * stride; p++)
            {
                double acc = b[oc];
                for (int ci = 0; ci < c; ci++)
                    for (int i = 0; i < t; i++)
                    {
                        int j = p - i * stride;
                        if (j >= 0 && j < k) acc += (double)x[ci * t + i] * w[(ci * o + oc) * k + j];
                    }
                expected[oc * t * stride + p] = (float)acc;
            }
        Assert.True(expected.Zip(ToArray(y), (e, g) => Math.Abs(e - g)).Max() < 1e-5f, "differs from the naive loop");
    }

    [Theory]
    [InlineData(20, 10)]
    [InlineData(16, 8)]
    [InlineData(7, 3)]
    [InlineData(4, 2)]
    public void NonCausalTransposedConv_HasLengthAndValuesOfTheNaiveLoop(int k, int stride)
    {
        const int c = 3, o = 2, t = 9;
        int pad = (k - stride) / 2;
        var layer = new ConvTranspose1dLayer(c, o, k, stride, causal: false);
        var w = Rand(c * o * k, 17); var b = Rand(o, 18); var x = Rand(c * t, 19);
        SetParam(layer, "weight", w, c, o, k); SetParam(layer, "bias", b, o);

        using var y = layer.forward(tensor(x).reshape(1, c, t));

        long length = (long)(t - 1) * stride + k - 2 * pad;                   // (T − 1)·s + k − 2p
        Assert.Equal(new long[] { 1, o, length }, y.shape);
        Assert.Equal((long)t * stride, length);                               // k − s is even, so exactly T·s
        var expected = new float[o * length];                                 // the full transposed convolution, cropped by `pad` on both sides
        for (int oc = 0; oc < o; oc++)
            for (long q = 0; q < length; q++)
            {
                double acc = b[oc];
                for (int ci = 0; ci < c; ci++)
                    for (int i = 0; i < t; i++)
                    {
                        long j = q + pad - (long)i * stride;
                        if (j >= 0 && j < k) acc += (double)x[ci * t + i] * w[(ci * o + oc) * k + (int)j];
                    }
                expected[oc * length + q] = (float)acc;
            }
        Assert.True(expected.Zip(ToArray(y), (e, g) => Math.Abs(e - g)).Max() < 1e-5f, "differs from the naive loop");
    }

    [Theory]
    [InlineData(7, 4)]    // odd difference
    [InlineData(3, 5)]    // kernel shorter than the stride
    public void NonCausalTransposedConv_RejectsAnUnevenOrNegativeOverlap(int k, int stride) =>
        Assert.Throws<ArgumentException>(() => new ConvTranspose1dLayer(2, 2, k, stride, causal: false));

    [Fact]
    public void TheDefaultMode_IsStillCausal_AndAnExplicitCausalTrueIsTheSame()
    {
        const int c = 2, o = 2, t = 6, stride = 5, k = 10;
        var byDefault = new ConvTranspose1dLayer(c, o, k, stride);
        var explicitCausal = new ConvTranspose1dLayer(c, o, k, stride, causal: true);
        var w = Rand(c * o * k, 27); var b = Rand(o, 28); var x = Rand(c * t, 29);
        foreach (var layer in new[] { byDefault, explicitCausal }) { SetParam(layer, "weight", w, c, o, k); SetParam(layer, "bias", b, o); }

        using var y1 = byDefault.forward(tensor(x).reshape(1, c, t));
        using var y2 = explicitCausal.forward(tensor(x).reshape(1, c, t));

        Assert.Equal(new long[] { 1, o, t * stride }, y1.shape);
        Assert.True(y1.Equals(y2));
        Assert.Throws<ArgumentException>(() => new ConvTranspose1dLayer(c, o, kernel: 16, stride: 10));   // causal needs kernel = 2·stride
    }

    [Fact]
    public void CausalTransposedConv_RejectsAKernelThatIsNotTwiceTheStride() =>
        Assert.Throws<ArgumentException>(() => new ConvTranspose1dLayer(2, 2, kernel: 7, stride: 3));

    [Fact]
    public void PReLU_UsesOneSharedParameter()
    {
        var layer = new PReLULayer();
        SetParam(layer, "weight", new[] { 0.1f }, 1);
        var x = new[] { -2f, -0.5f, 0f, 0.5f, 3f };

        using var y = layer.forward(tensor(x).reshape(1, 1, 5));

        Assert.Equal(x.Select(v => v >= 0 ? v : 0.1f * v), ToArray(y).Select(v => (float)Math.Round(v, 6)));
        Assert.Equal(new[] { "weight" }, layer.named_parameters().Select(p => p.name));
        Assert.Equal(new long[] { 1 }, layer.named_parameters().Single().parameter.shape);
    }

    [Fact]
    public void LayersRegisterExactlyWeightAndBias_WithTheCheckpointShapes()
    {
        var conv = new Conv1dLayer(3, 5, 7);
        var up = new ConvTranspose1dLayer(6, 4, 10, 5);

        Assert.Equal(new[] { "bias", "weight" }, conv.named_parameters().Select(p => p.name).Order());
        Assert.Equal(new long[] { 5, 3, 7 }, conv.named_parameters().Single(p => p.name == "weight").parameter.shape);   // [out, in, k]
        Assert.Equal(new long[] { 6, 4, 10 }, up.named_parameters().Single(p => p.name == "weight").parameter.shape);    // [in, out, k]
    }
}
