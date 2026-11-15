using MusicGen.Core.Audio;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

/// <summary>The cross-faded assembly against a naive double-precision loop, with a small, non-HeartCodec layout (window 1000, hop 750, overlap 250).</summary>
public class WindowAssemblerTests
{
    private const int W = 1000, Hop = 750, Ovlp = 250;

    private static float Sample(int window, int channel, int t) => (float)Math.Sin(0.013 * t + window * 1.3 + channel * 0.5);

    private static Tensor MakeWindow(int window, int channels = 2)
    {
        var data = new float[channels * W];
        for (int c = 0; c < channels; c++)
            for (int t = 0; t < W; t++) data[c * W + t] = Sample(window, c, t);
        return tensor(data, new long[] { channels, W });
    }

    [Fact]
    public void Ramp_is_numpy_linspace()
    {
        Assert.Equal(new[] { 0.0, 0.25, 0.5, 0.75, 1.0 }, WindowAssembler.Ramp(5));
        Assert.Equal(new[] { 0.0 }, WindowAssembler.Ramp(1));
        var big = WindowAssembler.Ramp(Ovlp);
        Assert.Equal(0.0, big[0]);
        Assert.Equal(1.0, big[^1]);
        Assert.True(Enumerable.Range(1, Ovlp - 1).All(i => big[i] > big[i - 1]));
    }

    [Theory]
    [InlineData(3, 2200)]    // trimmed: the untrimmed length is 1000 + 2 * 750 = 2500
    [InlineData(3, 2500)]    // not trimmed
    [InlineData(2, 1100)]
    [InlineData(1, 640)]
    public void Windows_are_joined_with_a_linear_cross_fade_and_trimmed(int windows, int target)
    {
        using var assembler = new WindowAssembler(W, Hop, Ovlp, windows, target);
        for (int k = 0; k < windows; k++) { using var w = MakeWindow(k); assembler.Add(w); }
        using var result = assembler.Finish();

        Assert.Equal(new long[] { 2, target }, result.shape);
        var got = result.data<float>().ToArray();
        var ramp = WindowAssembler.Ramp(Ovlp);
        double worst = 0;
        for (int c = 0; c < 2; c++)
            for (long t = 0; t < target; t++)
            {
                int k = (int)Math.Min(windows - 1, t / Hop);                  // window whose own samples cover t (the last one takes the tail)
                long local = t - (long)k * Hop;
                double expected = k > 0 && local < Ovlp
                    ? Sample(k - 1, c, (int)(local + Hop)) * (1 - ramp[local]) + (double)Sample(k, c, (int)local) * ramp[local]
                    : Sample(k, c, (int)local);
                worst = Math.Max(worst, Math.Abs(got[c * target + t] - expected));
            }
        Assert.True(worst < 1e-6, $"max difference {worst:E2}");
    }

    [Fact]
    public void Samples_outside_the_overlaps_are_bit_exact_and_the_overlap_edges_are_the_pure_signals()
    {
        using var assembler = new WindowAssembler(W, Hop, Ovlp, 2, 1750);
        using (var w = MakeWindow(0)) assembler.Add(w);
        using (var w = MakeWindow(1)) assembler.Add(w);
        using var result = assembler.Finish();
        var got = result.data<float>().ToArray();
        long len = result.shape[1];

        Assert.Equal(Sample(0, 1, 123), got[len + 123]);                      // untouched head of the first window
        Assert.Equal(Sample(1, 0, Ovlp + 77), got[Hop + Ovlp + 77]);          // untouched part of the second window, after the overlap
        Assert.Equal(Sample(0, 0, Hop), got[Hop]);                            // ramp 0: only the old window
        Assert.Equal(Sample(1, 0, Ovlp - 1), got[Hop + Ovlp - 1], 6);         // ramp 1: only the new window
    }

    [Fact]
    public void A_mono_layout_works()
    {
        using var assembler = new WindowAssembler(W, Hop, Ovlp, 2, 1750, channels: 1);
        using (var w = MakeWindow(0, 1)) assembler.Add(w);
        using (var w = MakeWindow(1, 1)) assembler.Add(w);
        using var result = assembler.Finish();
        Assert.Equal(new long[] { 1, 1750 }, result.shape);
    }

    [Fact]
    public void Misuse_is_rejected()
    {
        using var assembler = new WindowAssembler(W, Hop, Ovlp, 2, 1750);
        using var wrong = zeros(new long[] { 2, 100 });
        Assert.Throws<ArgumentException>(() => assembler.Add(wrong));
        Assert.Throws<InvalidOperationException>(() => assembler.Finish());   // no window added yet
        using (var w = MakeWindow(0)) assembler.Add(w);
        using (var w = MakeWindow(1)) assembler.Add(w);
        using var extra = zeros(new long[] { 2, W });
        Assert.Throws<InvalidOperationException>(() => assembler.Add(extra));
    }

    [Theory]
    [InlineData(1000, 700, 250, 2, 1700)]    // hop is not window - overlap
    [InlineData(1000, 750, 250, 2, 1751)]    // target longer than the joined audio
    [InlineData(1000, 750, 250, 0, 100)]     // no windows
    [InlineData(1000, 0, 1000, 2, 1000)]     // overlap as long as the window
    public void An_inconsistent_layout_is_rejected(int window, int hop, int overlap, int windows, long target) =>
        Assert.Throws<ArgumentException>(() => new WindowAssembler(window, hop, overlap, windows, target));
}
