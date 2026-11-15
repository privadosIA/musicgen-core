using System.Text;
using MusicGen.Core.Audio;
using MusicGen.Core.Testing;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public sealed class WavWriterTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("wav-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);
    private string P(string n) => Path.Combine(_dir, n);

    private static Tensor Stereo(int n, int seed) => rand(new long[] { 2, n }, generator: new Generator((ulong)seed)) * 2 - 1;

    [Fact]
    public void Float32Header_DescribesTheAudio()
    {
        WavWriter.Write(P("f.wav"), Stereo(1000, 1), 48000, WavFormat.Float32);
        var w = WavFile.Read(P("f.wav"));

        Assert.Equal(3, w.Tag);                                // IEEE float
        Assert.Equal(2, w.Channels);
        Assert.Equal(48000u, w.SampleRate);
        Assert.Equal(32, w.Bits);
        Assert.Equal(8, w.BlockAlign);                         // 2 channels × 4 bytes
        Assert.Equal(48000u * 8, w.ByteRate);
        Assert.Equal(1000 * 2 * 4, w.Data.Length);
        Assert.True(w.HasFact); Assert.Equal(1000u, w.FactSamples);
        Assert.Equal(w.FileLength - 8, w.RiffSize);            // the RIFF size matches the file
    }

    [Fact]
    public void Pcm16Header_DescribesTheAudio()
    {
        WavWriter.Write(P("p.wav"), Stereo(1000, 2), 48000, WavFormat.Pcm16);
        var w = WavFile.Read(P("p.wav"));

        Assert.Equal(1, w.Tag);                                // PCM
        Assert.Equal(2, w.Channels);
        Assert.Equal(48000u, w.SampleRate);
        Assert.Equal(16, w.Bits);
        Assert.Equal(4, w.BlockAlign);
        Assert.Equal(48000u * 4, w.ByteRate);
        Assert.Equal(1000 * 2 * 2, w.Data.Length);
        Assert.False(w.HasFact);
        Assert.Equal(w.FileLength - 8, w.RiffSize);
    }

    [Fact]
    public void Channels_AreInterleavedSampleBySample()
    {
        var audio = tensor(new[] { 0.1f, 0.2f, 0.3f, -0.1f, -0.2f, -0.3f }).reshape(2, 3);   // L = 0,1 0,2 0,3; R = −0,1 −0,2 −0,3
        WavWriter.Write(P("i.wav"), audio, 8000, WavFormat.Float32);

        Assert.Equal(new[] { 0.1f, -0.1f, 0.2f, -0.2f, 0.3f, -0.3f }, WavFile.Read(P("i.wav")).Floats());
    }

    [Fact]
    public void Float32_RoundTripIsExact()
    {
        var audio = Stereo(5000, 3);
        WavWriter.Write(P("f.wav"), audio, 48000, WavFormat.Float32);
        var back = WavFile.Read(P("f.wav")).Floats();

        var expected = new float[2 * 5000];
        var src = audio.data<float>().ToArray();
        for (int t = 0; t < 5000; t++) { expected[2 * t] = src[t]; expected[2 * t + 1] = src[5000 + t]; }
        Assert.Equal(expected, back);
    }

    [Fact]
    public void Pcm16_RoundTripErrsByAtMostHalfAQuantizationStep()
    {
        var audio = Stereo(5000, 4);
        WavWriter.Write(P("p.wav"), audio, 48000, WavFormat.Pcm16);
        var shorts = WavFile.Read(P("p.wav")).Shorts();
        var src = audio.data<float>().ToArray();

        double worst = 0;
        for (int t = 0; t < 5000; t++)
            for (int c = 0; c < 2; c++)
                worst = Math.Max(worst, Math.Abs(src[c * 5000 + t] - shorts[2 * t + c] / 32767.0));
        Assert.True(worst <= 0.5 / 32767 + 1e-7, $"maximum error {worst:E3} > half a step ({0.5 / 32767:E3})");
        Assert.True(worst > 1e-6, "with random values there should be some quantization error");
    }

    [Fact]
    public void Pcm16_SaturatesOutOfRangeValues_WithoutOverflow()
    {
        var audio = tensor(new[] { 1.5f, -2f, 1f, -1f, 0f, 100f, -100f, 0.99999f }).reshape(1, 8);
        WavWriter.Write(P("s.wav"), audio, 48000, WavFormat.Pcm16);

        Assert.Equal(new short[] { 32767, -32767, 32767, -32767, 0, 32767, -32767, 32767 }, WavFile.Read(P("s.wav")).Shorts());
    }

    [Fact]
    public void Mono_IsSupported_AndAnotherSampleRateIsRecorded()
    {
        WavWriter.Write(P("m.wav"), Stereo(10, 5)[0..1], 22050, WavFormat.Pcm16);
        var w = WavFile.Read(P("m.wav"));

        Assert.Equal(1, w.Channels);
        Assert.Equal(22050u, w.SampleRate);
        Assert.Equal(10 * 2, w.Data.Length);
    }

    [Fact]
    public void NonFiniteSamples_AreRejectedInPcm16_NamingTheirPosition()
    {
        var audio = tensor(new[] { 0f, float.NaN, 0f, 0f }).reshape(1, 4);

        var ex = Assert.Throws<InvalidDataException>(() => WavWriter.Write(P("n.wav"), audio, 48000, WavFormat.Pcm16));

        Assert.Contains("channel 0", ex.Message);
        Assert.Contains("index 1", ex.Message);
    }

    [Fact]
    public void InvalidArguments_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => WavWriter.Write(P("a.wav"), zeros(10), 48000, WavFormat.Pcm16));                  // 1D
        Assert.Throws<ArgumentException>(() => WavWriter.Write(P("b.wav"), zeros(2, 10), 0, WavFormat.Pcm16));                   // frecuencia 0
    }
}
