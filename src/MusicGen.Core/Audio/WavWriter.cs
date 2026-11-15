using System.Text;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Audio;

public enum WavFormat
{
    /// <summary>16-bit integer PCM (the most compatible).</summary>
    Pcm16,
    /// <summary>32-bit IEEE floating point (lossless).</summary>
    Float32,
}

/// <summary>
/// WAV (RIFF/WAVE) writer for audio <c>[channels, samples]</c> with values in <c>[-1, 1]</c>, with the channels interleaved sample by sample.
/// 16-bit PCM: clips at ±1 (so the range is ±32767) and rounds to nearest, with a maximum error of half a step; a non-finite value is rejected
/// instead of turning into noise. 32-bit floating point: values as they are, with the <c>fact</c> chunk the format requires.
/// </summary>
public static class WavWriter
{
    public static void Write(string path, Tensor audio, int sampleRate, WavFormat format)
    {
        using var fs = File.Create(path);
        Write(fs, audio, sampleRate, format);
    }

    public static void Write(Stream stream, Tensor audio, int sampleRate, WavFormat format)
    {
        if (audio.dim() != 2) throw new ArgumentException($"expected audio [channels, samples], got [{string.Join(", ", audio.shape)}]");
        if (sampleRate <= 0) throw new ArgumentException($"invalid sample rate: {sampleRate}");
        int channels = checked((int)audio.shape[0]);
        long frames = audio.shape[1];
        if (channels < 1 || channels > 65535) throw new ArgumentException($"invalid number of channels: {channels}");

        using var cpu = audio.detach().cpu().to(ScalarType.Float32).contiguous();
        var samples = cpu.data<float>().ToArray();                        // [channels, samples] flat

        int bytesPerSample = format == WavFormat.Pcm16 ? 2 : 4;
        long dataBytes = frames * channels * bytesPerSample;
        bool isFloat = format == WavFormat.Float32;
        int fmtSize = isFloat ? 18 : 16;
        long riffSize = 4 + (8 + fmtSize) + (isFloat ? 8 + 4 : 0) + (8 + dataBytes) + (dataBytes % 2);
        if (riffSize > uint.MaxValue) throw new ArgumentException("the audio does not fit in a 4 GiB WAV");

        using var w = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write((uint)riffSize); w.Write(Encoding.ASCII.GetBytes("WAVE"));

        w.Write(Encoding.ASCII.GetBytes("fmt ")); w.Write((uint)fmtSize);
        w.Write((ushort)(isFloat ? 3 : 1));                               // 1 = PCM, 3 = IEEE float
        w.Write((ushort)channels);
        w.Write((uint)sampleRate);
        w.Write((uint)(sampleRate * channels * bytesPerSample));          // bytes per second
        w.Write((ushort)(channels * bytesPerSample));                     // block alignment
        w.Write((ushort)(bytesPerSample * 8));
        if (isFloat) w.Write((ushort)0);                                  // cbSize

        if (isFloat) { w.Write(Encoding.ASCII.GetBytes("fact")); w.Write(4u); w.Write((uint)frames); }

        w.Write(Encoding.ASCII.GetBytes("data")); w.Write((uint)dataBytes);
        for (long t = 0; t < frames; t++)
            for (int c = 0; c < channels; c++)
            {
                float v = samples[c * frames + t];
                if (isFloat) { w.Write(v); continue; }
                if (!float.IsFinite(v)) throw new InvalidDataException($"non-finite sample ({v}) in channel {c}, index {t}: it cannot be written as 16-bit PCM");
                w.Write((short)MathF.Round(Math.Clamp(v, -1f, 1f) * 32767f, MidpointRounding.ToEven));
            }
        if (dataBytes % 2 == 1) w.Write((byte)0);                         // RIFF padding (does not happen with 16 and 32 bits, but it is the rule)
    }
}
