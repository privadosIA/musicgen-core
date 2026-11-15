using Xunit;
using System.Text;

namespace MusicGen.Core.Testing;

/// <summary>Minimal WAV reader written here, independent of WavWriter, so the writer is not checked against itself.</summary>
public sealed record WavFile(ushort Tag, ushort Channels, uint SampleRate, uint ByteRate, ushort BlockAlign, ushort Bits, byte[] Data, bool HasFact, uint FactSamples, long RiffSize, long FileLength)
{
    public static WavFile Read(string path)
    {
        var b = File.ReadAllBytes(path);
        string Tag4(int o) => Encoding.ASCII.GetString(b, o, 4);
        Assert.Equal("RIFF", Tag4(0));
        Assert.Equal("WAVE", Tag4(8));
        long riff = BitConverter.ToUInt32(b, 4);
        ushort tag = 0, ch = 0, align = 0, bits = 0; uint rate = 0, byteRate = 0, fact = 0; byte[] data = Array.Empty<byte>(); bool hasFact = false;
        int pos = 12;
        while (pos + 8 <= b.Length)
        {
            var id = Tag4(pos); int size = (int)BitConverter.ToUInt32(b, pos + 4); int body = pos + 8;
            if (id == "fmt ") { tag = BitConverter.ToUInt16(b, body); ch = BitConverter.ToUInt16(b, body + 2); rate = BitConverter.ToUInt32(b, body + 4); byteRate = BitConverter.ToUInt32(b, body + 8); align = BitConverter.ToUInt16(b, body + 12); bits = BitConverter.ToUInt16(b, body + 14); }
            else if (id == "fact") { hasFact = true; fact = BitConverter.ToUInt32(b, body); }
            else if (id == "data") data = b[body..(body + size)];
            pos = body + size + (size % 2);
        }
        return new WavFile(tag, ch, rate, byteRate, align, bits, data, hasFact, fact, riff, b.Length);
    }

    public float[] Floats() => Enumerable.Range(0, Data.Length / 4).Select(i => BitConverter.ToSingle(Data, i * 4)).ToArray();
    public short[] Shorts() => Enumerable.Range(0, Data.Length / 2).Select(i => BitConverter.ToInt16(Data, i * 2)).ToArray();
}
