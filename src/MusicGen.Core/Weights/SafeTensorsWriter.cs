using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Weights;

/// <summary>Minimal .safetensors writer (contiguous tensors on CPU).</summary>
public static class SafeTensorsWriter
{
    /// <summary>
    /// Writes the tensors. If <paramref name="metadata"/> is passed (strings, as safetensors requires) it is stored in
    /// <c>__metadata__</c> with the keys sorted, so two identical writes give the same file byte for byte;
    /// without metadata the header is exactly what it always was.
    /// </summary>
    public static void Save(string path, IDictionary<string, Tensor> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        var header = new Dictionary<string, object>();
        if (metadata is { Count: > 0 })
            header["__metadata__"] = metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(kv => kv.Key, kv => kv.Value);
        var blobs = new List<byte[]>();
        long offset = 0;

        foreach (var (name, t0) in tensors.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var t = t0.cpu().contiguous();
            var bytes = t.bytes.ToArray();
            header[name] = new Dictionary<string, object>
            {
                ["dtype"] = DtypeName(t.dtype),
                ["shape"] = t.shape,
                ["data_offsets"] = new[] { offset, offset + bytes.Length }
            };
            blobs.Add(bytes);
            offset += bytes.Length;
        }

        var json = JsonSerializer.SerializeToUtf8Bytes(header);
        // The header is padded with spaces to a multiple of 8 (safetensors convention)
        int pad = (8 - json.Length % 8) % 8;
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);
        bw.Write((long)(json.Length + pad));
        bw.Write(json);
        bw.Write(Encoding.ASCII.GetBytes(new string(' ', pad)));
        foreach (var b in blobs) bw.Write(b);
    }

    private static string DtypeName(ScalarType t) => t switch
    {
        ScalarType.BFloat16 => "BF16",
        ScalarType.Float16 => "F16",
        ScalarType.Float32 => "F32",
        ScalarType.Float64 => "F64",
        ScalarType.Int64 => "I64",
        ScalarType.Int32 => "I32",
        ScalarType.Bool => "BOOL",
        ScalarType.Byte => "U8",
        _ => throw new NotSupportedException(t.ToString())
    };
}
