using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Weights;

/// <summary>
/// Reader for .safetensors files (Hugging Face format) and for their shard index.
///
/// Format: [8 bytes little-endian: N] [N bytes JSON header] [raw data].
/// The header is { "tensor_name": { "dtype": "BF16", "shape": [..], "data_offsets": [start, end] }, ... , "__metadata__": {...} }.
/// The offsets are relative to the start of the data block (byte 8 + N).
///
/// Tensors are created on CPU with their original dtype (bf16 stays bf16) and moved afterwards.
/// </summary>
public static class SafeTensors
{
    internal sealed record HeaderEntry(string Name, string Dtype, long[] Shape, long Start, long End);

    /// <summary>Metadata (<c>__metadata__</c>, strings) of a .safetensors; empty if it has none.</summary>
    public static IReadOnlyDictionary<string, string> ReadMetadata(string file)
    {
        using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
        var (_, header) = ReadHeaderJson(fs, file);
        var result = new Dictionary<string, string>();
        if (header.TryGetProperty("__metadata__", out var meta))
            foreach (var p in meta.EnumerateObject())
                result[p.Name] = p.Value.GetString() ?? "";
        return result;
    }

    /// <summary>Largest header accepted (100 MB, the limit of the reference implementation): a bigger length is a corrupt or hostile file.</summary>
    public const long MaxHeaderBytes = 100_000_000;

    /// <summary>
    /// Reads and parses the JSON header, checking the length prefix against the limit and the file size before allocating anything, so that a hostile
    /// file cannot make the reader allocate gigabytes or read past the end.
    /// </summary>
    private static (long DataBase, JsonElement Header) ReadHeaderJson(FileStream fs, string file)
    {
        string name = Path.GetFileName(file);
        if (fs.Length < 8) throw new InvalidDataException($"{name}: too short to be a safetensors file ({fs.Length} bytes)");
        Span<byte> prefix = stackalloc byte[8];
        fs.ReadExactly(prefix);
        long headerLen = BitConverter.ToInt64(prefix);
        if (headerLen <= 0 || headerLen > MaxHeaderBytes || headerLen > fs.Length - 8)
            throw new InvalidDataException($"{name}: invalid safetensors header length {headerLen} (file size {fs.Length}, maximum {MaxHeaderBytes})");
        var bytes = new byte[headerLen];
        fs.ReadExactly(bytes);
        try { return (8 + headerLen, JsonDocument.Parse(bytes).RootElement); }
        catch (JsonException e) { throw new InvalidDataException($"{name}: the safetensors header is not valid JSON: {e.Message}"); }
    }

    /// <summary>Bytes per element of the dtypes this reader supports; 0 for any other (it is rejected later, naming the tensor).</summary>
    private static int DtypeSize(string dtype) => dtype switch
    {
        "BF16" or "F16" or "I16" => 2,
        "F32" or "I32" => 4,
        "F64" or "I64" => 8,
        "I8" or "U8" or "BOOL" => 1,
        _ => 0,
    };

    /// <summary>
    /// Reads only the header of a .safetensors. <c>DataBase</c> is the absolute offset of the data block. Every entry is validated before anything is
    /// allocated for it: non-negative dimensions, offsets inside the file, <c>start ≤ end</c>, and, for the supported dtypes, a byte count that matches
    /// the shape (computed with checked arithmetic, so a huge shape cannot overflow or trigger a huge allocation).
    /// </summary>
    internal static (long DataBase, List<HeaderEntry> Entries) ReadHeader(string file)
    {
        using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
        var (dataBase, header) = ReadHeaderJson(fs, file);
        string name = Path.GetFileName(file);
        long dataBytes = fs.Length - dataBase;

        var entries = new List<HeaderEntry>();
        try
        {
            foreach (var p in header.EnumerateObject())
            {
                if (p.Name == "__metadata__") continue;
                var dtype = p.Value.GetProperty("dtype").GetString() ?? throw new InvalidDataException($"{name}: tensor {p.Name} has no dtype");
                var shape = p.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                var off = p.Value.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                if (off.Length != 2) throw new InvalidDataException($"{name}: tensor {p.Name} must have 2 data_offsets, has {off.Length}");
                if (shape.Any(d => d < 0)) throw new InvalidDataException($"{name}: tensor {p.Name} has a negative dimension");
                if (off[0] < 0 || off[1] < off[0] || off[1] > dataBytes)
                    throw new InvalidDataException($"{name}: tensor {p.Name} has data_offsets [{off[0]}, {off[1]}] outside the {dataBytes} bytes of data");
                int size = DtypeSize(dtype);
                if (size > 0)
                {
                    long elements;
                    try { elements = shape.Aggregate(1L, (acc, d) => checked(acc * d)); }
                    catch (OverflowException) { throw new InvalidDataException($"{name}: tensor {p.Name} has a shape too large"); }
                    if (elements > long.MaxValue / size || elements * size != off[1] - off[0])
                        throw new InvalidDataException($"{name}: tensor {p.Name} expects {(elements > long.MaxValue / size ? "too many" : (elements * size).ToString())} bytes for its shape, the header says {off[1] - off[0]}");
                }
                entries.Add(new HeaderEntry(p.Name, dtype, shape, off[0], off[1]));
            }
        }
        catch (Exception e) when (e is KeyNotFoundException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException($"{name}: malformed tensor entry in the safetensors header: {e.Message}");
        }
        return (dataBase, entries);
    }

    /// <summary>
    /// Path of a shard named by an index: it must be a plain file name inside <paramref name="dir"/>. A rooted path or one with directories or
    /// <c>..</c> would let a hostile index make the loader open files outside the checkpoint.
    /// </summary>
    internal static string ResolveShardPath(string dir, string? shardName)
    {
        if (string.IsNullOrEmpty(shardName) || shardName is "." or ".." || Path.IsPathRooted(shardName) || shardName.AsSpan().IndexOfAny('/', '\\') >= 0)
            throw new InvalidDataException($"invalid shard name in the index: '{shardName}' (it must be a plain file name)");
        return Path.Combine(dir, shardName);
    }

    /// <summary>
    /// Loads a whole checkpoint: if <paramref name="path"/> is a directory, it looks for
    /// model.safetensors.index.json (sharded) or model.safetensors (single file).
    /// If it is a .safetensors file, it loads only that one.
    /// </summary>
    public static Dictionary<string, Tensor> LoadCheckpoint(string path, Func<string, bool>? filter = null)
    {
        if (File.Exists(path))
            return LoadFile(path, filter);

        var index = Path.Combine(path, "model.safetensors.index.json");
        if (File.Exists(index))
        {
            // nosemgrep: csharp.lang.security.filesystem.unsafe-path-combine.unsafe-path-combine -- path is the checkpoint folder the caller chose (a command-line argument), and the second part is a constant
            var weightMap = JsonDocument.Parse(File.ReadAllText(index))
                .RootElement.GetProperty("weight_map");
            var files = weightMap.EnumerateObject()
                .Select(p => p.Value.GetString() ?? throw new InvalidDataException($"the index maps {p.Name} to a non-string shard"))
                .Distinct()
                .OrderBy(f => f)
                .ToList();

            var all = new Dictionary<string, Tensor>();
            foreach (var f in files)
            {
                Console.WriteLine($"  loading shard {f}");
                foreach (var kv in LoadFile(ResolveShardPath(path, f), filter))
                    all[kv.Key] = kv.Value;
            }
            return all;
        }

        var single = Path.Combine(path, "model.safetensors");
        if (File.Exists(single))
            return LoadFile(single, filter);

        throw new FileNotFoundException($"No safetensors found at {path}");
    }

    public static Dictionary<string, Tensor> LoadFile(string file, Func<string, bool>? filter = null)
    {
        var (dataBase, all) = ReadHeader(file);
        var entries = all.Where(e => filter == null || filter(e.Name)).ToList();
        using var fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);

        // Read in offset order to get sequential disk reads.
        var result = new Dictionary<string, Tensor>(entries.Count);
        foreach (var e in entries.OrderBy(x => x.Start))
            result[e.Name] = ReadAt(fs, e.Name, e.Dtype, e.Shape, dataBase + e.Start, e.End - e.Start);
        return result;
    }

    /// <summary>
    /// Reads a single tensor from its shard without touching the rest, on CPU and with the file's dtype, shape and bytes.
    /// </summary>
    public static Tensor ReadTensor(TensorInfo info)
    {
        using var fs = new FileStream(info.File, FileMode.Open, FileAccess.Read, FileShare.Read);
        return ReadAt(fs, info.Name, info.Dtype, info.Shape, info.Offset, info.Length);
    }

    private static Tensor ReadAt(FileStream fs, string name, string dtype, long[] shape, long offset, long nbytes)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        var t = torch.empty(shape, dtype: MapDtype(dtype), device: CPU);
        if (nbytes > 0)
        {
            // Tensor.bytes exposes the tensor's memory as a Span<byte>; we fill it directly.
            var span = t.bytes;
            if (span.Length != nbytes)
                throw new InvalidDataException($"{name}: expected {span.Length} bytes, the file says {nbytes}");
            int read = 0;
            while (read < span.Length)
            {
                int n = fs.Read(span.Slice(read));
                if (n <= 0) throw new EndOfStreamException(name);
                read += n;
            }
        }
        return t;
    }

    public static ScalarType MapDtype(string dtype) => dtype switch
    {
        "BF16" => ScalarType.BFloat16,
        "F16" => ScalarType.Float16,
        "F32" => ScalarType.Float32,
        "F64" => ScalarType.Float64,
        "I64" => ScalarType.Int64,
        "I32" => ScalarType.Int32,
        "I16" => ScalarType.Int16,
        "I8" => ScalarType.Int8,
        "U8" => ScalarType.Byte,
        "BOOL" => ScalarType.Bool,
        _ => throw new NotSupportedException($"unsupported safetensors dtype: {dtype}")
    };

    /// <summary>Lists names/shapes without loading data (useful for inspecting a checkpoint).</summary>
    public static IEnumerable<(string name, string dtype, long[] shape)> Describe(string file) =>
        ReadHeader(file).Entries.Select(e => (e.Name, e.Dtype, e.Shape));
}
