using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Weights;

/// <summary>
/// A tensor of a checkpoint: the file (shard) that contains it, the dtype/shape from the header and its absolute
/// position (<paramref name="Offset"/>, <paramref name="Length"/> in bytes) within the file.
/// </summary>
public sealed record TensorInfo(string Name, string File, string Dtype, long[] Shape, long Offset, long Length);

/// <summary>
/// Index of a safetensors checkpoint built by reading only headers (never data).
/// Resolves a directory via model.safetensors.index.json (sharded) or model.safetensors (single file)
/// and checks that each shard contains exactly the keys the index assigns to it.
/// </summary>
public sealed class CheckpointIndex
{
    public IReadOnlyDictionary<string, TensorInfo> Tensors { get; }

    private CheckpointIndex(Dictionary<string, TensorInfo> tensors) => Tensors = tensors;

    /// <summary>Opens a checkpoint directory, or a single .safetensors file.</summary>
    public static CheckpointIndex Open(string path)
    {
        if (File.Exists(path))
            return new CheckpointIndex(ReadShard(path));

        var indexFile = Path.Combine(path, "model.safetensors.index.json");
        if (File.Exists(indexFile))
            return OpenSharded(path, indexFile);

        var single = Path.Combine(path, "model.safetensors");
        if (File.Exists(single))
            return new CheckpointIndex(ReadShard(single));

        throw new FileNotFoundException($"No safetensors found at {path}");
    }

    /// <summary>Reads the tensor <paramref name="key"/> without loading the rest, exactly as it is in the file.</summary>
    public Tensor Read(string key) =>
        Tensors.TryGetValue(key, out var info)
            ? SafeTensors.ReadTensor(info)
            : throw new KeyNotFoundException($"The checkpoint does not contain the tensor {key}");

    /// <summary>
    /// Reads <paramref name="key"/> and returns it as <paramref name="dtype"/>. F32→BF16 uses libtorch's
    /// round-to-nearest with ties to even; the tensor that was read (file dtype) is released after converting.
    /// </summary>
    public Tensor Read(string key, ScalarType dtype)
    {
        var raw = Read(key);
        if (raw.dtype == dtype) return raw;
        try { return raw.to(dtype); }
        finally { raw.Dispose(); }
    }

    private static CheckpointIndex OpenSharded(string dir, string indexFile)
    {
        var weightMap = JsonDocument.Parse(File.ReadAllText(indexFile)).RootElement.GetProperty("weight_map");
        var byShard = weightMap.EnumerateObject()
            .GroupBy(p => p.Value.GetString() ?? throw new InvalidDataException($"the index maps {p.Name} to a non-string shard"), p => p.Name)
            .OrderBy(g => g.Key, StringComparer.Ordinal);

        var tensors = new Dictionary<string, TensorInfo>();
        var problems = new List<string>();
        foreach (var shard in byShard)
        {
            var file = SafeTensors.ResolveShardPath(dir, shard.Key);
            if (!File.Exists(file))
                throw new FileNotFoundException($"The index references shard {shard.Key}, which does not exist in {dir}");

            var found = ReadShard(file);
            var expected = shard.ToHashSet(StringComparer.Ordinal);
            var missing = expected.Where(k => !found.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
            var extra = found.Keys.Where(k => !expected.Contains(k)).Order(StringComparer.Ordinal).ToList();
            if (missing.Count > 0)
                problems.Add($"shard {shard.Key}: missing {missing.Count} keys from the index: {string.Join(", ", missing)}");
            if (extra.Count > 0)
                problems.Add($"shard {shard.Key}: {extra.Count} extra keys that the index does not list: {string.Join(", ", extra)}");

            foreach (var kv in found)
                tensors[kv.Key] = kv.Value;
        }

        if (problems.Count > 0)
            throw new InvalidDataException("The checkpoint is not consistent with its index:\n  " + string.Join("\n  ", problems));
        return new CheckpointIndex(tensors);
    }

    private static Dictionary<string, TensorInfo> ReadShard(string file)
    {
        var (dataBase, entries) = SafeTensors.ReadHeader(file);
        foreach (var e in entries)
        {
            try { SafeTensors.MapDtype(e.Dtype); }
            catch (NotSupportedException)
            {
                throw new NotSupportedException($"Tensor {e.Name} in {Path.GetFileName(file)} has an unsupported dtype: {e.Dtype}");
            }
        }
        return entries.ToDictionary(
            e => e.Name,
            e => new TensorInfo(e.Name, file, e.Dtype, e.Shape, dataBase + e.Start, e.End - e.Start),
            StringComparer.Ordinal);
    }
}
