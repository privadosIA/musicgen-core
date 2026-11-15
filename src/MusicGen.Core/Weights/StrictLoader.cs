using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Weights;

/// <summary>How a checkpoint maps onto a model. Everything that differs between the ports' loaders is here; the algorithm is not.</summary>
/// <param name="ErrorPrefix">Start of the exception message, colon included (e.g. <c>"The checkpoint does not match the model:"</c>). Callers keep their own text because tests assert on it.</param>
public sealed record StrictLoadOptions(string ErrorPrefix)
{
    /// <summary>Only checkpoint keys that start with this are looked at; the rest are counted as outside and never read.</summary>
    public string Prefix { get; init; } = "";

    /// <summary>Leading part of the file key removed to obtain the model key. Defaults to <see cref="Prefix"/>.</summary>
    public string? StripPrefix { get; init; }

    /// <summary>Allow-list: model-side keys for which this returns true are skipped (not loaded, not reported as unexpected).</summary>
    public Func<string, bool>? Skip { get; init; }

    /// <summary>Fold <c>….parametrizations.weight.original0/original1</c> pairs into a plain <c>….weight</c> (<see cref="WeightNorm"/>).</summary>
    public bool FoldWeightNorm { get; init; }

    /// <summary>Dtype each tensor is read (and converted) to before it is copied into the model.</summary>
    public ScalarType ReadDtype { get; init; } = ScalarType.Float32;

    /// <summary>When set, the model is converted to this dtype while it is moved to the device.</summary>
    public ScalarType? ModelDtype { get; init; }

    /// <summary>Also load the model's buffers, not only its parameters.</summary>
    public bool IncludeBuffers { get; init; }

    /// <summary>Wording for the "keys in the checkpoint that the model does not have" line: "N {label} are not in the model: …".</summary>
    public string UnexpectedLabel { get; init; } = "checkpoint keys";
}

/// <param name="TensorsRead">Tensors read from the file (a folded weight-norm pair counts as two).</param>
/// <param name="PairsFolded">Weight-norm pairs folded into a plain weight.</param>
/// <param name="SkippedByAllowList">Keys skipped by <see cref="StrictLoadOptions.Skip"/>.</param>
/// <param name="OutsidePrefix">Checkpoint tensors that do not start with <see cref="StrictLoadOptions.Prefix"/>.</param>
/// <param name="ModelTensors">Parameters (and buffers) of the model.</param>
/// <param name="BytesRead">Bytes of the tensors after conversion to the read dtype.</param>
/// <param name="ReadOrder">Checkpoint keys in the order they were read (by file, then offset).</param>
public sealed record StrictLoadResult(int TensorsRead, int PairsFolded, int SkippedByAllowList, int OutsidePrefix, int ModelTensors, long BytesRead, IReadOnlyList<string> ReadOrder);

/// <summary>
/// Strict checkpoint-to-model loading. Headers alone decide whether the checkpoint fits: if any key is missing, extra or has another shape, one
/// <see cref="InvalidDataException"/> lists every discrepancy and nothing has been read or modified. Otherwise the model is moved to the device and
/// the tensors are read one at a time, ordered by file and offset (sequential disk reads), under <c>no_grad</c>, so memory holds the model plus one
/// tensor in flight.
/// </summary>
public static class StrictLoader
{
    private const string G = ".parametrizations.weight.original0";
    private const string V = ".parametrizations.weight.original1";

    private sealed record Entry(string ModelKey, long[] Shape, string[] SourceKeys);

    public static StrictLoadResult Load(nn.Module model, CheckpointIndex checkpoint, Device device, StrictLoadOptions options)
    {
        var problems = new List<string>();
        var (entries, skipped) = Plan(checkpoint, options, problems);
        Validate(model, entries, options, problems);
        if (problems.Count > 0)
            throw new InvalidDataException(options.ErrorPrefix + "\n  " + string.Join("\n  ", problems));

        if (options.ModelDtype is { } md) model.to(device, md); else model.to(device);   // the tensors are replaced: take the state afterwards
        var state = State(model, options);

        using var _ = no_grad();
        int read = 0, folded = 0;
        long bytes = 0;
        var order = new List<string>();
        foreach (var e in entries.Values.OrderBy(e => checkpoint.Tensors[e.SourceKeys[0]].File, StringComparer.Ordinal).ThenBy(e => checkpoint.Tensors[e.SourceKeys[0]].Offset))
        {
            if (e.SourceKeys.Length == 1)
            {
                using var w = checkpoint.Read(e.SourceKeys[0], options.ReadDtype);
                state[e.ModelKey].copy_(w);
                bytes += w.NumberOfElements * w.element_size();
                read++;
            }
            else
            {
                using var g = checkpoint.Read(e.SourceKeys[0], options.ReadDtype);
                using var v = checkpoint.Read(e.SourceKeys[1], options.ReadDtype);
                using var w = WeightNorm.Fold(g, v);
                state[e.ModelKey].copy_(w);
                bytes += (g.NumberOfElements + v.NumberOfElements) * g.element_size();
                read += 2;
                folded++;
            }
            order.AddRange(e.SourceKeys);
        }
        int outside = checkpoint.Tensors.Keys.Count(k => !k.StartsWith(options.Prefix, StringComparison.Ordinal));
        return new StrictLoadResult(read, folded, skipped, outside, state.Count, bytes, order);
    }

    private static Dictionary<string, Tensor> State(nn.Module model, StrictLoadOptions options)
    {
        var state = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, p) in model.named_parameters()) state[name] = p;
        if (options.IncludeBuffers) foreach (var (name, b) in model.named_buffers()) state[name] = b;
        return state;
    }

    /// <summary>Groups the checkpoint keys into model entries (folding weight-norm pairs), noting in <paramref name="problems"/> whatever does not fit.</summary>
    private static (Dictionary<string, Entry> Entries, int Skipped) Plan(CheckpointIndex checkpoint, StrictLoadOptions options, List<string> problems)
    {
        var entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        var strip = (options.StripPrefix ?? options.Prefix).Length;
        var keys = checkpoint.Tensors.Keys.Where(k => k.StartsWith(options.Prefix, StringComparison.Ordinal)).Order(StringComparer.Ordinal).ToList();
        var keySet = keys.ToHashSet(StringComparer.Ordinal);
        int skipped = 0;

        foreach (var key in keys)
        {
            var rest = key[strip..];
            if (options.FoldWeightNorm && rest.EndsWith(G, StringComparison.Ordinal))
            {
                var layer = rest[..^G.Length];
                var vKey = key[..^G.Length] + V;
                if (!keySet.Contains(vKey)) { problems.Add($"incomplete weight_norm pair in {layer}: it has original0 but not original1"); continue; }
                var gShape = checkpoint.Tensors[key].Shape;
                var vShape = checkpoint.Tensors[vKey].Shape;
                if (gShape.Length != vShape.Length || gShape[0] != vShape[0] || gShape.Skip(1).Any(d => d != 1))
                    problems.Add($"{layer}: original0 [{string.Join(", ", gShape)}] is not a per-channel scale of original1 [{string.Join(", ", vShape)}]");
                AddEntry(entries, layer + ".weight", vShape, new[] { key, vKey }, problems);
            }
            else if (options.FoldWeightNorm && rest.EndsWith(V, StringComparison.Ordinal))
            {
                var layer = rest[..^V.Length];
                if (!keySet.Contains(key[..^V.Length] + G))
                    problems.Add($"incomplete weight_norm pair in {layer}: it has original1 but not original0");
            }
            else if (options.Skip?.Invoke(rest) == true)
            {
                skipped++;
            }
            else
            {
                AddEntry(entries, rest, checkpoint.Tensors[key].Shape, new[] { key }, problems);
            }
        }
        return (entries, skipped);
    }

    private static void AddEntry(Dictionary<string, Entry> entries, string modelKey, long[] shape, string[] sources, List<string> problems)
    {
        if (entries.ContainsKey(modelKey)) problems.Add($"{modelKey} appears twice in the checkpoint (as a plain weight and as a weight_norm pair)");
        else entries[modelKey] = new Entry(modelKey, shape, sources);
    }

    private static void Validate(nn.Module model, Dictionary<string, Entry> entries, StrictLoadOptions options, List<string> problems)
    {
        var state = State(model, options).ToDictionary(kv => kv.Key, kv => kv.Value.shape, StringComparer.Ordinal);
        var missing = state.Keys.Where(k => !entries.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
        var unexpected = entries.Keys.Where(k => !state.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
        var shapes = state.Keys.Where(entries.ContainsKey).Order(StringComparer.Ordinal)
            .Where(k => !state[k].AsSpan().SequenceEqual(entries[k].Shape))
            .Select(k => $"{k}: the model has [{string.Join(", ", state[k])}], the checkpoint [{string.Join(", ", entries[k].Shape)}]")
            .ToList();

        if (missing.Count > 0) problems.Add($"{missing.Count} model keys are missing from the checkpoint: {string.Join(", ", missing)}");
        if (unexpected.Count > 0) problems.Add($"{unexpected.Count} {options.UnexpectedLabel} are not in the model: {string.Join(", ", unexpected)}");
        if (shapes.Count > 0) problems.Add($"shape mismatches ({shapes.Count}): {string.Join("; ", shapes)}");
    }
}
