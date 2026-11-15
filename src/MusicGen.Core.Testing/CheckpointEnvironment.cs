namespace MusicGen.Core.Testing;

/// <summary>
/// Where a port's tests find the real checkpoint, the optional directory for listening audio and the device, each from an
/// environment variable whose name the port chooses (HeartMuLa.Net uses HEARTMULA_CKPT, HEARTMULA_AUDIO_DIR, HEARTMULA_DEVICE).
/// </summary>
public sealed class CheckpointEnvironment
{
    public string CheckpointVariable { get; }
    public string? AudioDirVariable { get; }
    public string? DeviceVariable { get; }

    public CheckpointEnvironment(string checkpointVariable, string? audioDirVariable = null, string? deviceVariable = null)
    {
        CheckpointVariable = checkpointVariable;
        AudioDirVariable = audioDirVariable;
        DeviceVariable = deviceVariable;
    }

    /// <summary>The checkpoint directory (never inside the repository), or null if the variable is unset or the directory does not exist.</summary>
    public string? Dir => Get(CheckpointVariable) is { } d && Directory.Exists(d) ? d : null;

    /// <summary>Directory where the tests may leave audio to listen to. Without it, no test writes audio.</summary>
    public string? AudioDir => AudioDirVariable is null ? null : Get(AudioDirVariable);

    /// <summary>Device for the real-model tests: the variable's value in lower case, cpu by default.</summary>
    public string DeviceName => DeviceVariable is null ? "cpu" : Get(DeviceVariable)?.ToLowerInvariant() ?? "cpu";

    /// <summary>Skip reason when there is no checkpoint, or null when there is one.</summary>
    public string? NoCheckpointReason => Dir is null ? $"{CheckpointVariable} not set or the directory does not exist" : null;

    private static string? Get(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : null;
}
