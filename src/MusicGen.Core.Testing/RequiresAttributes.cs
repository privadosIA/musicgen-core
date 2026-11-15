using Xunit;

namespace MusicGen.Core.Testing;

// Each port derives a thin attribute that passes its CheckpointEnvironment, so the tests keep a short [RequiresCheckpoint]
// while the environment variable names stay the port's own. A gated test is reported as skipped with a reason, never as passed.

/// <summary>[Fact] that is skipped unless the checkpoint directory variable points to an existing directory.</summary>
public abstract class RequiresCheckpointFactAttribute : FactAttribute
{
    protected RequiresCheckpointFactAttribute(CheckpointEnvironment env) => Skip = env.NoCheckpointReason;
}

/// <summary>[Fact] that, besides needing the checkpoint, only makes sense on CPU (e.g. the host RSS peak).</summary>
public abstract class RequiresCpuCheckpointFactAttribute : FactAttribute
{
    protected RequiresCpuCheckpointFactAttribute(CheckpointEnvironment env)
    {
        Skip = env.NoCheckpointReason
            ?? (env.DeviceName != "cpu" ? $"only applies on CPU ({env.DeviceVariable}={env.DeviceName})" : null);
    }
}

/// <summary>[Fact] that writes audio to be listened to: it needs the checkpoint and the audio directory variable.</summary>
public abstract class RequiresAudioDirFactAttribute : FactAttribute
{
    protected RequiresAudioDirFactAttribute(CheckpointEnvironment env)
    {
        Skip = env.NoCheckpointReason
            ?? (env.AudioDir is null ? $"{env.AudioDirVariable} not set: the tests do not write audio by default" : null);
    }
}
