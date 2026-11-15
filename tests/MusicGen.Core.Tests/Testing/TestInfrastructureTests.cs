using MusicGen.Core.Testing;

namespace MusicGen.Core.Tests;

public class FixtureDirectoryTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("fixtures-").FullName;
    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private const string HelloSha = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"; // sha256("hello\n")

    private void Write(string name, string text) => File.WriteAllText(Path.Combine(_dir, name), text);

    [Fact]
    public void MatchingChecksum_ReturnsThePath()
    {
        Write("a.txt", "hello\n");
        Write("MANIFEST.md", $"| File | Size | sha256 |\n|---|---|---|\n| `a.txt` | 6 | `{HelloSha}` | note |\n");
        Assert.Equal(Path.Combine(_dir, "a.txt"), new FixtureDirectory(_dir).PathOf("a.txt"));
    }

    [Fact]
    public void ChecksumMismatch_NamesTheFileAndBothChecksums()
    {
        Write("a.txt", "hello, changed\n");
        Write("MANIFEST.md", $"| `a.txt` | 6 | `{HelloSha}` |\n");
        var ex = Assert.Throws<InvalidDataException>(() => new FixtureDirectory(_dir).PathOf("a.txt"));
        Assert.Contains("a.txt", ex.Message);
        Assert.Contains($"expected {HelloSha}", ex.Message);
        Assert.Matches("actual [0-9a-f]{64}", ex.Message);
        Assert.DoesNotContain($"actual {HelloSha}", ex.Message);
    }

    [Fact]
    public void FilesNotInTheManifest_AreNotVerified()
    {
        Write("b.txt", "anything");
        Write("MANIFEST.md", $"| `a.txt` | 6 | `{HelloSha}` |\n");
        Assert.EndsWith("b.txt", new FixtureDirectory(_dir).PathOf("b.txt"));
    }
}

public class CheckpointGatingTests
{
    private sealed class NeedsCheckpoint(CheckpointEnvironment env) : RequiresCheckpointFactAttribute(env);
    private sealed class NeedsCpu(CheckpointEnvironment env) : RequiresCpuCheckpointFactAttribute(env);
    private sealed class NeedsAudio(CheckpointEnvironment env) : RequiresAudioDirFactAttribute(env);

    private const string Unset = "MUSICGEN_CORE_TEST_UNSET_VARIABLE";

    [Fact]
    public void WithoutTheVariable_TheTestIsSkippedWithAReasonNamingIt()
    {
        var env = new CheckpointEnvironment(Unset, "MUSICGEN_CORE_TEST_UNSET_AUDIO", "MUSICGEN_CORE_TEST_UNSET_DEVICE");
        Assert.Contains(Unset, new NeedsCheckpoint(env).Skip);
        Assert.Contains(Unset, new NeedsCpu(env).Skip);
        Assert.Contains(Unset, new NeedsAudio(env).Skip);
    }

    [Fact]
    public void WithAnExistingDirectory_TheTestRuns_AndTheDeviceDefaultsToCpu()
    {
        var name = "MUSICGEN_CORE_TEST_CKPT";
        Environment.SetEnvironmentVariable(name, AppContext.BaseDirectory);
        try
        {
            var env = new CheckpointEnvironment(name, "MUSICGEN_CORE_TEST_UNSET_AUDIO", "MUSICGEN_CORE_TEST_UNSET_DEVICE");
            Assert.Null(new NeedsCheckpoint(env).Skip);
            Assert.Null(new NeedsCpu(env).Skip);
            Assert.Contains("MUSICGEN_CORE_TEST_UNSET_AUDIO", new NeedsAudio(env).Skip);
        }
        finally { Environment.SetEnvironmentVariable(name, null); }
    }

    [Fact]
    public void ADirectoryThatDoesNotExist_CountsAsUnset()
    {
        var name = "MUSICGEN_CORE_TEST_CKPT2";
        Environment.SetEnvironmentVariable(name, Path.Combine(AppContext.BaseDirectory, "does-not-exist"));
        try { Assert.NotNull(new NeedsCheckpoint(new CheckpointEnvironment(name)).Skip); }
        finally { Environment.SetEnvironmentVariable(name, null); }
    }

    [Fact]
    public void TheCpuGate_SkipsWhenTheDeviceVariableSaysCuda()
    {
        var ckpt = "MUSICGEN_CORE_TEST_CKPT3"; var dev = "MUSICGEN_CORE_TEST_DEV3";
        Environment.SetEnvironmentVariable(ckpt, AppContext.BaseDirectory);
        Environment.SetEnvironmentVariable(dev, "CUDA");
        try { Assert.Contains($"{dev}=cuda", new NeedsCpu(new CheckpointEnvironment(ckpt, null, dev)).Skip); }
        finally { Environment.SetEnvironmentVariable(ckpt, null); Environment.SetEnvironmentVariable(dev, null); }
    }
}
