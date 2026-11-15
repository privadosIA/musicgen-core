using System.Security.Cryptography;
using System.Text.RegularExpressions;
using MusicGen.Core.Weights;
using TorchSharp;

namespace MusicGen.Core.Testing;

/// <summary>
/// A directory of reference fixtures (the golden run of the Python implementation, versioned in the repository). Files listed in its
/// MANIFEST.md (a table whose first cell is the file name in backticks and which has a 64-hex SHA-256 cell) are verified before
/// they are used, so a regenerated or corrupted fixture fails with a message naming the file and both checksums.
/// </summary>
public sealed class FixtureDirectory
{
    private static readonly Regex Sha256 = new("^`?([0-9a-f]{64})`?$", RegexOptions.Compiled);
    private readonly Dictionary<string, string> _expected;
    private readonly HashSet<string> _verified = new();

    public string Dir { get; }

    public FixtureDirectory(string dir)
    {
        Dir = dir;
        _expected = ReadManifest(Path.Combine(dir, "MANIFEST.md"));
    }

    /// <summary>Finds <c>tests/fixtures</c> walking up from the test binaries.</summary>
    public static FixtureDirectory Find()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            var c = Path.Combine(d.FullName, "tests", "fixtures");
            if (Directory.Exists(c)) return new FixtureDirectory(c);
        }
        throw new DirectoryNotFoundException("tests/fixtures not found");
    }

    /// <summary>Full path of a fixture, after verifying its checksum when the manifest lists it.</summary>
    public string PathOf(string file)
    {
        var path = Path.Combine(Dir, file);
        lock (_verified)
        {
            if (_verified.Contains(file)) return path;
            if (_expected.TryGetValue(file, out var expected)) Verify(path, expected);
            _verified.Add(file);
        }
        return path;
    }

    public Dictionary<string, torch.Tensor> Load(string file) => SafeTensors.LoadFile(PathOf(file));

    public string ReadAllText(string file) => File.ReadAllText(PathOf(file));

    /// <summary>Throws an <see cref="InvalidDataException"/> naming the file, the expected and the actual SHA-256 when they differ.</summary>
    public static void Verify(string path, string expectedSha256)
    {
        using var fs = File.OpenRead(path);
        var actual = Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
        if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"fixture {Path.GetFileName(path)} does not match its recorded SHA-256: expected {expectedSha256.ToLowerInvariant()}, actual {actual}");
    }

    private static Dictionary<string, string> ReadManifest(string manifest)
    {
        var map = new Dictionary<string, string>();
        if (!File.Exists(manifest)) return map;
        foreach (var line in File.ReadLines(manifest))
        {
            if (!line.TrimStart().StartsWith('|')) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim()).ToArray();
            if (cells.Length < 2 || !(cells[0].StartsWith('`') && cells[0].EndsWith('`'))) continue;
            var sha = cells.Skip(1).Select(c => Sha256.Match(c)).FirstOrDefault(m => m.Success);
            if (sha != null) map[cells[0].Trim('`')] = sha.Groups[1].Value;
        }
        return map;
    }
}
