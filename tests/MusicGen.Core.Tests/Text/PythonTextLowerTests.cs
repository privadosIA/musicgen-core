using System.Text.Json;
using MusicGen.Core.Text;
using Xunit.Abstractions;

namespace MusicGen.Core.Tests;

/// <summary>
/// PythonText.Lower against Python's str.lower() (3.11, Unicode 14.0). The explicit cases are the ones in the spec; the exhaustive test walks
/// ALL code points against lower_reference.json (lowercase and the two final-sigma properties, obtained with probes).
/// </summary>
public class PythonTextLowerTests
{
    private readonly ITestOutputHelper _out;
    public PythonTextLowerTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData("ΑΣ", "ας")]
    [InlineData("ΟΔΥΣΣΕΥΣ", "οδυσσευς")]
    [InlineData("ΣΑΣ ΑΣ.", "σας ας.")]
    [InlineData("İstanbul", "i̇stanbul")]
    [InlineData("Σ", "σ")]
    [InlineData("ΑΣΑ", "ασα")]
    [InlineData("ÀÉÎ ǅ ẞ Ａ", "àéî ǆ ß ａ")]
    [InlineData("", "")]
    public void SpecCases_MatchPython(string input, string expected) => Assert.Equal(expected, PythonText.Lower(input));

    [Fact]
    public void DiffersFromToLowerInvariant_WhereTheSpecSays()
    {
        Assert.NotEqual("ΑΣ".ToLowerInvariant(), PythonText.Lower("ΑΣ"));
        Assert.NotEqual("İstanbul".ToLowerInvariant(), PythonText.Lower("İstanbul"));
    }

    [Fact]
    public void LoneSurrogates_AreLeftUntouched()
    {
        Assert.Equal("\uD800a", PythonText.Lower("\uD800A"));
        Assert.Equal("a\uDC00", PythonText.Lower("A\uDC00"));
    }

    [Fact]
    public void FixtureTexts_GiveExactlyPythonsLowercase()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(CoreFixtures.PathOf("text_cases.json")));
        var cases = doc.RootElement.GetProperty("cases").EnumerateArray().ToList();
        Assert.Equal(28, cases.Count);

        foreach (var c in cases)
            Assert.Equal(c.GetProperty("lower").GetString(), PythonText.Lower(c.GetProperty("text").GetString()!));
    }

    private static HashSet<int> Expand(JsonElement ranges)
    {
        var set = new HashSet<int>();
        foreach (var r in ranges.EnumerateArray())
            for (int cp = r[0].GetInt32(); cp <= r[1].GetInt32(); cp++) set.Add(cp);
        return set;
    }

    private static string Ranges(IEnumerable<int> cps)
    {
        var parts = new List<string>();
        int? start = null, prev = null;
        foreach (var cp in cps.OrderBy(x => x))
        {
            if (start is null) { start = prev = cp; continue; }
            if (cp == prev + 1) { prev = cp; continue; }
            parts.Add(start == prev ? $"{start:X}" : $"{start:X}-{prev:X}");
            start = prev = cp;
        }
        if (start is not null) parts.Add(start == prev ? $"{start:X}" : $"{start:X}-{prev:X}");
        return string.Join(" ", parts);
    }

    [Fact]
    public void EveryCodePoint_MatchesPython_LowercaseAndSigmaProperties()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(CoreFixtures.PathOf("lower_reference.json")));
        var root = doc.RootElement;
        var lowerDiff = root.GetProperty("lower_diff").EnumerateArray()
            .ToDictionary(e => e[0].GetInt32(), e => string.Concat(e[1].EnumerateArray().Select(x => char.ConvertFromUtf32(x.GetInt32()))));
        var casedNonIgnorable = Expand(root.GetProperty("cased_nonignorable_ranges"));
        var caseIgnorable = Expand(root.GetProperty("case_ignorable_ranges"));

        var wrongLower = new List<int>();
        var wrongCased = new List<int>();
        var wrongIgnorable = new List<int>();
        int checkedCount = 0;
        for (int cp = 0; cp < 0x110000; cp++)
        {
            if (cp is >= 0xD800 and <= 0xDFFF) continue;
            checkedCount++;
            var c = char.ConvertFromUtf32(cp);

            var expectedLower = lowerDiff.TryGetValue(cp, out var low) ? low : c;
            if (PythonText.Lower(c) != expectedLower) wrongLower.Add(cp);

            // The same probes Python used: A = (c + Σ) ends in ς; B = (a + c + Σ) ends in ς.
            bool a = PythonText.Lower(c + "Σ").EndsWith('ς');
            bool b = PythonText.Lower("a" + c + "Σ").EndsWith('ς');
            if (a != casedNonIgnorable.Contains(cp)) wrongCased.Add(cp);
            if ((b && !a) != caseIgnorable.Contains(cp)) wrongIgnorable.Add(cp);
        }

        _out.WriteLine($"{checkedCount} code points; mismatches: lowercase {wrongLower.Count}, Cased {wrongCased.Count}, Case_Ignorable {wrongIgnorable.Count}");
        if (wrongLower.Count > 0) _out.WriteLine("lowercase: " + Ranges(wrongLower));
        if (wrongCased.Count > 0) _out.WriteLine("Cased: " + Ranges(wrongCased));
        if (wrongIgnorable.Count > 0) _out.WriteLine("Case_Ignorable: " + Ranges(wrongIgnorable));
        Assert.Equal(0, wrongLower.Count + wrongCased.Count + wrongIgnorable.Count);
    }
}
