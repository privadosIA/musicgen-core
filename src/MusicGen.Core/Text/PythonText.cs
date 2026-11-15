using System.Text;

namespace MusicGen.Core.Text;

/// <summary>
/// Text handling with the Python semantics heartlib uses before tokenizing: <c>str.lower()</c>.
/// .NET's <c>ToLowerInvariant()</c> does not match: it gives <c>ασ</c> where Python gives <c>ας</c> (final sigma), leaves <c>İ</c> (U+0130) as is
/// while Python turns it into <c>i̇</c> (<c>i</c> + U+0307), and it also uses ICU, so its result changes with the ICU version of each
/// system (on Linux, for Unicode 16 characters that Python 3.11 does not change). Here the <see cref="PythonTextData"/> tables are used,
/// dumped from Python 3.11 (Unicode 14.0): the same result on Linux, Windows and macOS.
/// </summary>
public static class PythonText
{
    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>
    /// Reads a text file like Python's <c>open(path, encoding="utf-8").read()</c>, which is how heartlib reads the tags and the lyrics:
    /// strict UTF-8, a BOM is kept as the U+FEFF character (it is not discarded, unlike <c>File.ReadAllText</c>) and newlines
    /// are universal (<c>\r\n</c> and <c>\r</c> become <c>\n</c>, unlike .NET, which keeps them).
    /// </summary>
    public static string ReadAllText(string path)
    {
        try { return DecodeText(File.ReadAllBytes(path)); }
        catch (InvalidDataException e) { throw new InvalidDataException($"{path}: {e.Message}", e); }
    }

    /// <summary>The same reading as <see cref="ReadAllText"/> over bytes that are already loaded. Invalid UTF-8 throws <see cref="InvalidDataException"/>.</summary>
    public static string DecodeText(byte[] bytes)
    {
        string text;
        try { text = StrictUtf8.GetString(bytes); }
        catch (DecoderFallbackException e) { throw new InvalidDataException("not valid UTF-8: " + e.Message, e); }
        return text.Replace("\r\n", "\n").Replace('\r', '\n');
    }

    /// <summary>
    /// Same as Python 3.11's <c>str.lower()</c>: simple lowercase of each code point, <c>U+0130</c> → <c>i</c> + <c>U+0307</c> and CPython's
    /// Final_Sigma rule for <c>U+03A3</c>. A lone surrogate (not valid UTF-16) is left as is, without turning it into U+FFFD.
    /// </summary>
    public static string Lower(string s)
    {
        var cps = ToCodePoints(s);
        var sb = new StringBuilder(s.Length + 4);
        for (int i = 0; i < cps.Length; i++)
        {
            int cp = cps[i];
            if (cp == 0x3A3) { sb.Append(IsFinalSigma(cps, i) ? 'ς' : 'σ'); continue; }
            if (cp is >= 0xD800 and <= 0xDFFF) { sb.Append((char)cp); continue; }

            if (TryMulti(cp, out var multi)) { foreach (var x in multi) AppendCodePoint(sb, x); }
            else AppendCodePoint(sb, SimpleLower(cp));
        }
        return sb.ToString();
    }

    private static void AppendCodePoint(StringBuilder sb, int cp)
    {
        if (cp < 0x10000) sb.Append((char)cp);
        else sb.Append(char.ConvertFromUtf32(cp));
    }

    private static int[] ToCodePoints(string s)
    {
        var list = new List<int>(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                list.Add(char.ConvertToUtf32(c, s[i + 1]));
                i++;
            }
            else list.Add(c);
        }
        return list.ToArray();
    }

    /// <summary>Python's simple lowercase; the code point itself if it does not change.</summary>
    private static int SimpleLower(int cp)
    {
        var t = PythonTextData.LowerPairs;
        int lo = 0, hi = t.Length / 2 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1, key = t[2 * mid];
            if (key == cp) return t[2 * mid + 1];
            if (key < cp) lo = mid + 1; else hi = mid - 1;
        }
        return cp;
    }

    private static bool TryMulti(int cp, out int[] to)
    {
        foreach (var (from, result) in PythonTextData.LowerMulti)
            if (from == cp) { to = result; return true; }
        to = Array.Empty<int>();
        return false;
    }

    /// <summary>CPython's handle_capital_sigma: a cased letter before (skipping case-ignorable) and none after (skipping case-ignorable).</summary>
    private static bool IsFinalSigma(int[] cps, int i)
    {
        int j = i - 1;
        while (j >= 0 && IsCaseIgnorable(cps[j])) j--;
        if (!(j >= 0 && IsCased(cps[j]))) return false;

        j = i + 1;
        while (j < cps.Length && IsCaseIgnorable(cps[j])) j++;
        return j == cps.Length || !IsCased(cps[j]);
    }

    /// <summary>Case_Ignorable according to CPython.</summary>
    internal static bool IsCaseIgnorable(int cp) => InRanges(PythonTextData.CaseIgnorableRanges, cp);

    /// <summary>Cased and not case-ignorable according to CPython: only queried on characters that are not case-ignorable.</summary>
    internal static bool IsCased(int cp) => InRanges(PythonTextData.CasedNonIgnorableRanges, cp);

    private static bool InRanges(int[] flat, int cp)
    {
        int lo = 0, hi = flat.Length / 2 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (cp < flat[2 * mid]) hi = mid - 1;
            else if (cp > flat[2 * mid + 1]) lo = mid + 1;
            else return true;
        }
        return false;
    }
}
