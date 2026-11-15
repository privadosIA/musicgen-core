using MusicGen.Core.Testing;

namespace MusicGen.Core.Tests;

/// <summary>Access to the core's own small fixtures (tests/fixtures of this repository), verified against MANIFEST.md.</summary>
internal static class CoreFixtures
{
    private static readonly FixtureDirectory Fixtures = FixtureDirectory.Find();

    public static string PathOf(string file) => Fixtures.PathOf(file);
}
