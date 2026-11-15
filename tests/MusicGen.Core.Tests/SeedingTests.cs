using MusicGen.Core.Testing;
using MusicGen.Core;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

public class SeedingTests
{
    private static Tensor Draw(Device device) => rand(new long[] { 16 }, device: device);

    [Fact]
    public void Cpu_SameSeed_GivesTheSameSequence_AndAnotherSeedAnotherOne()
    {
        Seeding.Apply(5);
        var a = Draw(CPU);
        Seeding.Apply(5);
        var b = Draw(CPU);
        Seeding.Apply(6);
        var c = Draw(CPU);

        Assert.True(a.Equals(b), "the same seed should give the same sequence");
        Assert.False(a.Equals(c), "another seed should give a different sequence");
    }

    [Fact]
    public void Apply_DoesNotFail_WhateverTheBackend()
    {
        // With the CPU backend (or no GPU) there is no CUDA to seed and it must not fail.
        var ex = Record.Exception(() => Seeding.Apply(1234));
        Assert.Null(ex);
    }

    [RequiresCudaFact]
    public void Cuda_SameSeed_GivesTheSameSequence_OnEveryGpu_AndAnotherSeedAnotherOne()
    {
        for (int i = 0; i < cuda.device_count(); i++)
        {
            var gpu = new Device(DeviceType.CUDA, i);
            Seeding.Apply(5);
            var a = Draw(gpu).cpu();
            Seeding.Apply(5);
            var b = Draw(gpu).cpu();
            Seeding.Apply(6);
            var c = Draw(gpu).cpu();

            Assert.True(a.Equals(b), $"cuda:{i}: the same seed should give the same sequence");
            Assert.False(a.Equals(c), $"cuda:{i}: another seed should give a different sequence");
        }
    }

    [Fact]
    public void NewSeed_IsNonNegative32Bit_AndVaries()
    {
        var seeds = Enumerable.Range(0, 8).Select(_ => Seeding.NewSeed()).ToList();

        Assert.All(seeds, s => Assert.InRange(s, 0L, (1L << 32) - 1));
        Assert.True(seeds.Distinct().Count() > 1, "8 consecutive random seeds should not all be equal");
    }
}
