using MusicGen.Core.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

/// <summary>Folding of weight_norm: w = v · (g / ‖v‖), norm over all dimensions except 0. It does not need a checkpoint.</summary>
public class WeightNormTests
{
    private static Tensor RowNorms(Tensor w) => w.pow(2).sum(Enumerable.Range(1, (int)w.dim() - 1).Select(i => (long)i).ToArray(), keepdim: false).sqrt();

    [Fact]
    public void HandComputedExample()
    {
        // v = [[3, 4], [0, 2]] (norm 5 and 2), g = [10, 3]  →  w = [[6, 8], [0, 3]]
        var v = tensor(new[] { 3f, 4f, 0f, 2f }).reshape(2, 1, 2);
        var g = tensor(new[] { 10f, 3f }).reshape(2, 1, 1);

        using var w = WeightNorm.Fold(g, v);

        Assert.Equal(new long[] { 2, 1, 2 }, w.shape);
        Assert.Equal(new[] { 6f, 8f, 0f, 3f }, w.data<float>().ToArray().Select(x => (float)Math.Round(x, 5)));
    }

    [Theory]
    [InlineData(5, 3, 7)]      // convolution [out, in, k]
    [InlineData(6, 4, 10)]     // transposed [in, out, k]: the dimension is 0 (in) in both cases
    [InlineData(1, 8, 7)]      // last convolution: g [1,1,1]
    public void WhenGIsTheNormOfV_TheFoldedWeightIsV(int d0, int d1, int k)
    {
        var v = randn(new long[] { d0, d1, k }, generator: new Generator(1)) * 0.3;
        var g = RowNorms(v).reshape(d0, 1, 1);

        using var w = WeightNorm.Fold(g, v);

        Assert.True((w - v).abs().max().item<float>() < 1e-6f);
    }

    [Theory]
    [InlineData(5, 3, 7)]
    [InlineData(6, 4, 10)]
    public void WithArbitraryG_EachRowHasNormGAndTheDirectionOfV(int d0, int d1, int k)
    {
        var v = randn(new long[] { d0, d1, k }, generator: new Generator(2));
        var g = rand(new long[] { d0, 1, 1 }, generator: new Generator(3)) + 0.5;

        using var w = WeightNorm.Fold(g, v);

        Assert.True((RowNorms(w) - g.reshape(d0)).abs().max().item<float>() < 1e-5f, "the norm of each row should be g");
        // Same direction: w/‖w‖ == v/‖v‖
        var dirW = w / RowNorms(w).reshape(d0, 1, 1);
        var dirV = v / RowNorms(v).reshape(d0, 1, 1);
        Assert.True((dirW - dirV).abs().max().item<float>() < 1e-5f, "the direction of each row should be v's");
    }

    [Fact]
    public void ResultIsFloat32_EvenFromHalfPrecisionInputs()
    {
        var v = randn(new long[] { 3, 2, 4 }, generator: new Generator(4)).to(ScalarType.Float16);
        var g = ones(new long[] { 3, 1, 1 }).to(ScalarType.Float16);

        using var w = WeightNorm.Fold(g, v);

        Assert.Equal(ScalarType.Float32, w.dtype);
    }

    [Fact]
    public void AGThatIsNotAPerChannelScale_IsRejected()
    {
        var v = randn(new long[] { 4, 2, 3 }, generator: new Generator(5));

        Assert.Throws<ArgumentException>(() => WeightNorm.Fold(ones(new long[] { 3, 1, 1 }), v));       // different channels
        Assert.Throws<ArgumentException>(() => WeightNorm.Fold(ones(new long[] { 4, 2, 1 }), v));       // g is not [C,1,1]
    }
}
