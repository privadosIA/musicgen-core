using MusicGen.Core.Layers;
using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Tests;

/// <summary>The checkpoints name these parameters; a rename of a field or class must never change them.</summary>
public class LayerParameterNamesTests
{
    private static string[] Names(nn.Module m) => m.named_parameters().Select(p => p.name).Order().ToArray();

    [Fact] public void RmsNorm_RegistersScale() => Assert.Equal(new[] { "scale" }, Names(new RmsNorm(8, 1e-5)));

    [Fact] public void FlowRmsNorm_RegistersWeight() => Assert.Equal(new[] { "weight" }, Names(new FlowRmsNorm(8)));

    [Fact] public void Conv1dLayer_RegistersWeightAndBias() => Assert.Equal(new[] { "bias", "weight" }, Names(new Conv1dLayer(2, 3, 5)));

    [Fact] public void ConvTranspose1dLayer_RegistersWeightAndBias() => Assert.Equal(new[] { "bias", "weight" }, Names(new ConvTranspose1dLayer(2, 3, 4, 2)));

    [Fact] public void PReLULayer_RegistersWeight() => Assert.Equal(new[] { "weight" }, Names(new PReLULayer()));
}

public class NormLayersTests
{
    private static double[] Naive(float[] x, float[] w, int dim, double eps)
    {
        var y = new double[x.Length];
        for (int r = 0; r < x.Length / dim; r++)
        {
            double mean = Enumerable.Range(0, dim).Average(i => (double)x[r * dim + i] * x[r * dim + i]);
            for (int i = 0; i < dim; i++) y[r * dim + i] = w[i] * x[r * dim + i] / Math.Sqrt(mean + eps);
        }
        return y;
    }

    [Fact]
    public void RmsNorm_MatchesTheNaiveFormula_AndKeepsTheInputDtype()
    {
        const int dim = 12;
        var norm = new RmsNorm(dim, 1e-5);
        using var x = randn(new long[] { 3, 4, dim }, generator: new Generator(1)) * 3;
        using var y32 = norm.forward(x);
        norm.to(ScalarType.BFloat16);   // as the models do: the parameters follow the activations
        using var y = norm.forward(x.to(ScalarType.BFloat16));
        Assert.Equal(ScalarType.BFloat16, y.dtype);
        var expected = Naive(x.data<float>().ToArray(), Enumerable.Repeat(1f, dim).ToArray(), dim, 1e-5);
        Assert.True(expected.Zip(y32.data<float>().ToArray(), (e, g) => Math.Abs(e - g)).Max() < 1e-5);
    }

    [Fact]
    public void FlowRmsNorm_MatchesTheNaiveFormula_InTheInputDtype()
    {
        const int dim = 12;
        var norm = new FlowRmsNorm(dim);
        using var x = randn(new long[] { 3, 4, dim }, generator: new Generator(2)) * 3;
        using var y = norm.forward(x);
        Assert.Equal(ScalarType.Float32, y.dtype);
        var expected = Naive(x.data<float>().ToArray(), Enumerable.Repeat(1f, dim).ToArray(), dim, 1e-6);
        Assert.True(expected.Zip(y.data<float>().ToArray(), (e, g) => Math.Abs(e - g)).Max() < 1e-5);
        norm.to(ScalarType.BFloat16);   // as the models do: the parameters follow the activations
        using var xb = x.to(ScalarType.BFloat16);
        using var yb = norm.forward(xb);
        Assert.Equal(ScalarType.BFloat16, yb.dtype);
    }
}
