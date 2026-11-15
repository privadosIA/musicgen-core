using TorchSharp;
using static TorchSharp.torch;

namespace MusicGen.Core.Audio;

/// <summary>
/// Joins the audio of consecutive equal-size windows (HeartCodec's <c>detokenize</c> and similar chunked decoders): the last <c>overlapSamples</c> of what is
/// there are mixed with the first ones of the next window with a linear ramp (old <c>· (1 − ramp)</c>, new <c>· ramp</c>) and the rest of the new window is
/// appended. The result is written into one preallocated float32 CPU tensor, so only a window's worth of audio is ever held besides it.
/// </summary>
public sealed class WindowAssembler : IDisposable
{
    private readonly int _windowSamples;
    private readonly int _overlapSamples;
    private readonly int _windows;
    private readonly long _targetSamples;
    private readonly Tensor _output;
    private readonly long _channels;
    private long _length;

    /// <param name="windowSamples">Length of every window.</param>
    /// <param name="hopSamples">Distance between the starts of consecutive windows; must equal <paramref name="windowSamples"/> − <paramref name="overlapSamples"/>.</param>
    /// <param name="overlapSamples">Samples shared by consecutive windows (cross-faded).</param>
    /// <param name="windows">How many windows will be added.</param>
    /// <param name="targetSamples">Length the joined audio is cut to by <see cref="Finish"/>; at most <c>windowSamples + (windows − 1)·hopSamples</c>.</param>
    public WindowAssembler(int windowSamples, int hopSamples, int overlapSamples, int windows, long targetSamples, long channels = 2)
    {
        if (windowSamples <= 0 || overlapSamples < 0 || overlapSamples >= windowSamples || windows <= 0)
            throw new ArgumentException($"invalid window layout (window {windowSamples}, overlap {overlapSamples}, windows {windows})");
        if (hopSamples != windowSamples - overlapSamples)
            throw new ArgumentException($"hop {hopSamples} must equal window − overlap ({windowSamples} − {overlapSamples})");
        long total = windowSamples + (long)(windows - 1) * hopSamples;
        if (targetSamples <= 0 || targetSamples > total)
            throw new ArgumentException($"target length {targetSamples} must be within 1..{total}");
        _windowSamples = windowSamples;
        _overlapSamples = overlapSamples;
        _windows = windows;
        _targetSamples = targetSamples;
        _channels = channels;
        _output = zeros(new long[] { channels, total }, ScalarType.Float32);
    }

    public int WindowsAdded { get; private set; }

    /// <summary><c>numpy.linspace(0, 1, n)</c>: <c>i · (1 / (n − 1))</c> with the last value set to exactly 1.</summary>
    public static double[] Ramp(int n)
    {
        var ramp = new double[n];
        if (n == 1) return ramp;
        double step = 1.0 / (n - 1);
        for (int i = 0; i < n; i++) ramp[i] = i * step;
        ramp[n - 1] = 1.0;
        return ramp;
    }

    /// <param name="audio"><c>[channels, WindowSamples]</c> of the next window (any device, any float type)</param>
    public void Add(Tensor audio)
    {
        if (WindowsAdded >= _windows) throw new InvalidOperationException($"the plan has {_windows} windows");
        if (audio.shape[0] != _channels || audio.shape[1] != _windowSamples)
            throw new ArgumentException($"expected [{_channels}, {_windowSamples}], got [{string.Join(", ", audio.shape)}]");
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var next = audio.detach().to(CPU).to(ScalarType.Float32);
        int ovlp = _overlapSamples;
        if (WindowsAdded == 0)
        {
            _output[.., .._windowSamples].copy_(next);
            _length = _windowSamples;
        }
        else
        {
            var ramp = tensor(Ramp(ovlp), ScalarType.Float64).unsqueeze(0);                  // fade-in of the new window
            var old = _output[.., (int)(_length - ovlp)..(int)_length];
            old.copy_(old.to(ScalarType.Float64) * (1 - ramp) + next[.., ..ovlp].to(ScalarType.Float64) * ramp);
            long appended = _windowSamples - ovlp;
            _output[.., (int)_length..(int)(_length + appended)].copy_(next[.., ovlp..]);
            _length += appended;
        }
        WindowsAdded++;
    }

    /// <summary>The joined audio cut to the plan's target length, <c>[channels, TargetSamples]</c> (a copy).</summary>
    public Tensor Finish()
    {
        if (WindowsAdded != _windows) throw new InvalidOperationException($"{WindowsAdded} of {_windows} windows were added");
        return _output[.., ..(int)_targetSamples].clone();
    }

    public void Dispose() => _output.Dispose();
}
