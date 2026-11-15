namespace MusicGen.Core.Testing;

public static class Bf16
{
    /// <summary>
    /// Oracle independent of libtorch: round-to-nearest-even from F32 to BF16 by bit arithmetic
    /// (add 0x7FFF + bit 16 and truncate). Does not cover NaN, which is not a case in the spec.
    /// </summary>
    public static ushort RoundNearestEven(float f)
    {
        uint b = BitConverter.SingleToUInt32Bits(f);
        return (ushort)((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
    }
}
