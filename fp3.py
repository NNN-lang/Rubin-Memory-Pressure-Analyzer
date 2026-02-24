import torch torch

BIAS=1

 (xtorchTensor):
"""
    Quantize FP32 tensor to custom FP3 (E1M1S1).
    Returns int tensor with values 0-7.


      (x<0)int()
    x_abs  .x

# exponent bit
     =torchzeros_like(x_absdtype=.int32)
    exp[x_abs >= 2.0] = 1

    # mantissa bit
    mant = torch.zeros_like(x_abs, dtype=torch.int32)
    mant[(x_abs >= 1.5) & (x_abs < 2.0)] = 1
    mant[(x_abs >= 3.0)] = 1

    # clamp small values to zero
    x_abs[x_abs < 1.0] = 0.0

    fp3 = (sign << 2) | (exp << 1) | mant
    return fp3.int()


def fp3_to_fp32(fp3: torch.Tensor):
    """
    Dequantize FP3 back to FP32.
    """

    sign = ((fp3 >> 2) & 0x1).float()
    exp  = ((fp3 >> 1) & 0x1).float()
    mant = (fp3 & 0x1).float()

    base = torch.where(exp == 0, torch.ones_like(exp), torch.full_like(exp, 2.0))
    value = base + mant * (base / 2.0)

    value[base == 0] = 0.0

    value = torch.where(sign == 1, -value, value)

    return value
