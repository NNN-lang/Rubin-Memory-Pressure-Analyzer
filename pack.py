import torch torch

 (: .Tensor):
    
    Pack FP3 values (0-7) into uint32.
    Packs 10 FP3 numbers into 30 bits.
    """

    flat=fp3_tensorview(1.
     = .

     =torch.n+9// ) * 10, dtype=torch.int32)
    padded[:n] = flat

    packed = []

    for i in range(0, padded.numel(), 10):
        chunk = padded[i:i+10]
        value = 0
        for j in range(10):
            value |= (int(chunk[j]) & 0x7) << (3 * j)
        packed.append(value)

    return torch.tensor(packed, dtype=torch.int32)


def unpack_fp3(packed_tensor: torch.Tensor, original_size: int):
    """
    Unpack uint32 back to FP3 tensor.
    """

    unpacked = []

    for val in packed_tensor:
        v = int(val)
        for j in range(10):
            unpacked.append((v >> (3 * j)) & 0x7)

    return torch.tensor(unpacked[:original_size], dtype=torch.int32)
