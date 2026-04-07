import torch


def encode_words_from_scalar(y, word_bits=16):
    """
    y is assumed in [-1, 1]. Convert to signed fixed-point int32, then split into two words.
    Returns normalized hi/lo words in [0, 1].
    """
    scale = (2 ** 31) - 1
    y_scaled = torch.clamp(y, -1.0, 1.0)
    y_int = torch.round(y_scaled * scale).to(torch.int64)

    y_uint = y_int & 0xFFFFFFFF
    hi = ((y_uint >> word_bits) & ((1 << word_bits) - 1)).float()
    lo = (y_uint & ((1 << word_bits) - 1)).float()

    denom = float((1 << word_bits) - 1)
    return hi / denom, lo / denom


def decode_words_to_scalar(hi, lo, word_bits=16):
    denom = float((1 << word_bits) - 1)
    hi_int = torch.round(torch.clamp(hi, 0.0, 1.0) * denom).to(torch.int64)
    lo_int = torch.round(torch.clamp(lo, 0.0, 1.0) * denom).to(torch.int64)

    y_uint = (hi_int << word_bits) | lo_int
    y_int = torch.where(y_uint >= (1 << 31), y_uint - (1 << 32), y_uint)

    scale = float((2 ** 31) - 1)
    return y_int.float() / scale


def encode_bits_from_scalar(y, total_bits=32):
    scale = (2 ** 31) - 1
    y_scaled = torch.clamp(y, -1.0, 1.0)
    y_int = torch.round(y_scaled * scale).to(torch.int64)
    y_uint = y_int & 0xFFFFFFFF

    bits = []
    for i in range(total_bits - 1, -1, -1):
        bits.append(((y_uint >> i) & 1).float())
    return torch.stack(bits, dim=-1)


def decode_bits_to_scalar(bits):
    probs = torch.clamp(bits, 0.0, 1.0)
    hard = torch.round(probs).to(torch.int64)

    total_bits = hard.shape[-1]
    y_uint = torch.zeros(hard.shape[0], dtype=torch.int64, device=hard.device)
    for i in range(total_bits):
        shift = total_bits - 1 - i
        y_uint |= (hard[:, i] << shift)

    y_int = torch.where(y_uint >= (1 << 31), y_uint - (1 << 32), y_uint)
    scale = float((2 ** 31) - 1)
    return y_int.float() / scale