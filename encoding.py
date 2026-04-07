import torch


def _validate_word_layout(word_bits, target_bits):
    if target_bits != 2 * word_bits:
        raise ValueError(
            f"Expected target_bits == 2 * word_bits, got target_bits={target_bits}, "
            f"word_bits={word_bits}"
        )


def _fixed_point_scale(target_bits):
    return float((1 << (target_bits - 1)) - 1)


def encode_words_from_scalar(y, word_bits=16, target_bits=32):
    """
    y is assumed in [-1, 1]. Convert to signed fixed-point int(2*word_bits)
    and split into high/low words. Returns normalized hi/lo words in [0, 1].
    """
    _validate_word_layout(word_bits, target_bits)

    scale = _fixed_point_scale(target_bits)
    y_scaled = torch.clamp(y, -1.0, 1.0)
    y_int = torch.round(y_scaled * scale).to(torch.int64)

    mask = (1 << target_bits) - 1
    y_uint = y_int & mask

    hi = ((y_uint >> word_bits) & ((1 << word_bits) - 1)).float()
    lo = (y_uint & ((1 << word_bits) - 1)).float()

    denom = float((1 << word_bits) - 1)
    return hi / denom, lo / denom


def decode_words_to_scalar(hi, lo, word_bits=16, target_bits=32):
    _validate_word_layout(word_bits, target_bits)

    denom = float((1 << word_bits) - 1)
    hi_int = torch.round(torch.clamp(hi, 0.0, 1.0) * denom).to(torch.int64)
    lo_int = torch.round(torch.clamp(lo, 0.0, 1.0) * denom).to(torch.int64)

    y_uint = (hi_int << word_bits) | lo_int
    sign_threshold = 1 << (target_bits - 1)
    y_int = torch.where(y_uint >= sign_threshold, y_uint - (1 << target_bits), y_uint)

    scale = _fixed_point_scale(target_bits)
    return y_int.float() / scale


def encode_bits_from_scalar(y, total_bits=32):
    scale = _fixed_point_scale(total_bits)
    y_scaled = torch.clamp(y, -1.0, 1.0)
    y_int = torch.round(y_scaled * scale).to(torch.int64)
    y_uint = y_int & ((1 << total_bits) - 1)

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

    sign_threshold = 1 << (total_bits - 1)
    y_int = torch.where(y_uint >= sign_threshold, y_uint - (1 << total_bits), y_uint)
    scale = _fixed_point_scale(total_bits)
    return y_int.float() / scale
