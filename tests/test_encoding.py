import torch

from encoding import (
    decode_bits_to_scalar,
    decode_words_to_scalar,
    encode_bits_from_scalar,
    encode_words_from_scalar,
)
from utils import quantize_tensor


def test_word_round_trip_is_close_default_16_32():
    y = torch.linspace(-0.95, 0.95, steps=100).unsqueeze(1)
    hi, lo = encode_words_from_scalar(y, word_bits=16, target_bits=32)
    y_hat = decode_words_to_scalar(hi, lo, word_bits=16, target_bits=32)

    max_err = torch.max(torch.abs(y.squeeze(1) - y_hat.squeeze(1))).item()
    assert max_err < 1e-4


def test_word_round_trip_is_close_for_8_16():
    y = torch.linspace(-0.9, 0.9, steps=80).unsqueeze(1)
    hi, lo = encode_words_from_scalar(y, word_bits=8, target_bits=16)
    y_hat = decode_words_to_scalar(hi, lo, word_bits=8, target_bits=16)

    max_err = torch.max(torch.abs(y.squeeze(1) - y_hat.squeeze(1))).item()
    assert max_err < 1e-2


def test_bit_round_trip_is_close_default_32():
    y = torch.linspace(-0.95, 0.95, steps=64)
    bits = encode_bits_from_scalar(y, total_bits=32)
    y_hat = decode_bits_to_scalar(bits)

    max_err = torch.max(torch.abs(y - y_hat)).item()
    assert max_err < 1e-4


def test_quantize_tensor_respects_bounds():
    x = torch.tensor([[-2.0, -0.5, 0.25, 2.0]])
    q = quantize_tensor(x, num_bits=8, x_min=-1.0, x_max=1.0)

    assert torch.all(q >= -1.0)
    assert torch.all(q <= 1.0)
