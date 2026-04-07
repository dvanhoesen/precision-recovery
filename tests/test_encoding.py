import torch

from encoding import decode_fixed32_from_words, encode_fixed32, quantize_uniform


def test_fixed32_round_trip_is_close():
    y = torch.linspace(-0.95, 0.95, steps=100).unsqueeze(1)
    hi, lo, _ = encode_fixed32(y)
    y_hat = decode_fixed32_from_words(hi, lo)
    max_err = torch.max(torch.abs(y.squeeze(1) - y_hat.squeeze(1))).item()
    assert max_err < 1e-4


def test_quantize_uniform_respects_bounds():
    x = torch.tensor([[-2.0, -0.5, 0.25, 2.0]])
    q = quantize_uniform(x, bits=8, x_min=-1.0, x_max=1.0)
    assert torch.all(q >= -1.0)
    assert torch.all(q <= 1.0)
