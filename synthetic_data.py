from __future__ import annotations

from typing import Tuple

import torch


TensorTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _normalize_target(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize to [-1, 1)."""
    scale = y.abs().max().clamp_min(eps)
    y = y / scale
    return y.clamp(-0.999999, 0.999999)


class SyntheticGenerator:
    def __init__(self, d_in: int, noise_std: float = 0.0, seed: int = 0) -> None:
        self.d_in = d_in
        self.noise_std = noise_std
        self.generator = torch.Generator().manual_seed(seed)

    def sample_inputs(self, n_samples: int) -> torch.Tensor:
        return 2.0 * torch.rand((n_samples, self.d_in), generator=self.generator) - 1.0

    def _add_noise(self, y: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return y
        noise = self.noise_std * torch.randn(y.shape, generator=self.generator)
        return y + noise

    def linear(self, n_samples: int) -> TensorTriplet:
        x = self.sample_inputs(n_samples)
        w = torch.randn(self.d_in, generator=self.generator)
        b = torch.randn(1, generator=self.generator)
        y = x @ w + b
        y = self._add_noise(y)
        y = _normalize_target(y)
        return x, y.unsqueeze(1), x.clone()

    def polynomial(self, n_samples: int) -> TensorTriplet:
        x = self.sample_inputs(n_samples)
        a = torch.randn(self.d_in, generator=self.generator)
        y = (a * x.square()).sum(dim=1)
        for i in range(self.d_in):
            for j in range(i + 1, self.d_in):
                coeff = 0.05 * torch.randn(1, generator=self.generator).item()
                y = y + coeff * x[:, i] * x[:, j]
        y = self._add_noise(y)
        y = _normalize_target(y)
        return x, y.unsqueeze(1), x.clone()

    def multiplicative(self, n_samples: int) -> TensorTriplet:
        x = self.sample_inputs(n_samples)
        alpha = 0.5 * torch.randn(self.d_in, generator=self.generator)
        y = torch.prod(1.0 + alpha * x, dim=1) - 1.0
        y = self._add_noise(y)
        y = _normalize_target(y)
        return x, y.unsqueeze(1), x.clone()

    def oscillatory(self, n_samples: int) -> TensorTriplet:
        x = self.sample_inputs(n_samples)
        amp = torch.randn(self.d_in, generator=self.generator)
        freq = torch.randint(1, 6, (self.d_in,), generator=self.generator).float()
        phase = 2.0 * torch.pi * torch.rand(self.d_in, generator=self.generator)
        y = torch.zeros(n_samples)
        for i in range(self.d_in):
            y = y + amp[i] * torch.sin(freq[i] * x[:, i] + phase[i])
        y = self._add_noise(y)
        y = _normalize_target(y)
        return x, y.unsqueeze(1), x.clone()


def make_dataset(task: str, n_samples: int, d_in: int, noise_std: float = 0.0, seed: int = 0) -> TensorTriplet:
    gen = SyntheticGenerator(d_in=d_in, noise_std=noise_std, seed=seed)
    if task == "linear":
        return gen.linear(n_samples)
    if task == "polynomial":
        return gen.polynomial(n_samples)
    if task == "multiplicative":
        return gen.multiplicative(n_samples)
    if task == "oscillatory":
        return gen.oscillatory(n_samples)
    raise ValueError(f"Unknown task: {task}")
