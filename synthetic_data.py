import torch


def generate_synthetic_data(task, n_samples, d_in, noise_std=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = 2.0 * torch.rand(n_samples, d_in, generator=g) - 1.0

    if task == "linear":
        w = torch.randn(d_in, generator=g)
        y = x @ w

    elif task == "polynomial":
        y = (x ** 2).sum(dim=1) + 0.5 * x[:, 0] * x[:, 1]

    elif task == "multiplicative":
        y = torch.prod(1.0 + 0.5 * x, dim=1) - 1.0

    elif task == "oscillatory":
        freqs = torch.arange(1, d_in + 1, dtype=torch.float32)
        y = sum(torch.sin(freqs[i] * x[:, i]) for i in range(d_in))

    else:
        raise ValueError(f"Unknown task: {task}")

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y, generator=g)

    y = y / (y.abs().max() + 1e-8)
    return x.float(), y.float()