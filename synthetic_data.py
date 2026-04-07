import torch


def sample_task_params(task, d_in, seed=0):
    g = torch.Generator().manual_seed(seed)

    if task == "linear":
        return {"w": torch.randn(d_in, generator=g)}
    if task in {"polynomial", "multiplicative", "oscillatory"}:
        return {}

    raise ValueError(f"Unknown task: {task}")


def generate_synthetic_data(
    task,
    n_samples,
    d_in,
    noise_std=0.0,
    seed=0,
    task_params=None,
    target_scale=None,
    return_scale=False,
):
    g = torch.Generator().manual_seed(seed)
    x = 2.0 * torch.rand(n_samples, d_in, generator=g) - 1.0

    if task_params is None:
        task_params = sample_task_params(task, d_in, seed=seed)

    if task == "linear":
        w = task_params.get("w")
        if w is None:
            raise ValueError("Missing task parameter 'w' for linear task")
        y = x @ w

    elif task == "polynomial":
        y = (x ** 2).sum(dim=1) + 0.5 * x[:, 0] * x[:, 1]

    elif task == "multiplicative":
        y = torch.prod(1.0 + 0.5 * x, dim=1) - 1.0

    elif task == "oscillatory":
        freqs = torch.arange(1, d_in + 1, dtype=torch.float32)
        terms = [torch.sin(freqs[i] * x[:, i]) for i in range(d_in)]
        y = torch.stack(terms, dim=0).sum(dim=0)

    else:
        raise ValueError(f"Unknown task: {task}")

    if noise_std > 0:
        noise = torch.randn(y.shape, generator=g, dtype=y.dtype, device=y.device)
        y = y + noise_std * noise

    if target_scale is None:
        target_scale = y.abs().max() + 1e-8

    y = y / target_scale
    x = x.float()
    y = y.float()
    target_scale = float(target_scale)

    if return_scale:
        return x, y, target_scale
    return x, y
