import torch

from synthetic_data import generate_synthetic_data, sample_task_params


def test_linear_uses_shared_task_params_across_splits():
    params = sample_task_params(task="linear", d_in=8, seed=123)

    x_train, y_train, target_scale = generate_synthetic_data(
        task="linear",
        n_samples=128,
        d_in=8,
        noise_std=0.0,
        seed=0,
        task_params=params,
        return_scale=True,
    )
    x_val, y_val = generate_synthetic_data(
        task="linear",
        n_samples=128,
        d_in=8,
        noise_std=0.0,
        seed=1,
        task_params=params,
        target_scale=target_scale,
    )

    # Different seeds should produce different x samples.
    assert not torch.allclose(x_train, x_val)

    # With shared params and scale, val target should match deterministic projection.
    expected_val = (x_val @ params["w"]) / target_scale
    assert torch.allclose(y_val, expected_val, atol=1e-6, rtol=1e-6)


def test_target_scale_is_respected_when_provided():
    params = sample_task_params(task="linear", d_in=4, seed=9)

    _, y_train, target_scale = generate_synthetic_data(
        task="linear",
        n_samples=64,
        d_in=4,
        noise_std=0.0,
        seed=2,
        task_params=params,
        return_scale=True,
    )
    x_test, y_test = generate_synthetic_data(
        task="linear",
        n_samples=64,
        d_in=4,
        noise_std=0.0,
        seed=3,
        task_params=params,
        target_scale=target_scale,
    )

    expected_test = (x_test @ params["w"]) / target_scale
    assert torch.allclose(y_test, expected_test, atol=1e-6, rtol=1e-6)

    # Sanity: training data was normalized with same scale convention.
    assert torch.max(torch.abs(y_train)).item() <= 1.0 + 1e-6
