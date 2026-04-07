from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encoding import encode_fixed32, quantize_uniform
from losses import build_loss
from metrics import summarize_metrics
from models import build_model
from synthetic_data import make_dataset
from utils import ensure_dir


def _make_bit_targets(y_int: torch.Tensor) -> torch.Tensor:
    y_u32 = y_int & 0xFFFFFFFF
    shifts = torch.arange(31, -1, -1, dtype=torch.int64)
    bits = ((y_u32.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    return bits


def _build_split(cfg: Dict, split: str, seed_offset: int = 0) -> TensorDataset:
    n_samples = cfg["data"][f"n_{split}"]
    x, y, _ = make_dataset(
        task=cfg["data"]["task"],
        n_samples=n_samples,
        d_in=cfg["data"]["d_in"],
        noise_std=cfg["data"].get("noise_std", 0.0),
        seed=cfg["seed"] + seed_offset,
    )

    x_q = quantize_uniform(x, bits=cfg["precision"]["input_bits"], x_min=-1.0, x_max=1.0)
    hi, lo, y_int = encode_fixed32(y)
    bits = _make_bit_targets(y_int.squeeze(1))
    return TensorDataset(x_q.float(), y.float(), hi.float(), lo.float(), bits.float())


def _make_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = _build_split(cfg, "train", seed_offset=0)
    val_ds = _build_split(cfg, "val", seed_offset=1)
    test_ds = _build_split(cfg, "test", seed_offset=2)

    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _step_targets(y: torch.Tensor, hi: torch.Tensor, lo: torch.Tensor, bits: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"y": y, "hi": hi, "lo": lo, "bits": bits}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, cfg: Dict, device: str) -> Dict[str, float]:
    model.eval()
    all_outputs = {"y": [], "hi": [], "lo": []}
    need_bits = cfg["model"]["name"] == "bitwise"
    if need_bits:
        all_outputs["bits"] = []

    all_targets = {"y": [], "hi": [], "lo": [], "bits": []}

    for x, y, hi, lo, bits in loader:
        x = x.to(device)
        y = y.to(device)
        hi = hi.to(device)
        lo = lo.to(device)
        bits = bits.to(device)
        outputs = model(x)

        all_outputs["y"].append(outputs["y"].cpu())
        if "hi" in outputs:
            all_outputs["hi"].append(outputs["hi"].cpu())
            all_outputs["lo"].append(outputs["lo"].cpu())
        if need_bits:
            all_outputs["bits"].append(outputs["bits"].cpu())

        all_targets["y"].append(y.cpu())
        all_targets["hi"].append(hi.cpu())
        all_targets["lo"].append(lo.cpu())
        all_targets["bits"].append(bits.cpu())

    merged_outputs = {k: torch.cat(v) for k, v in all_outputs.items() if len(v) > 0}
    merged_targets = {k: torch.cat(v) for k, v in all_targets.items() if len(v) > 0}
    return summarize_metrics(merged_outputs, merged_targets)


def run_training(cfg: Dict, device: str = "cpu") -> Dict[str, float]:
    output_dir = ensure_dir(cfg.get("output_dir", "outputs"))
    train_loader, val_loader, test_loader = _make_loaders(cfg)

    model = build_model(
        model_name=cfg["model"]["name"],
        d_in=cfg["data"]["d_in"],
        width=cfg["model"].get("width", 128),
        depth=cfg["model"].get("depth", 3),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"].get("lr", 1e-3),
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for x, y, hi, lo, bits in train_loader:
            x = x.to(device)
            y = y.to(device)
            hi = hi.to(device)
            lo = lo.to(device)
            bits = bits.to(device)

            outputs = model(x)
            targets = _step_targets(y, hi, lo, bits)

            loss = build_loss(cfg["loss"]["name"], outputs, targets, cfg["loss"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        val_metrics = _evaluate(model, val_loader, cfg, device)
        avg_train_loss = running_loss / len(train_loader.dataset)
        print(
            f"epoch={epoch:03d} train_loss={avg_train_loss:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = output_dir / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return _evaluate(model, test_loader, cfg, device)
