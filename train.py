import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from synthetic_data import generate_synthetic_data
from models import build_model
from losses import build_loss_fn
from metrics import compute_metrics
from encoding import encode_words_from_scalar, encode_bits_from_scalar
from utils import quantize_tensor, save_json


def prepare_batch(x, y, cfg, device):
    mode = cfg["experiment"]["mode"]
    model_name = cfg["model"]["name"]
    word_bits = cfg["precision"]["word_bits"]
    input_bits = cfg["precision"]["input_bits"]
    quantize_inputs = cfg["precision"].get("quantize_inputs", True)

    if mode == "constrained_nbit" and quantize_inputs:
        x = quantize_tensor(x, num_bits=input_bits)

    batch = {
        "x": x.to(device),
        "y": y.to(device),
        "word_bits": word_bits,
    }

    if model_name in {"two_word", "coarse_residual", "sequential"}:
        y_hi, y_lo = encode_words_from_scalar(y, word_bits=word_bits)
        batch["y_hi"] = y_hi.to(device)
        batch["y_lo"] = y_lo.to(device)

    if model_name == "bitwise":
        batch["y_bits"] = encode_bits_from_scalar(y, total_bits=cfg["precision"]["target_bits"]).to(device)

    return batch


def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, loss_fn, cfg, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        batch = prepare_batch(x, y, cfg, device)
        optimizer.zero_grad()
        outputs = model(batch["x"])
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    model_name = cfg["model"]["name"]

    preds = []
    targets = []
    all_hi_pred = []
    all_lo_pred = []
    all_hi_true = []
    all_lo_true = []
    all_bits_pred = []
    all_bits_true = []

    for x, y in loader:
        batch = prepare_batch(x, y, cfg, device)
        outputs = model(batch["x"])

        preds.append(outputs)
        targets.append(batch)

        if model_name in {"two_word", "coarse_residual", "sequential"}:
            all_hi_pred.append(outputs["hi"].cpu())
            all_lo_pred.append(outputs["lo"].cpu())
            all_hi_true.append(batch["y_hi"].cpu())
            all_lo_true.append(batch["y_lo"].cpu())
        elif model_name == "bitwise":
            all_bits_pred.append(outputs["bits"].cpu())
            all_bits_true.append(batch["y_bits"].cpu())
        else:
            pass

    if model_name == "scalar":
        out = {"y": torch.cat([p["y"].cpu() for p in preds], dim=0)}
        batch = {"y": torch.cat([b["y"].cpu() for b in targets], dim=0)}
    elif model_name in {"two_word", "coarse_residual", "sequential"}:
        out = {
            "hi": torch.cat(all_hi_pred, dim=0),
            "lo": torch.cat(all_lo_pred, dim=0),
        }
        batch = {
            "y": torch.cat([b["y"].cpu() for b in targets], dim=0),
            "y_hi": torch.cat(all_hi_true, dim=0),
            "y_lo": torch.cat(all_lo_true, dim=0),
            "word_bits": cfg["precision"]["word_bits"],
        }
    elif model_name == "bitwise":
        out = {"bits": torch.cat(all_bits_pred, dim=0)}
        batch = {
            "y": torch.cat([b["y"].cpu() for b in targets], dim=0),
            "y_bits": torch.cat(all_bits_true, dim=0),
        }
    else:
        raise ValueError(model_name)

    return compute_metrics(out, batch, model_name)


def run_experiment(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = generate_synthetic_data(
        task=cfg["data"]["task"],
        n_samples=cfg["data"]["n_train"],
        d_in=cfg["data"]["d_in"],
        noise_std=cfg["data"]["noise_std"],
        seed=cfg["seed"],
    )
    x_val, y_val = generate_synthetic_data(
        task=cfg["data"]["task"],
        n_samples=cfg["data"]["n_val"],
        d_in=cfg["data"]["d_in"],
        noise_std=cfg["data"]["noise_std"],
        seed=cfg["seed"] + 1,
    )
    x_test, y_test = generate_synthetic_data(
        task=cfg["data"]["task"],
        n_samples=cfg["data"]["n_test"],
        d_in=cfg["data"]["d_in"],
        noise_std=cfg["data"]["noise_std"],
        seed=cfg["seed"] + 2,
    )

    train_loader = make_loader(x_train, y_train, cfg["train"]["batch_size"], shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg["train"]["batch_size"], shuffle=False)
    test_loader = make_loader(x_test, y_test, cfg["train"]["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    loss_fn = build_loss_fn(cfg)

    best_val = float("inf")
    best_state = None

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        val_rmse = val_metrics["rmse"]

        print(
            f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | "
            f"val_rmse={val_rmse:.6f} | val_mae={val_metrics['mae']:.6f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, cfg, device)
    test_metrics["experiment_mode"] = cfg["experiment"]["mode"]
    test_metrics["model_name"] = cfg["model"]["name"]
    test_metrics["task"] = cfg["data"]["task"]

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    save_json(
        os.path.join(cfg["output"]["save_dir"], f"{cfg['experiment_name']}_metrics.json"),
        test_metrics,
    )

    return test_metrics