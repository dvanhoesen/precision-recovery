import torch
import torch.nn.functional as F
from encoding import decode_words_to_scalar, decode_bits_to_scalar


def scalar_loss(outputs, batch):
    return F.mse_loss(outputs["y"], batch["y"])


def two_word_loss(outputs, batch, lambda_hi=0.2, lambda_lo=0.2):
    pred_y = decode_words_to_scalar(
        outputs["hi"],
        outputs["lo"],
        word_bits=batch["word_bits"],
        target_bits=batch["target_bits"],
    )
    recon_loss = F.mse_loss(pred_y, batch["y"])
    hi_loss = F.mse_loss(outputs["hi"], batch["y_hi"])
    lo_loss = F.mse_loss(outputs["lo"], batch["y_lo"])
    return recon_loss + lambda_hi * hi_loss + lambda_lo * lo_loss


def bitwise_loss(outputs, batch):
    bits = outputs["bits"]
    target_bits = batch["y_bits"]
    bit_loss = F.binary_cross_entropy(bits, target_bits)
    pred_y = decode_bits_to_scalar(bits)
    recon_loss = F.mse_loss(pred_y, batch["y"])
    return recon_loss + 0.1 * bit_loss


def build_loss_fn(cfg):
    model_name = cfg["model"]["name"]
    lambda_hi = cfg["loss"].get("lambda_hi", 0.2)
    lambda_lo = cfg["loss"].get("lambda_lo", 0.2)

    if model_name == "scalar":
        return scalar_loss
    if model_name in {"two_word", "coarse_residual", "sequential"}:
        return lambda outputs, batch: two_word_loss(
            outputs, batch, lambda_hi=lambda_hi, lambda_lo=lambda_lo
        )
    if model_name == "bitwise":
        return bitwise_loss

    raise ValueError(f"Unsupported model for loss: {model_name}")
