import torch
import torch.nn as nn
import torch.nn.functional as F


QUANTIZED_HIDDEN_MODES = {"constrained_nbit", "quant_full", "quant_io_hidden"}


def _ste_round(x):
    return x + (torch.round(x) - x).detach()


def _quantize_activation(x, bits, clip_value):
    levels = (1 << bits) - 1
    x_clamped = torch.clamp(x, -clip_value, clip_value)
    x_norm = (x_clamped + clip_value) / (2.0 * clip_value)
    x_q = _ste_round(x_norm * levels) / levels
    return x_q * (2.0 * clip_value) - clip_value


def _quantize_signed_tensor(x, bits):
    levels = (1 << (bits - 1)) - 1
    scale = torch.clamp(x.detach().abs().max(), min=1e-8)
    x_norm = torch.clamp(x / scale, -1.0, 1.0)
    x_q = _ste_round(x_norm * levels) / levels
    return x_q * scale


def _use_quantized_hidden_compute(quant_cfg):
    return quant_cfg["mode"] in QUANTIZED_HIDDEN_MODES


def _linear_forward(linear, x, quant_cfg):
    if not _use_quantized_hidden_compute(quant_cfg):
        return linear(x)

    weight = _quantize_signed_tensor(linear.weight, quant_cfg["weight_bits"])
    bias = linear.bias
    if bias is not None:
        bias = _quantize_signed_tensor(bias, quant_cfg["weight_bits"])
    return F.linear(x, weight, bias)


def _post_activation(x, quant_cfg):
    if not _use_quantized_hidden_compute(quant_cfg):
        return x
    return _quantize_activation(x, quant_cfg["activation_bits"], quant_cfg["activation_clip"])


class MLPBackbone(nn.Module):
    def __init__(self, d_in, width, depth, quant_cfg):
        super().__init__()
        self.quant_cfg = quant_cfg
        self.layers = nn.ModuleList()
        in_features = d_in
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, width))
            in_features = width

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = _linear_forward(layer, h, self.quant_cfg)
            h = F.relu(h)
            h = _post_activation(h, self.quant_cfg)
        return h


class ScalarModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, quant_cfg=None):
        super().__init__()
        self.quant_cfg = quant_cfg or {}
        self.backbone = MLPBackbone(d_in, width, depth, self.quant_cfg)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        y = _linear_forward(self.head, h, self.quant_cfg).squeeze(-1)
        return {"y": y}


class TwoWordModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, quant_cfg=None):
        super().__init__()
        self.quant_cfg = quant_cfg or {}
        self.backbone = MLPBackbone(d_in, width, depth, self.quant_cfg)
        self.hi_head = nn.Linear(width, 1)
        self.lo_head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(_linear_forward(self.hi_head, h, self.quant_cfg)).squeeze(-1)
        lo = torch.sigmoid(_linear_forward(self.lo_head, h, self.quant_cfg)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class CoarseResidualModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, quant_cfg=None):
        super().__init__()
        self.quant_cfg = quant_cfg or {}
        self.backbone = MLPBackbone(d_in, width, depth, self.quant_cfg)
        self.coarse_head = nn.Linear(width, 1)
        self.residual_head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(_linear_forward(self.coarse_head, h, self.quant_cfg)).squeeze(-1)
        lo = torch.sigmoid(_linear_forward(self.residual_head, h, self.quant_cfg)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class SequentialModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, quant_cfg=None):
        super().__init__()
        self.quant_cfg = quant_cfg or {}
        self.backbone = MLPBackbone(d_in, width, depth, self.quant_cfg)
        self.hi_head = nn.Linear(width, 1)
        self.lo_in = nn.Linear(width + 1, width)
        self.lo_out = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(_linear_forward(self.hi_head, h, self.quant_cfg)).squeeze(-1)

        lo_hidden_in = torch.cat([h, hi.unsqueeze(-1)], dim=-1)
        lo_hidden = _linear_forward(self.lo_in, lo_hidden_in, self.quant_cfg)
        lo_hidden = F.relu(lo_hidden)
        lo_hidden = _post_activation(lo_hidden, self.quant_cfg)
        lo = torch.sigmoid(_linear_forward(self.lo_out, lo_hidden, self.quant_cfg)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class BitwiseModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, n_bits=32, quant_cfg=None):
        super().__init__()
        self.n_bits = n_bits
        self.quant_cfg = quant_cfg or {}
        self.backbone = MLPBackbone(d_in, width, depth, self.quant_cfg)
        self.head = nn.Linear(width, n_bits)

    def forward(self, x):
        h = self.backbone(x)
        bits = torch.sigmoid(_linear_forward(self.head, h, self.quant_cfg))
        return {"bits": bits}


def build_model(cfg):
    d_in = cfg["data"]["d_in"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    name = cfg["model"]["name"]

    precision_cfg = cfg["precision"]
    quant_cfg = {
        "mode": cfg["experiment"]["mode"],
        "activation_bits": precision_cfg.get("activation_bits", precision_cfg["input_bits"]),
        "weight_bits": precision_cfg.get("weight_bits", precision_cfg["input_bits"]),
        "activation_clip": precision_cfg.get("activation_clip", 2.0),
    }

    if name == "scalar":
        return ScalarModel(d_in, width, depth, quant_cfg=quant_cfg)
    if name == "upper_bound":
        float_quant_cfg = dict(quant_cfg)
        float_quant_cfg["mode"] = "float_full"
        return ScalarModel(d_in, width, depth, quant_cfg=float_quant_cfg)
    if name == "two_word":
        return TwoWordModel(d_in, width, depth, quant_cfg=quant_cfg)
    if name == "coarse_residual":
        return CoarseResidualModel(d_in, width, depth, quant_cfg=quant_cfg)
    if name == "sequential":
        return SequentialModel(d_in, width, depth, quant_cfg=quant_cfg)
    if name == "bitwise":
        return BitwiseModel(
            d_in,
            width,
            depth,
            n_bits=cfg["precision"]["target_bits"],
            quant_cfg=quant_cfg,
        )

    raise ValueError(f"Unknown model name: {name}")
