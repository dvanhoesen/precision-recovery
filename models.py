import torch
import torch.nn as nn


def _make_mlp(d_in, width, depth):
    layers = []
    in_features = d_in
    for _ in range(depth):
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.ReLU())
        in_features = width
    return nn.Sequential(*layers)


class ScalarModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3):
        super().__init__()
        self.backbone = _make_mlp(d_in, width, depth)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        y = self.head(h).squeeze(-1)
        return {"y": y}


class TwoWordModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3):
        super().__init__()
        self.backbone = _make_mlp(d_in, width, depth)
        self.hi_head = nn.Linear(width, 1)
        self.lo_head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(self.hi_head(h)).squeeze(-1)
        lo = torch.sigmoid(self.lo_head(h)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class CoarseResidualModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3):
        super().__init__()
        self.backbone = _make_mlp(d_in, width, depth)
        self.coarse_head = nn.Linear(width, 1)
        self.residual_head = nn.Linear(width, 1)

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(self.coarse_head(h)).squeeze(-1)
        lo = torch.sigmoid(self.residual_head(h)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class SequentialModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3):
        super().__init__()
        self.backbone = _make_mlp(d_in, width, depth)
        self.hi_head = nn.Linear(width, 1)
        self.lo_head = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        hi = torch.sigmoid(self.hi_head(h)).squeeze(-1)
        lo_in = torch.cat([h, hi.unsqueeze(-1)], dim=-1)
        lo = torch.sigmoid(self.lo_head(lo_in)).squeeze(-1)
        return {"hi": hi, "lo": lo}


class BitwiseModel(nn.Module):
    def __init__(self, d_in, width=128, depth=3, n_bits=32):
        super().__init__()
        self.n_bits = n_bits
        self.backbone = _make_mlp(d_in, width, depth)
        self.head = nn.Linear(width, n_bits)

    def forward(self, x):
        h = self.backbone(x)
        bits = torch.sigmoid(self.head(h))
        return {"bits": bits}


def build_model(cfg):
    d_in = cfg["data"]["d_in"]
    width = cfg["model"]["width"]
    depth = cfg["model"]["depth"]
    name = cfg["model"]["name"]

    if name == "scalar":
        return ScalarModel(d_in, width, depth)
    if name == "two_word":
        return TwoWordModel(d_in, width, depth)
    if name == "coarse_residual":
        return CoarseResidualModel(d_in, width, depth)
    if name == "sequential":
        return SequentialModel(d_in, width, depth)
    if name == "bitwise":
        return BitwiseModel(d_in, width, depth, n_bits=cfg["precision"]["target_bits"])

    raise ValueError(f"Unknown model name: {name}")