import json
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quantize_tensor(x, num_bits=16, x_min=-1.0, x_max=1.0):
    levels = (2 ** num_bits) - 1
    x = torch.clamp(x, x_min, x_max)
    x_norm = (x - x_min) / (x_max - x_min)
    x_q = torch.round(x_norm * levels) / levels
    return x_q * (x_max - x_min) + x_min


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)