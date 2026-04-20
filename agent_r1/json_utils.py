from collections.abc import Mapping, Sequence

import numpy as np
import torch


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return make_json_safe(value.detach().cpu().tolist())

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, Mapping):
        return {key: make_json_safe(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [make_json_safe(item) for item in value]

    return value
