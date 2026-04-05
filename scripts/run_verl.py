"""Thin wrapper that patches json to handle numpy types, then launches verl."""

import json
import sys

import numpy as np

_original_default = json.JSONEncoder.default


def _numpy_safe_default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _original_default(self, obj)


json.JSONEncoder.default = _numpy_safe_default

from verl.trainer.main_ppo import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
