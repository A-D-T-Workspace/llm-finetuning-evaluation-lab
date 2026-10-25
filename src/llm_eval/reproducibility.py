import random

import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible ML experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

import random

import numpy as np

from src.llm_eval.reproducibility import set_seed

def test_python_random_seed_is_reproducible():
    set_seed(42)
    first = random.random()

    set_seed(42)
    second = random.random()

    assert first == second

def test_numpy_random_seed_is_reproducible():
    set_seed(42)
    first = np.random.rand(5)

    set_seed(42)
    second = np.random.rand(5)

    assert np.array_equal(first, second)