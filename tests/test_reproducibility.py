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