import random

import numpy as np
import torch

from src.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(42)

    python_1 = random.random()
    numpy_1 = np.random.rand()
    torch_1 = torch.rand(1)

    set_seed(42)

    python_2 = random.random()
    numpy_2 = np.random.rand()
    torch_2 = torch.rand(1)

    assert python_1 == python_2
    assert numpy_1 == numpy_2
    assert torch.equal(torch_1, torch_2)