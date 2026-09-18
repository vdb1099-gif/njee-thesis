import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Imposta il seed per Python, NumPy e PyTorch.

    Parameters
    ----------
    seed : int
        Seed utilizzato per rendere riproducibili
        le operazioni pseudo-casuali.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)