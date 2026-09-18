import numpy as np


def generate_independent_bernoulli(
    n: int,
    d: int,
    p: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Genera un dataset binario con componenti Bernoulli indipendenti.

    Parameters
    ----------
    n : int
        Numero di osservazioni.

    d : int
        Numero di variabili/dimensioni.

    p : float, default=0.5
        Probabilità di ottenere 1 per ciascuna componente.

    seed : int or None, default=None
        Seed utilizzato per rendere riproducibile la generazione.

    Returns
    -------
    np.ndarray
        Matrice di forma (n, d) contenente valori 0 e 1.
    """

    if n <= 0:
        raise ValueError("n deve essere maggiore di 0.")

    if d <= 0:
        raise ValueError("d deve essere maggiore di 0.")

    if not 0 <= p <= 1:
        raise ValueError("p deve essere compreso tra 0 e 1.")

    rng = np.random.default_rng(seed)

    X = rng.binomial(
        n=1, #una variabile binomiale con n=1 è una variabile Bernoulli
        p=p,
        size=(n, d), # d variabili bernulli indipendenti per n osservazioni
    )

    return X