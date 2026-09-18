import numpy as np


def generate_binary_chain(
    n: int,
    d: int,
    epsilon: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Genera una catena binaria dipendente.

    Il processo è definito da:

        X_1 ~ Bernoulli(1/2)
        X_j = X_{j-1} XOR E_j
        E_j ~ Bernoulli(epsilon)

    Parameters
    ----------
    n : int
        Numero di osservazioni.

    d : int
        Numero di componenti del vettore.

    epsilon : float
        Probabilità di flip tra una componente
        e la successiva.

    seed : int or None, default=None
        Seed del generatore pseudo-casuale.

    Returns
    -------
    np.ndarray
        Matrice binaria di forma (n, d).
    """

    if n <= 0:
        raise ValueError("n deve essere maggiore di 0.")

    if d <= 0:
        raise ValueError("d deve essere maggiore di 0.")

    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon deve essere compreso tra 0 e 1.")

    rng = np.random.default_rng(seed)

    X = np.empty((n, d), dtype=np.int8)

    # Prima componente: Bernoulli(1/2)
    X[:, 0] = rng.binomial(
        n=1,
        p=0.5,
        size=n,
    )

    # Componenti successive
    for j in range(1, d):
        E = rng.binomial(
            n=1,
            p=epsilon,
            size=n,
        )

        X[:, j] = np.bitwise_xor(
            X[:, j - 1],
            E,
        )

    return X