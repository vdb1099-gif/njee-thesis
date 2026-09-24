import numpy as np


def binary_entropy(p: float) -> float:
    """
    Calcola l'entropia di una variabile Bernoulli(p) in nats.

    Parameters
    ----------
    p : float
        Probabilità P(X = 1).

    Returns
    -------
    float
        Entropia binaria in nats.
    """

    if not 0 <= p <= 1:
        raise ValueError("p deve essere compreso tra 0 e 1.")

    if p == 0 or p == 1:
        return 0.0

    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def independent_bernoulli_entropy(
    p: float,
    d: int,
) -> float:
    """
    Calcola l'entropia congiunta di d variabili
    Bernoulli indipendenti con la stessa probabilità p.

    Parameters
    ----------
    p : float
        Probabilità P(X_j = 1).

    d : int
        Numero di componenti indipendenti.

    Returns
    -------
    float
        Entropia congiunta in nats.
    """

    if d <= 0:
        raise ValueError("d deve essere maggiore di 0.")

    return d * binary_entropy(p)



def binary_chain_entropy(
    epsilon: float,
    d: int,
) -> float:
    """
    Entropia teorica della catena binaria dipendente.

    H(X) = log(2) + (d - 1) h(epsilon)
    """

    if not 0 <= epsilon <= 1:
        raise ValueError(
            "epsilon deve essere compreso tra 0 e 1."
        )

    if d <= 0:
        raise ValueError(
            "d deve essere maggiore di 0."
        )

    return np.log(2) + (d - 1) * binary_entropy(epsilon)