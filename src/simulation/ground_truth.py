import numpy as np


def exact_entropy(probabilities: np.ndarray) -> float:
    """Exact entropy in nats of a finite, explicitly enumerated PMF.

    Accept a probability vector over joint states or a tensor whose axes
    follow component order. Sum over all entries; zero-mass states contribute
    zero. This is a population calculation, not a sample estimator. The PMF
    must already be normalized (absolute tolerance 1e-12); no renormalization
    or enumeration of an exponentially large alphabet is performed here.
    Numeric lists and arrays are accepted, but MaskedArray inputs are rejected.
    Normalization is checked after conversion to float64: float32 PMFs may fail
    the tolerance. Construct probabilities at float64 precision; casting an
    already rounded float32 PMF does not recover precision.
    """
    if np.ma.isMaskedArray(probabilities):
        raise ValueError("probabilities must not be a MaskedArray; masked inputs are unsupported.")
    raw = np.asarray(probabilities)
    if raw.dtype.kind not in "biuf":
        raise ValueError("probabilities must be real numeric values.")
    pmf = raw.astype(float)
    if pmf.ndim == 0 or pmf.size == 0:
        raise ValueError("probabilities must be a non-empty vector or tensor.")
    if not np.isfinite(pmf).all() or (pmf < 0).any() or (pmf > 1).any():
        raise ValueError("probabilities must be finite and between 0 and 1.")
    if not np.isclose(pmf.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to 1.")
    positive = pmf[pmf > 0]
    return float(-np.sum(positive * np.log(positive)))


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
