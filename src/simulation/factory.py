from src.simulation.bernoulli import generate_independent_bernoulli
from src.simulation.binary_chain import generate_binary_chain

from src.simulation.ground_truth import (
    independent_bernoulli_entropy,
    binary_chain_entropy,
)


def generate_dataset(config: dict):
    """
    Genera il dataset indicato nella configurazione.

    Parameters
    ----------
    config : dict
        Configurazione completa dell'esperimento.

    Returns
    -------
    dataset
        Dataset sintetico generato.

    Raises
    ------
    ValueError
        Se il generatore non è riconosciuto o se mancano
        i parametri necessari.
    """

    seed = config["seed"]

    data_config = config["data"]

    generator_name = data_config["generator"]
    n = data_config["n"]
    d = data_config["d"]
    params = data_config["params"]

    if generator_name == "independent_bernoulli":

        if "p" not in params:
            raise ValueError(
                "Il generatore 'independent_bernoulli' richiede il parametro 'p'."
            )

        return generate_independent_bernoulli(
            n=n,
            d=d,
            p=params["p"],
            seed=seed,
        )

    elif generator_name == "binary_chain":

        if "epsilon" not in params:
            raise ValueError(
                "Il generatore 'binary_chain' richiede il parametro 'epsilon'."
            )

        return generate_binary_chain(
            n=n,
            d=d,
            epsilon=params["epsilon"],
            seed=seed,
        )

    else:
        raise ValueError(
            f"Generatore sconosciuto: '{generator_name}'."
        )


def get_ground_truth(config: dict) -> float:
    """
    Calcola la ground truth associata
    al generatore indicato nella configurazione.
    """

    data_config = config["data"]

    generator_name = data_config["generator"]
    d = data_config["d"]
    params = data_config["params"]

    if generator_name == "independent_bernoulli":

        if "p" not in params:
            raise ValueError(
                "Il generatore 'independent_bernoulli' richiede il parametro 'p'."
            )

        return independent_bernoulli_entropy(
            p=params["p"],
            d=d,
        )

    elif generator_name == "binary_chain":

        if "epsilon" not in params:
            raise ValueError(
                "Il generatore 'binary_chain' richiede il parametro 'epsilon'."
            )

        return binary_chain_entropy(
            epsilon=params["epsilon"],
            d=d,
        )

    else:
        raise ValueError(
            f"Generatore sconosciuto: '{generator_name}'."
        )