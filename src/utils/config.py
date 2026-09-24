from pathlib import Path
from typing import Any

import yaml


ALLOWED_GENERATORS = {
    "independent_bernoulli",
    "binary_chain",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Carica e valida un file YAML di configurazione.

    Parameters
    ----------
    path : str | Path
        Percorso del file YAML.

    Returns
    -------
    dict[str, Any]
        Configurazione validata.

    Raises
    ------
    FileNotFoundError
        Se il file non esiste.
    ValueError
        Se il file YAML è vuoto, malformato o contiene
        valori non validi.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML syntax in configuration file '{path}': {exc}"
        ) from exc

    if config is None:
        raise ValueError(
            f"Configuration file '{path}' is empty."
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{path}' must contain a YAML mapping."
        )

    validate_config(config, path)

    return config


def validate_config(
    config: dict[str, Any],
    path: str | Path = "<config>",
) -> None:
    """
    Valida struttura e valori della configurazione.
    """

    path = Path(path)

    # ----- seed -----

    if "seed" not in config:
        raise ValueError(
            f"Missing key 'seed' in configuration file '{path}'."
        )

    seed = config["seed"]

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            f"Invalid value for 'seed' in '{path}': "
            f"expected int, got {seed!r}."
        )

    # ----- data -----

    if "data" not in config:
        raise ValueError(
            f"Missing key 'data' in configuration file '{path}'."
        )

    data = config["data"]

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid value for 'data' in '{path}': "
            "expected a mapping."
        )

    required_data_keys = {
        "generator",
        "n",
        "d",
        "params",
    }

    for key in required_data_keys:
        if key not in data:
            raise ValueError(
                f"Missing key 'data.{key}' "
                f"in configuration file '{path}'."
            )

    # ----- n -----

    n = data["n"]

    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(
            f"Invalid value for 'data.n' in '{path}': "
            f"expected int, got {n!r}."
        )

    if n <= 0:
        raise ValueError(
            f"Invalid value for 'data.n' in '{path}': "
            f"expected n > 0, got {n!r}."
        )

    # ----- d -----

    d = data["d"]

    if not isinstance(d, int) or isinstance(d, bool):
        raise ValueError(
            f"Invalid value for 'data.d' in '{path}': "
            f"expected int, got {d!r}."
        )

    if d <= 0:
        raise ValueError(
            f"Invalid value for 'data.d' in '{path}': "
            f"expected d > 0, got {d!r}."
        )

    # ----- generator -----

    generator = data["generator"]

    if not isinstance(generator, str):
        raise ValueError(
            f"Invalid value for 'data.generator' in '{path}': "
            f"expected str, got {generator!r}."
        )

    if generator not in ALLOWED_GENERATORS:
        raise ValueError(
            f"Unknown generator '{generator}' in '{path}'. "
            f"Allowed generators: {sorted(ALLOWED_GENERATORS)}."
        )

    # ----- params -----

    params = data["params"]

    if not isinstance(params, dict):
        raise ValueError(
            f"Invalid value for 'data.params' in '{path}': "
            "expected a mapping."
        )

    if generator == "independent_bernoulli":
        _validate_independent_bernoulli_params(params, path)

    elif generator == "binary_chain":
        _validate_binary_chain_params(params, path)


def _validate_independent_bernoulli_params(
    params: dict[str, Any],
    path: Path,
) -> None:

    if "p" not in params:
        raise ValueError(
            f"Missing key 'data.params.p' in '{path}' "
            "for generator 'independent_bernoulli'."
        )

    if set(params) != {"p"}:
        raise ValueError(
            f"Invalid parameters for generator "
            f"'independent_bernoulli' in '{path}': "
            f"expected only 'p', got {sorted(params)}."
        )

    p = params["p"]

    if (
        not isinstance(p, (int, float))
        or isinstance(p, bool)
    ):
        raise ValueError(
            f"Invalid value for 'data.params.p' in '{path}': "
            f"expected a number, got {p!r}."
        )

    if not 0 <= p <= 1:
        raise ValueError(
            f"Invalid value for 'data.params.p' in '{path}': "
            f"expected 0 <= p <= 1, got {p!r}."
        )


def _validate_binary_chain_params(
    params: dict[str, Any],
    path: Path,
) -> None:

    if "epsilon" not in params:
        raise ValueError(
            f"Missing key 'data.params.epsilon' in '{path}' "
            "for generator 'binary_chain'."
        )

    if set(params) != {"epsilon"}:
        raise ValueError(
            f"Invalid parameters for generator "
            f"'binary_chain' in '{path}': "
            f"expected only 'epsilon', got {sorted(params)}."
        )

    epsilon = params["epsilon"]

    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
    ):
        raise ValueError(
            f"Invalid value for 'data.params.epsilon' in '{path}': "
            f"expected a number, got {epsilon!r}."
        )

    if not 0 <= epsilon <= 1:
        raise ValueError(
            f"Invalid value for 'data.params.epsilon' in '{path}': "
            f"expected 0 <= epsilon <= 1, got {epsilon!r}."
        )