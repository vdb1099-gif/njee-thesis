from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Carica una configurazione YAML.

    Parameters
    ----------
    path : str or Path
        Percorso del file YAML.

    Returns
    -------
    dict
        Configurazione caricata come dizionario Python.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config