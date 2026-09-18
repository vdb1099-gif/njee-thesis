import json
from pathlib import Path

from src.simulation.bernoulli import generate_independent_bernoulli
from src.simulation.binary_chain import generate_binary_chain
from src.simulation.ground_truth import (
    independent_bernoulli_entropy,
    binary_chain_entropy,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    # 1. Carica configurazione
    config = load_config("configs/baseline.yaml")

    seed = config["seed"]
    n = config["data"]["n"]
    d = config["data"]["d"]
    p = config["generator"]["p"]
    epsilon = config["generator"]["epsilon"]

    # 2. Imposta seed globale
    set_seed(seed)

    # =========================================================
    # Scenario 1: Bernoulli indipendenti
    # =========================================================

    X_ind = generate_independent_bernoulli(
        n=n,
        d=d,
        p=p,
        seed=seed,
    )

    H_ind_true = independent_bernoulli_entropy(
        p=p,
        d=d,
    )

    # =========================================================
    # Scenario 2: Binary chain
    # =========================================================

    X_chain = generate_binary_chain(
        n=n,
        d=d,
        epsilon=epsilon,
        seed=seed,
    )

    H_chain_true = binary_chain_entropy(
        epsilon=epsilon,
        d=d,
    )

    flip_rate = (X_chain[:, 1:] != X_chain[:, :-1]).mean()

    # 3. Stampa risultati
    print("=== SMOKE TEST ===")

    print("\n[Independent Bernoulli]")
    print(f"Seed: {seed}")
    print(f"Shape: {X_ind.shape}")
    print(f"Media empirica: {X_ind.mean():.6f}")
    print(f"Entropia teorica: {H_ind_true:.6f} nats")

    print("\n[Binary Chain]")
    print(f"Seed: {seed}")
    print(f"Shape: {X_chain.shape}")
    print(f"Flip rate empirico: {flip_rate:.6f}")
    print(f"Epsilon teorico: {epsilon}")
    print(f"Entropia teorica: {H_chain_true:.6f} nats")

    # 4. Salvataggio risultati
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    result = {
        "seed": seed,
        "n": n,
        "d": d,
        "independent_bernoulli": {
            "p": p,
            "dataset_shape": list(X_ind.shape),
            "empirical_mean": float(X_ind.mean()),
            "true_entropy_nats": float(H_ind_true),
        },
        "binary_chain": {
            "epsilon": epsilon,
            "dataset_shape": list(X_chain.shape),
            "empirical_flip_rate": float(flip_rate),
            "true_entropy_nats": float(H_chain_true),
        },
    }

    output_path = results_dir / "smoke_test_result.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()