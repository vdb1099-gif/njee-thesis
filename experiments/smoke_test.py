import argparse
import json
from pathlib import Path

from src.simulation.factory import (
    generate_dataset,
    get_ground_truth,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a smoke test for a synthetic NJEE scenario."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where the smoke test result will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    config = load_config(config_path)

    seed = config["seed"]
    generator_name = config["data"]["generator"]

    set_seed(seed)

    X = generate_dataset(config)
    H_true = get_ground_truth(config)

    flip_rate = None

    if generator_name == "binary_chain":
        d = config["data"]["d"]

        if d > 1:
            flip_rate = float((X[:, 1:] != X[:, :-1]).mean())

    print("=== SMOKE TEST ===")
    print(f"Generator: {generator_name}")
    print(f"Seed: {seed}")
    print(f"Shape: {X.shape}")
    print(f"Ground truth: {H_true:.6f} nats")
    if generator_name == "binary_chain":
        if flip_rate is None:
            print("Empirical flip rate: not applicable for d = 1")
        else:
            print(f"Empirical flip rate: {flip_rate:.6f}")

    result = {
        "seed": seed,
        "generator": generator_name,
        "n": config["data"]["n"],
        "d": config["data"]["d"],
        "params": config["data"]["params"],
        "dataset_shape": list(X.shape),
        "true_entropy_nats": float(H_true),
    }

    if generator_name == "binary_chain":
        result["empirical_flip_rate"] = flip_rate

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "smoke_test_result.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print(f"Risultati salvati in: {output_path}")


if __name__ == "__main__":
    main()

