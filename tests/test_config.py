from src.utils.config import load_config


def test_load_config():
    config = load_config("configs/baseline.yaml")

    assert config["seed"] == 42
    assert config["data"]["n"] == 1000
    assert config["data"]["d"] == 8
    assert config["data"]["generator"] == "independent_bernoulli"
    assert config["generator"]["p"] == 0.5
    assert config["generator"]["epsilon"] == 0.15