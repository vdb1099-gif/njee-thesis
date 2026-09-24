import pytest

from src.utils.config import load_config


def test_load_valid_binary_chain_config(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: binary_chain
  n: 1000
  d: 8
  params:
    epsilon: 0.15
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["seed"] == 42
    assert config["data"]["generator"] == "binary_chain"
    assert config["data"]["n"] == 1000
    assert config["data"]["d"] == 8
    assert config["data"]["params"]["epsilon"] == 0.15


def test_load_valid_bernoulli_config(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: independent_bernoulli
  n: 1000
  d: 8
  params:
    p: 0.5
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config["data"]["generator"] == "independent_bernoulli"
    assert config["data"]["params"]["p"] == 0.5


def test_empty_config(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_config(config_file)


def test_missing_config_file(tmp_path):
    config_file = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(config_file)


def test_negative_n(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: binary_chain
  n: -1
  d: 8
  params:
    epsilon: 0.15
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.n"):
        load_config(config_file)


def test_unknown_generator(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: unknown_generator
  n: 1000
  d: 8
  params: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown generator"):
        load_config(config_file)


def test_invalid_probability(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: independent_bernoulli
  n: 1000
  d: 8
  params:
    p: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.params.p"):
        load_config(config_file)


def test_invalid_epsilon(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: binary_chain
  n: 1000
  d: 8
  params:
    epsilon: -0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.params.epsilon"):
        load_config(config_file)


def test_missing_required_key(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: binary_chain
  d: 8
  params:
    epsilon: 0.15
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.n"):
        load_config(config_file)


def test_non_numeric_probability(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42

data:
  generator: independent_bernoulli
  n: 1000
  d: 8
  params:
    p: hello
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.params.p"):
        load_config(config_file)


def test_malformed_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
seed: 42
data:
  generator: binary_chain
  n: [1000
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_config(config_file)