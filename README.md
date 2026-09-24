# NJEE Thesis Project

Implementation and validation framework for Neural Joint Entropy Estimation (NJEE) and related entropy-estimation experiments.

## Requirements

- Python 3.11 or newer
- Git
- Recommended: virtual environment

The project is currently developed and tested with Python 3.13.

## Installation

Clone the repository:

```bash
git clone https://github.com/vdb1099-gif/njee-thesis.git
cd njee-thesis
```

### Windows PowerShell

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Project structure

```text
njee-thesis/
├── configs/              # YAML experiment configurations
├── experiments/          # Executable experiment and smoke-test scripts
├── results/              # Generated outputs (not version-controlled)
├── src/
│   ├── estimators/       # Entropy estimators
│   ├── metrics/          # Evaluation metrics
│   ├── models/           # Neural models
│   ├── simulation/       # Synthetic generators and analytical ground truth
│   └── utils/            # Configuration and reproducibility utilities
├── tests/                # Automated tests
├── pytest.ini
├── requirements.txt
└── README.md
```

## Configuration

Experiments are controlled through YAML configuration files.

The default configuration file is:

```text
configs/baseline.yaml
```

Only one generator is selected at a time, together with the parameters required by that generator.

Example: independent Bernoulli variables

```yaml
seed: 42

data:
  generator: independent_bernoulli
  n: 1000
  d: 8
  params:
    p: 0.5
```

Example: dependent binary chain

```yaml
seed: 42

data:
  generator: binary_chain
  n: 1000
  d: 8
  params:
    epsilon: 0.15
```

## Run the test suite

Run all automated tests with:

```powershell
python -m pytest -v
```

Pytest is configured through `pytest.ini` to collect tests only from the `tests/` directory.

## Run the smoke test

Run the default smoke test with:

```powershell
python -m experiments.smoke_test
```

By default, the command reads:

```text
configs/baseline.yaml
```

and writes generated output to:

```text
results/
```

A different configuration file or output directory can be specified through the command line:

```powershell
python -m experiments.smoke_test --config configs/baseline.yaml --output-dir results
```

## Generated results

Smoke-test outputs are generated automatically and are not version-controlled.

They can be reproduced by running:

```powershell
python -m experiments.smoke_test
```

## Reproducibility

The project currently supports seeded execution for:

- Python
- NumPy
- PyTorch
- CUDA, when available

Seeded execution improves reproducibility but does not necessarily guarantee strict deterministic execution for every GPU operation.

Additional deterministic settings will be introduced where required during the neural-estimator implementation.

## Current status

The current implementation includes:

- independent Bernoulli synthetic generator;
- dependent binary-chain synthetic generator;
- analytical ground-truth entropy calculations;
- YAML-based experiment configuration;
- generator factory;
- configuration validation;
- seed and reproducibility utilities;
- command-line configurable smoke-test pipeline;
- automated unit tests.

The neural entropy estimators and the full experimental evaluation pipeline are still under development.