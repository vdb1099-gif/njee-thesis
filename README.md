# NJEE Thesis Project

Implementation and validation framework for Neural Joint Entropy Estimation (NJEE) and related entropy-estimation experiments.

## Requirements

- Python 3.12 or newer for the pinned dependencies; Python 3.13 is recommended
- Git
- Recommended: virtual environment

The current environment is tested with Python 3.13.0. Python 3.12 is permitted
by the installed dependencies' Python-version metadata, but has not been tested
here. The roadmap's Python 3.11 suggestion is insufficient for the current pins.

## Installation

Clone the repository:

```bash
git clone https://github.com/vdb1099-gif/njee-thesis.git
cd njee-thesis
```

### Windows PowerShell

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
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
python -m pip install -r requirements.txt
```

Run the commands below from the repository root with this environment activated.

## Project structure

```text
njee-thesis/
├── configs/              # YAML experiment configurations
├── experiments/          # Executable experiment and smoke-test scripts
├── results/              # Generated outputs (not version-controlled)
├── src/
│   ├── estimators/       # Entropy estimators
│   ├── metrics/          # Placeholder for future evaluation metrics
│   ├── models/           # Placeholder for future neural models
│   ├── simulation/       # Generators, analytical ground truth and finite-PMF entropy
│   └── utils/            # Configuration and reproducibility utilities
├── tests/                # Automated tests
├── pytest.ini
├── requirements.txt
└── README.md
```

## Configuration

The synthetic smoke-test scenario is controlled through a YAML configuration file.

The default configuration file is:

```text
configs/baseline.yaml
```

Only one generator is selected at a time, together with the parameters required by that generator.
The checked-in configuration selects `binary_chain`. `n` and `d` must be positive
integers; `p` and `epsilon` must lie in `[0, 1]`. For the smoke test, use an integer
seed in `[0, 2**32 - 1]`, compatible with the NumPy global seed utility.

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

Relative `--config` and `--output-dir` paths are resolved against the repository
root; absolute paths are also accepted.

## Generated results

The smoke test writes `results/smoke_test_result.json` by default, creating the
output directory if needed. Reusing an output directory overwrites this file.
Outputs under `results/` are excluded from version control by `.gitignore`.

The JSON contains:

- `seed`, `generator`, `n`, `d`, `params`, and `dataset_shape`;
- `true_entropy_nats`: analytical ground truth for the configured generator;
- `entropy_unit`: `"nats"`, and `aggregation`: `"joint"`;
- `component_order`: `[0, ..., d-1]`, the supplied column order;
- `estimators.plug_in` and `estimators.miller_madow`, each containing
  `entropy_nats`, `error_nats`, and `runtime_seconds`;
- `empirical_flip_rate` only for `binary_chain`, with `null` when `d = 1`.

`error_nats` is the signed difference `estimate - ground_truth`.
`runtime_seconds` measures one call to `estimate`, including input validation
and counting, but excluding data generation, ground truth and file writing.
This single timing is not an aggregated performance benchmark.

The estimates can be reproduced with the same configuration and environment;
runtime measurements vary between runs. Run:

```powershell
python -m experiments.smoke_test
```

## Classical entropy estimators (week 2)

```python
import numpy as np
from src.estimators import EntropyEstimator, PlugInEstimator, MillerMadowEstimator
from src.simulation.ground_truth import exact_entropy

samples = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
estimator: EntropyEstimator = PlugInEstimator()
print(estimator.estimate(samples))              # ln(2)
print(MillerMadowEstimator().estimate(samples)) # ln(2) + 1/8
print(exact_entropy([0.5, 0, 0, 0.5]))          # ln(2)
```

Both estimators implement `estimate(samples) -> float`. Samples must be a NumPy
`ndarray`; lists and tuples raise `TypeError` instead of undergoing implicit
dtype conversion. Input shape is `(n, d)`
(rows are observations, columns are components), or `(n,)` for one component.
Labels may be integers, booleans or finite integer-valued floats, including
negative and non-contiguous codes. Encode categorical strings before calling;
missing values, empty inputs and continuous measurements are rejected.
`numpy.ma.MaskedArray` inputs raise `ValueError`, even with an all-False mask.
When constructing arrays, use a dtype that preserves category identity: for
example, `np.array([2**53, 2**53 + 1], dtype=np.int64)`. Converting these labels
to `float64` can merge them; estimators cannot recover precision already lost
before the call. Inputs are not modified.

All outputs use natural logarithms and **nats**. Each distinct row is one joint
state: aggregation is joint entropy, never the sum or mean of marginal entropies.
Components follow the supplied column order. For the classical joint-entropy
estimators, permuting columns does not change the estimate. These estimators
do not permute components or perform a random split.

Plug-in computes `-sum(p_hat * ln(p_hat))` over observed states. Miller–Madow adds
`(K_observed - 1) / (2*n)`, where `n` counts rows and `K_observed` (K+) counts
distinct joint states with positive observed frequency. The correction
is not clipped and may exceed the true entropy or the log of the alphabet size.
For balanced samples over K symbols, Plug-in is exactly `ln(K)`; Miller–Madow
adds its finite-sample correction. Both return zero for constant samples.
Counting only observed states avoids allocating the full joint alphabet.

The generators' analytical ground truth is computed by
`independent_bernoulli_entropy(p, d)` and `binary_chain_entropy(epsilon, d)` in
`src/simulation/ground_truth.py`, using their model-specific formulas.

Separately, `exact_entropy(probabilities)` calculates population entropy from an enumerated
probability vector or joint probability tensor for small spaces. Tensor axes
follow component order and all entries are summed, with `0*ln(0) = 0`. Inputs
must sum to one (absolute tolerance `1e-12`); they are never silently normalized.
Unlike the sample estimators, this PMF function accepts numeric lists, tuples
or arrays, but rejects `MaskedArray` inputs. The PMF must be a nonempty vector
or tensor of finite real probabilities in `[0, 1]`; scalars, complex values,
strings and object-dtype arrays are rejected. "Exact" means no sampling error
relative to the supplied PMF; floating-point rounding still applies.
The sum is checked after conversion to `float64`. Construct PMFs at `float64`
precision: rounded `float32` probabilities such as `[0.1, 0.2, 0.7]` may fail
the tolerance, and casting them afterwards cannot recover precision. Exactly
representable `float32` PMFs such as `[0.25, 0.75]` remain valid.

The smoke-test command above runs both estimators on the same generated dataset
and saves their estimates, signed errors (`estimate - ground truth`), runtimes,
unit, aggregation and zero-based component order in `smoke_test_result.json`.
Tests cover constants, balanced uniform samples, nonuniform joint distributions,
unobserved states, enumeration of small binary chains, permutations and invalid
inputs. Run them with `python -m pytest -v`.

## Reproducibility

The project currently supports seeded execution for:

- Python
- NumPy
- PyTorch
- CUDA, when available

The synthetic generators use their own NumPy `default_rng(seed)`. For a fixed
configuration in the same environment, the classical smoke-test samples and
estimates are reproducible; timings are not.

Seeded execution sets random seeds. Strict deterministic execution would also
require control of potentially nondeterministic operations. The current seed
utility does not enable PyTorch deterministic algorithms or configure cuDNN
benchmark/determinism settings; strict determinism is not guaranteed.

Additional deterministic settings will be introduced where required during the neural-estimator implementation.

## Current status

The current implementation includes:

- independent Bernoulli synthetic generator;
- dependent binary-chain synthetic generator;
- analytical ground-truth entropy calculations;
- exact entropy from finite probability tables;
- common estimator protocol, Plug-in and Miller–Madow joint entropy;
- YAML-based experiment configuration;
- generator factory;
- configuration validation;
- seed and reproducibility utilities;
- command-line configurable smoke-test pipeline;
- automated unit and smoke-test integration tests.

## Roadmap status

- Week 1 — Infrastructure and generators: complete.
- Week 2 — Classical estimators: complete.
- Week 3 — NJEE: next.

NJEE, C-NJEE, MI, CMI, Transfer Entropy and the full experimental evaluation
pipeline are not implemented yet.
