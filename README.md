# NJEE Thesis Project

Implementation and experimental validation of **Neural Joint Entropy Estimation (NJEE)** for discrete multivariate data, with classical entropy baselines and synthetic distributions with analytical ground truth.

**Status updated: 9 October 2026.** The NJEE estimator, single-run pipeline, multi-replica campaigns and train/validation optimization diagnostic are implemented. Preliminary development experiments have been completed locally; a final scientific evaluation is still pending.

## Current project status

| Area | Implemented functionality | Status |
| --- | --- | --- |
| Simulation | Independent Bernoulli and dependent binary-chain generators; analytical entropy; finite-PMF entropy | Implemented and tested |
| Classical estimators | Plug-in and Miller–Madow joint entropy | Implemented and tested |
| Data protocol | Reproducible IID train/validation/test splits; categorical identities, encoding and overlap checks | Implemented and tested |
| Neural models | Conditional MLP classifiers with one or two ReLU hidden layers | Implemented and tested |
| NJEE | Training plug-in marginal plus held-out conditional cross-entropies; per-term diagnostics | Implemented and tested |
| Experiment runners | Single replica and sequential multi-replica campaigns with provenance and failure records | Implemented and tested |
| Metrics | Bias, MAE, RMSE, sample SD, timings, replica bootstrap and paired comparisons | Implemented and tested |
| Development validation | Five pilot scenarios; paired optimization diagnostic on four scenarios | Completed locally on 8 October 2026 |
| Final evaluation | Larger, frozen-protocol study supporting thesis conclusions | Pending |
| Extensions | C-NJEE, MI, CMI and Transfer Entropy | Not implemented |

The checked-in neural configurations are deliberately tiny **software correctness examples**. Their sample sizes, epoch budgets and replica counts are not a recommended scientific evaluation protocol.

## Installation

Use Python 3.13 (the local environment uses **3.13.0**) and Git. The pinned dependencies require Python 3.12 or newer; Python 3.12 has not been verified here. Python 3.11 is insufficient for the current pins.

Run from Windows PowerShell:

```powershell
git clone https://github.com/vdb1099-gif/njee-thesis.git
cd njee-thesis
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all following commands from the repository root with the virtual environment active. On other platforms, create and activate an equivalent Python virtual environment.

## Quick start

```powershell
# Classical estimators on synthetic data
python -m experiments.smoke_test --config configs/baseline.yaml --output-dir results

# One NJEE replica and four classical comparison variants
python -m experiments.neural_baseline --config configs/neural_baseline.yaml --run-id example-run

# Two-replica development campaign
python -m experiments.neural_campaign --config configs/neural_campaign.yaml

# Inspect saved campaign artifacts without rerunning
python -m experiments.neural_campaign --inspect results/neural_campaign/tiny-development

# Automated test suite; use a writable, ignored temporary directory
python -m pytest -q --basetemp=results/pytest_tmp
```

Neural runs and campaigns refuse existing output directories, including incomplete or failed runs. Use a fresh `--run-id` for another baseline run, or a new `campaign_id`/`--output-dir` for another campaign. There is no automatic resume. The classical smoke test overwrites its result file when reusing an output directory.

## NJEE estimation protocol

For the configured component order, the estimator uses the chain rule:

```text
H(X1, ..., Xd) = H(X1) + sum(i=2..d) H(Xi | X1, ..., X(i-1))
```

The implementation estimates the first marginal with the plug-in estimator on **training data**. Each subsequent term is a separate conditional classifier trained on training data, with dense one-hot conditioning features and categorical target indices. Adam minimizes cross-entropy; validation loss controls early stopping and checkpoint selection. The selected checkpoint is restored and evaluated on **held-out test data**. Joint entropy is the sum of the training marginal and the mean test cross-entropy of each conditional term, in **nats**.

The categorical encoder is fitted on training data. It supports caller-declared alphabets or training-observed alphabets; unknown categories are rejected. The binary experiment runners use declared `{0, 1}` alphabets. Dataset/split identities, variable order, row indices and disjoint partitions are checked to prevent leakage. Neural estimates can depend on component order because conditional models are fitted separately.

The API is `NeuralJointEntropyEstimator(config, alphabets=...).fit(train, validation).estimate(test)`, using `CategoricalBatch` partitions and a validated `NeuralConfig`. It returns a `NeuralEntropyReport` with the total and per-term diagnostics. This differs from the classical `estimate(ndarray) -> float` interface. Refit after a successful fit is rejected; construct a fresh estimator for another run.

Held-out cross-entropy remains a model-based estimate of conditional entropy. Finite samples, optimization and model approximation affect its error; implementing the chain rule does not establish accuracy or convergence. The API also cannot enforce one-time scientific use of test results.

## Configuration

| File | Purpose | Checked-in example |
| --- | --- | --- |
| `configs/baseline.yaml` | Classical smoke test | Binary-chain synthetic data |
| `configs/neural_baseline.yaml` | One NJEE run | Binary chain, n=23, d=3; hidden [4]; 2 epochs; strict CPU |
| `configs/neural_campaign.yaml` | Replicas of one baseline scenario | 2 development replicas; 200 bootstrap draws |

Synthetic data settings include `seed`, `data.generator`, `data.n`, `data.d` and `data.params`. Supported generators are `independent_bernoulli` (`p`) and `binary_chain` (`epsilon`). Sample count and dimension are positive integers; probabilities lie in `[0, 1]`; seeds lie in `[0, 2**32 - 1]`.

The neural baseline additionally requires `neural`, `output_dir` and `run_id`. Every neural field is explicit and validated: split seed/fractions, training seed, component order, alphabet policy, marginal method, hidden sizes, activation, optimizer, learning rate, batch size, epoch cap, patience, minimum improvement, device, reproducibility mode and worker count. Current choices include one or two ReLU hidden layers, Adam, `train_plugin` marginal, `seeded` or `strict` execution, CPU or an explicit CUDA device, and zero data-loader workers.

Campaign configuration selects a baseline configuration and supplies `campaign_id`, `purpose` (`development` or `final`), replica count, output directory and aggregation settings. Seeds come from either a `master_seed` or explicit replica seed records. Data, split and training seeds are derived separately; dataset-seed collisions are rejected. A campaign runs one scenario sequentially. Setting `purpose: final` labels intent and does not by itself establish scientific adequacy.

**Path rules:** smoke-test relative CLI paths are resolved from the repository root. Neural CLI/output paths are resolved from the current working directory. A campaign's `baseline_config` reference is resolved relative to its campaign YAML file.

## Results and provenance

Generated outputs under `results/` are ignored by Git and are **not included in a fresh clone**.

### Classical smoke test

`results/smoke_test_result.json` records the generation settings, dataset shape, analytical truth, component order, joint aggregation and units. Each classical estimator records its estimate, signed error (`estimate - truth`) and runtime. Binary-chain output also includes empirical flip rate (`null` for d=1).

### Single NJEE run

A successful baseline creates `output_dir/run_id/` with:

- `manifest.json`: resolved settings, seeds, sample counts, dataset hash, Git/environment provenance and completion status;
- `dataset.npz`: generated samples;
- `split.npz` and `split.json`: reproducible partition indices and metadata;
- `results.json`: `njee`, `plug_in_full`, `miller_madow_full`, `plug_in_train` and `miller_madow_train` estimates, errors and runtimes;
- `njee_diagnostics.json`: chain contributions, selected epochs, stopping reasons, loss histories, effective term seeds and backend metadata.

Only `manifest.status = "complete"` certifies successful completion. Execution failures retain a failed manifest; an interrupted process can leave a running manifest. Model checkpoints are used in memory; this pipeline does not export trained model weights.

Classical `full` variants use all generated observations, while classical `train` variants use only training rows. NJEE uses training, validation and test for different roles. These sample protocols must accompany any method comparison.

### Multi-replica campaigns

Campaign directories contain `campaign_manifest.json`, `raw_results.json`, `aggregation.json`, generated replica configurations and individual runs under `replicas/`. Failed replicas are recorded and remaining replicas continue; interruptions preserve pending entries. Overall status distinguishes complete, complete with failures and interrupted execution.

Aggregation reports successful/failed/not-attempted counts, mean estimate, bias, MAE, RMSE, sample standard deviation and runtime summaries. Deterministic 95% percentile bootstrap intervals resample **replicas**, not dataset rows. Paired comparisons use the shared successful replica IDs. Fewer than two successes yield no interval; small development samples and success-only intervals are not final evidence.

## Development experiments already executed locally

The following summaries come from the local artifacts dated **8 October 2026**, not from the tiny checked-in examples. The reports, experimental configurations, CSV exports and raw artifacts live under ignored `results/` directories and are unavailable in a fresh clone.

### Scientific validation pilot (Task 012)

Five development campaigns completed with **3 replicas each**, **15 successful replicas**, **75 estimator records** and **69 conditional fits**, with no failed runs or nonfinite estimates reported:

| Distribution | d | n |
| --- | --- | --- |
| Independent Bernoulli, p=0.5 | 4 | 200 |
| Independent Bernoulli, p=0.5 | 4 | 800 |
| Binary chain, epsilon=0.15 | 4 | 200 |
| Binary chain, epsilon=0.15 | 4 | 800 |
| Binary chain, epsilon=0.15 | 12 | 800 |

The pilot used IID 60/20/20 splits, hidden [16], ReLU, Adam 0.01, batch 64, maximum 20 epochs, patience 5, min_delta 0.0001 and strict CPU execution. Results varied by scenario: NJEE had lower observed error than the classical variants in the d=12 chain pilot, while several d=4 comparisons favored classical methods. Three replicas support preliminary diagnostics, not general superiority claims.

Local report: `results/task012_pilot_20261008/TASK012_report.md`.

### Train/validation optimization diagnostic (Task 013A)

The diagnostic completed **96 candidate evaluations on 32 fresh datasets**, with **672 conditional fits** and no failed or incomplete evaluations reported. Four scenarios, eight replicas per scenario, compared:

| Candidate | Maximum epochs | Patience |
| --- | --- | --- |
| A | 20 | 5 |
| B | 60 | 5 |
| C | 60 | 10 |

All candidates shared datasets, splits, encoding and per-term seeds within each replica. Selection used training/validation data only: no test cross-entropy or joint entropy was evaluated. The frozen decision rule recommended **A (20 epochs, patience 5)**; larger budgets did not meet its required improvement across scenarios. The saved report labels this a development recommendation pending approval, and no final evaluation was launched. The checked-in correctness configuration still uses its separate 2-epoch budget.

Local report: `results/task013a_optimization_20261008/TASK013A_report.md`.

The diagnostic runner is available as:

```powershell
python -m experiments.neural_optimization --output-dir results/optimization-new-run
```

It executes a fixed four-scenario study with eight replicas and three candidates, a 1800-second budget and a fresh output directory. It saves manifests, paired evaluations, train/validation artifacts, split metadata and selection output. This command launches the full development diagnostic; it is not a quick smoke test.

## Classical estimators and ground truth

`PlugInEstimator` and `MillerMadowEstimator` accept NumPy categorical samples of shape `(n, d)` or `(n,)`. Each distinct row is a joint state. Labels may be integers, booleans or finite integer-valued floats, including negative and non-contiguous codes. Strings must be encoded first. Empty inputs, continuous measurements, missing values and masked arrays are rejected; input arrays are not modified. Preserve category identity when choosing a dtype: precision lost before estimation cannot be recovered.

Plug-in computes `-sum(p_hat * ln(p_hat))` over observed states. Miller–Madow adds `(K_observed - 1) / (2*n)` without clipping. Both are invariant to column permutation and return zero for constant samples. They count observed states without allocating the full joint alphabet.

`independent_bernoulli_entropy(p, d)` and `binary_chain_entropy(epsilon, d)` provide analytical ground truth. `exact_entropy(probabilities)` computes entropy from an enumerated finite PMF or joint probability tensor with `0*ln(0)=0`. It requires finite probabilities in `[0, 1]` summing to one within absolute tolerance `1e-12`, with no silent normalization. Use float64 precision for PMF construction.

## Reproducibility and verification

Synthetic generators use local NumPy RNGs. Neural execution separates generation, splitting, initialization and minibatch shuffle seeds, and records effective seeds with software/device metadata.

`seeded` mode sets seeds while retaining existing backend flags. `strict` mode also enables deterministic PyTorch algorithms, disables cuDNN benchmarking and configures deterministic cuDNN execution. Strict CUDA requires compatible cuBLAS workspace settings before initialization; unsupported deterministic operations raise errors. Requested unavailable devices are rejected without CPU fallback. Backend settings are restored after the scoped execution. No bitwise identity is guaranteed across different hardware or software environments, and runtimes vary.

Tests cover synthetic ground truth, classical estimates, invalid inputs, split integrity, categorical leakage checks, configuration, seed derivation, MLP behavior, checkpoint selection, held-out NJEE evaluation, runner artifacts/failures, bootstrap metrics and the optimization diagnostic's test-access boundary. Latest local verification on **9 October 2026**: **644 tests passed** using the command above (29.41 seconds). This verifies the exercised software behavior, not scientific estimator accuracy.

## Repository structure

```text
configs/                 YAML configurations for smoke test, baseline and campaign
experiments/             Executable smoke, baseline, campaign and optimization runners
src/
  estimators/            Classical estimators, NJEE and conditional-term training
  metrics/               Replica aggregation, bootstrap and paired comparisons
  models/                Conditional MLP classifier
  simulation/            Generators, factory and entropy ground truth
  utils/                 Configuration, categorical contracts, splits and reproducibility
tests/                   Unit and integration tests
results/                 Local generated artifacts (ignored by Git)
requirements.txt         Pinned dependencies
pytest.ini               Test discovery restricted to tests/
```

## Remaining work

The next scientific step is to settle the development training policy and conduct a larger final evaluation with frozen settings, independent replicas, sufficient uncertainty assessment and explicit sample-protocol comparisons. Broader scenario coverage, sensitivity to component order/model choices and thesis-ready result exports remain to be consolidated. C-NJEE, mutual information, conditional mutual information and Transfer Entropy remain future extensions.
