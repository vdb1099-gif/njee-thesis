# Neural Information Estimation

Research software for a Master's thesis on **Neural Joint Entropy Estimation (NJEE)** and its conditional extension. The project implements discrete entropy, mutual information (MI), conditional mutual information (CMI) and transfer entropy (TE), with classical baselines, synthetic validation protocols, ablations and reproducible reporting.

The estimators and experiment infrastructure are implemented. Development pilots and a training/validation optimization diagnostic have been completed locally. Final scientific campaigns and validation of the intended Linux deployment remain separate milestones. Software correctness tests do not establish statistical accuracy or final scientific conclusions.

NJEE uses the entropy chain rule:

```text
H(X_1, ..., X_d) = H(X_1) + sum(j=2..d) H(X_j | X_1, ..., X_(j-1))
```

The implementation combines a training-set plug-in first marginal with neural conditional cross-entropies evaluated on held-out test data. C-NJEE estimates conditional joint entropy by fitting a classifier for every chain-rule term, including the first.

## Quick Start

Create and activate the environment using [Environment setup](#environment-setup). Then check the software and inspect the four study designs:

```bash
python -m pytest -q --basetemp=results/pytest_tmp
python -m experiments.final_validation --dry-run
python -m experiments.information_validation --dry-run
python -m experiments.transfer_validation --dry-run
python -m experiments.ablation_validation --dry-run
```

The regression suite includes small training fixtures. Dry-runs resolve protocol metadata without training models or producing scientific evidence. Preparation and full campaign execution are described under [Frozen studies](#frozen-studies).

## Implemented capabilities

| Area | Functionality |
| --- | --- |
| Estimators | Plug-in, observed-support Miller–Madow, NJEE, C-NJEE, neural MI, CMI and TE |
| Data contracts | Categorical identities, declared alphabets, one-hot features, IID splits, temporal embedding and purged chronological splits |
| Simulation | Bernoulli vectors, binary chains, binary symmetric channels, conditional chains/forks and coupled binary paths |
| Training | Conditional ReLU MLPs, Adam, validation-selected checkpoints and explicit random streams |
| Validation | Frozen entropy, MI/CMI, TE and ablation studies with failures and provenance |
| Reporting | 50 figure recipes, 13 table export types, CSV/LaTeX tables and SVG/PNG figures |
| Deployment | Linux dependency profiles, preflight and artifact inventory/verification |

All information quantities use natural logarithms and **nats**. Continuous inputs require explicit discretization before categorical estimation.

## Repository structure

```text
configs/                  Examples and checked-in frozen-policy mirrors
experiments/              CLI entry points, protocol builders and controllers
src/estimators/           Classical and neural information estimators
src/models/               Conditional MLPs and ablation models
src/simulation/           Generators and reference quantities
src/utils/                Configuration, encoding, splits, temporal data and RNGs
src/metrics/              Accuracy, paired comparisons and inference
src/reporting/            Validated report generation, tables and figures
tests/                    Unit and integration tests
scripts/linux-preflight.sh
requirements.txt          Local development environment pins
requirements-linux.txt    Candidate Linux runtime dependency closure
requirements-linux-test.txt
```

## Environment setup

The local reference environment uses CPython 3.13. The intended Linux profile targets **CPython 3.13.x on x86-64**, using CPU execution. Validate pinned wheels, system compatibility and thread allocation on the destination machine before scientific execution.

Windows development setup:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

After reviewing package availability and the site's installation policy, the candidate Linux setup is:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux-test.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
python -m pip check
```

Candidate runtime pins include NumPy 2.5.3, SciPy 1.18.1, PyYAML 6.0.3 and Matplotlib 3.11.2. Consult the requirements files for the complete closure. `requirements-linux.txt` contains runtime dependencies; `requirements-linux-test.txt` adds pytest and its dependencies. PyTorch is installed separately. Target wheel availability, ABI compatibility and the complete transitive resolution require validation on a compatible Linux system. For offline installation, acquire compatible wheels on an approved machine, inspect their Python/platform tags and dependencies, record SHA-256 checksums and install from that verified wheelhouse using `--no-index --find-links`. Resolve unavailable or incompatible pins before execution.

The installed NumPy and SciPy package metadata requires Python 3.12 or later, and SciPy permits NumPy versions from 2.0 up to, but excluding, 2.8. The declared NumPy pin meets that constraint. The deployment preflight specifically requires CPython 3.13. These metadata checks do not validate Linux wheel availability or the destination environment.

Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to the approved allocation before starting Python. Use a confirmed UTF-8 environment and a writable Matplotlib cache.

For example, after assigning a positive integer to `REVIEWED_THREADS` from the site's allocation:

```bash
export OMP_NUM_THREADS="$REVIEWED_THREADS" MKL_NUM_THREADS="$REVIEWED_THREADS"
export OPENBLAS_NUM_THREADS="$REVIEWED_THREADS" NUMEXPR_NUM_THREADS="$REVIEWED_THREADS"
export MPLBACKEND=Agg PYTHONUTF8=1 PYTHONHASHSEED=0
```

Deployment preflight requires an explicit scratch parent and performs format checks there:

```bash
python -B -m experiments.deployment preflight --target-linux --include-tests --expected-commit REVIEWED_FULL_COMMIT --scratch-parent /path/to/approved/scratch --manifest /path/to/new-preflight.json
```

Use a clean checkout at the full reviewed commit, an existing writable scratch directory and a new manifest path whose parent exists. The target profile checks the exact CPU PyTorch build, dependency pins and effective thread/UTF-8 settings. The Linux wrapper accepts the same operational inputs and sets the three non-thread environment variables above:

```bash
bash scripts/linux-preflight.sh REVIEWED_FULL_COMMIT /path/to/approved/scratch /path/to/new-preflight.json
```

Preflight records environment and protocol metadata and leaves `launch_authorized` false. Successful preflight does not establish workload feasibility or scientific execution authorization.

## Development examples and tests

These commands perform computation and write local outputs. The checked-in neural baseline and campaign are small software examples with their own settings.

```bash
python -m experiments.smoke_test --config configs/baseline.yaml --output-dir results
python -m experiments.neural_baseline --config configs/neural_baseline.yaml --run-id example-run
python -m experiments.neural_campaign --config configs/neural_campaign.yaml
python -m experiments.neural_campaign --inspect results/neural_campaign/tiny-development
python -m pytest -q --basetemp=results/pytest_tmp
```

Neural output directories must be fresh, including after failure. Choose a new baseline run ID or campaign destination for another invocation. The classical smoke test overwrites its result file when its destination is reused. The optimization diagnostic is a separate development study.

## Estimation and reproducibility

NJEE adds a training plug-in first marginal to held-out conditional cross-entropies in the declared chain-rule order. C-NJEE fits a conditional classifier for every term. MI uses `H(X) - H(X|Y)` and CMI uses `H(X|Z) - H(X|Y,Z)`. TE delegates to CMI after explicit temporal embedding.

Encoders fit on training rows. Features exclude the current target. Training batches optimize cross-entropy on raw logits, validation selects checkpoints, and test observations evaluate selected models. Components preserve common dataset and split identities. Raw signed MI, CMI and TE estimates are retained.

Classical `full` variants use all observations; `train` variants use training rows only. Pairing by dataset does not equalize sample access between classical variants and neural evaluation.

The principal frozen policy uses one 16-unit ReLU layer, Adam at learning rate 0.01, batch size 64, at most 20 epochs, patience 5 and minimum improvement 0.0001. It uses declared alphabets, strict CPU execution and zero DataLoader workers. IID splits are 60/20/20. Ablations vary settings explicitly; temporal studies use aligned chronological splits, configured gaps and purging.

Generation, splitting, initialization, shuffle and analysis streams have recorded identities and deterministic derivations. Strict mode can reject unsupported deterministic operations. Reproducibility depends on resolved configurations, seeds, code, dependencies, hardware and threads. Bitwise identity across operating systems, hardware and library versions is not guaranteed.

## Frozen studies

| Study | Design | Independent sources | Focus |
| --- | --- | --- | --- |
| Entropy | 27 cells × 30 replicas | 810 datasets | NJEE versus plug-in and Miller–Madow, FULL and TRAIN |
| MI/CMI | 85 cells × 50 replicas | 4,250 datasets | Signed errors, component relationships and null permutations |
| TE | 51 cells × 50 replicas | 2,550 paths | Direction, validation-selected lag, history and path surrogates |
| Ablations | 39 cells × 30 records | 120 shared datasets/paths | Ordering, architecture, sharing, training size, stopping, discretization and embedding |

TE plans 30,600 TE fits, 61,200 conditional MLP fits and 20,379,600 surrogate statistics. Ablations produce 1,170 records using shared sources for paired comparisons. Terms, candidates, permutations and surrogate draws do not increase independent replication.

The four primary entropy cells define 16 paired neural-versus-classical absolute-error contrasts. Each requires 30 complete pairs and uses a two-sided Bonferroni-adjusted 99.6875% t interval. Other performance summaries use separately labelled exploratory pointwise uncertainty. Failed replicas, incomplete pairs and unavailable decisions retain explicit counts. No automatic replacement is performed.

MI/CMI null permutations use frozen models and exact conditioning strata where required. TE surrogate calibration depends on the configured structural-null path laws. Block permutations and circular shifts remain exploratory. Observational TE does not establish structural causality, and the physical delay need not uniquely maximize finite-history TE.

References distinguish analytical truths from numerical stationary TE calculations. Truncation bounds exclude floating-point roundoff. A stationarity residual is a diagnostic, and reference approximation is separate from replica uncertainty.

### Configuration architecture

Python modules define and resolve authoritative frozen policies:

| Study | Protocol file | CLI module | Checked-in mirror |
| --- | --- | --- | --- |
| Entropy | `experiments/final_protocol.py` | `experiments.final_validation` | Preparation generates baseline/campaign YAMLs |
| MI/CMI | `experiments/information_protocol.py` | `experiments.information_validation` | `configs/information_validation.yaml` |
| TE | `experiments/transfer_protocol.py` | `experiments.transfer_validation` | `configs/transfer_validation.yaml` |
| Ablations | `experiments/ablation_protocol.py` | `experiments.ablation_validation` | `configs/ablation_validation.yaml` |

The later CLIs check their YAML mirrors against Python definitions. Builders expand scenarios, audit effective seeds, compute relevant references and validate bundles. Experiment 1 workers consume generated campaign YAMLs referencing generated baseline YAMLs. Editing a mirror alone does not create an approved new protocol.

### Preview, preparation and execution

Metadata previews do not train models or generate scientific datasets:

```bash
python -m experiments.final_validation --dry-run
python -m experiments.information_validation --dry-run
python -m experiments.transfer_validation --dry-run
python -m experiments.ablation_validation --dry-run
```

Preparation creates a new bundle containing resolved configurations and a protocol manifest:

```bash
python -m experiments.final_validation --prepare results/entropy-preparation
python -m experiments.information_validation --prepare results/information-preparation
python -m experiments.transfer_validation --prepare results/transfer-preparation
python -m experiments.ablation_validation --prepare results/ablation-preparation
```

Review bundle hashes, environment, storage, threads, workload and scientific authorization before launch. Execution requires a clean, identifiable Git checkout. After those gates, the execution commands are:

```bash
python -m experiments.final_validation --execute results/entropy-preparation --authorize-execution
python -m experiments.information_validation --execute results/information-preparation --authorize-execution
python -m experiments.transfer_validation --execute results/transfer-preparation --authorize-execution
python -m experiments.ablation_validation --execute results/ablation-preparation --authorize-execution
```

These launch full scientific workloads. Windows additionally requires the explicit `--allow-local-windows` exception. Consult each runner's `--help` for its path and control options.

Baseline and campaign output paths are relative to the current working directory; a campaign resolves its `baseline_config` relative to its YAML file. The classical smoke test resolves relative paths from the checkout root. Experiment 1 resolves its scientific root from the prepared bundle against `--workdir` (the checkout by default); select a different root with `--output-root` during preview/preparation. The other three runners support `--output-dir` and `--log-dir` overrides. Preparation directories and scientific output directories must remain disjoint.

Experiment 1 has an 18,000-second global controller budget and a shutdown reserve. Other controllers expose cancellation-file and interruption handling at their work boundaries. Do not assume that every fit stops immediately on cancellation or every runner has a hard deadline. Preserve partial artifacts. Existing scientific roots are refused; automatic retry, replacement and resume are not provided.

## Scientific artifacts and reporting

Artifacts record resolved settings, raw estimates, ledgers, seeds, dataset/split or timestamp identities, term diagnostics and timings. Available numeric checkpoints and completed artifacts carry hashes. Review completion and failure states before interpreting summaries. Scientific outputs are absent from a fresh clone.

The reporting CLI supports `entropy`, `information`, `transfer` and `ablation`:

```bash
# Inspect figure/table contracts without scientific results
python -m experiments.scientific_report --plan

# Validate an existing scientific source
python -m experiments.scientific_report --validate results/experiment2_final --experiment information

# Generate a diagnostic report
python -m experiments.scientific_report --generate results/experiment3_final --experiment transfer --mode diagnostic --output results/reports/transfer-diagnostic

# Generate a reviewed final report
python -m experiments.scientific_report --generate results/experiment2_final --experiment information --mode final --approval /path/to/reviewed-approval.json --output results/reports/information-final

# Verify outputs and original inputs
python -m experiments.scientific_report --verify-report results/reports/information-final --verify-inputs
```

Final mode requires complete validated authentic inputs and approval bound to their inventory hashes. Diagnostic mode labels incomplete evidence as non-final and still rejects corruption. Report destinations must be fresh and disjoint from scientific sources. For entropy archives, the default controller ledger is `control/ledger.json` inside the source; `--ledger` selects another archived ledger within that root.

Keep the approval JSON outside the scientific source archive. It must contain `experiment`, `protocol_hash`, `source_inventory_hash`, `approved_for_final_reporting` set to `true`, and a nonempty `reviewed_by` string. The first three values must match source validation exactly. Supply this record only after reviewing those inputs. Relative report paths are resolved from the current working directory.

Reports contain `report_manifest.json`, `validated_analysis.json`, `figure_index.json`, captions, CSV/LaTeX tables and SVG/PNG figures. Facets retain source-row provenance. Manifests record approval, code, environments and input/output hashes. Raw records preserve full precision independently of display rounding. LaTeX table consumers need `longtable`.

For transferring reviewed artifacts:

```bash
python -m experiments.deployment inventory /path/to/artifact-tree /path/to/new-inventory.json
python -m experiments.deployment verify /path/to/artifact-tree /path/to/new-inventory.json
```

Keep the inventory outside its payload. Verification checks safe entries, sizes and SHA-256 hashes. Hashes detect accidental changes; they do not replace scientific approval.

## Remaining milestones

Validate the destination environment, execute separately approved campaigns, review completed and partial artifacts, generate approved final reports and interpret the evidence. Final performance claims require those results. The optional financial-data application is outside the implemented studies.
