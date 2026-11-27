# Neural Information Estimation

Research software for a Master's thesis on **Neural Joint Entropy Estimation (NJEE)** and its conditional extension. The project implements discrete entropy, mutual information (MI), conditional mutual information (CMI) and transfer entropy (TE), with classical baselines, synthetic validation protocols, ablations and reproducible reporting.

The four thesis studies have been run. The results below link the figures and numerical tables to the experiment code and frozen protocols.

## Thesis results

| Study | Results | Experiment entry point |
| --- | --- | --- |
| 1. Joint entropy | [Entropy estimation](thesis-results/entropy/README.md) | [`experiments.final_validation`](experiments/final_validation.py) |
| 2. Mutual and conditional mutual information | [MI and CMI estimation](thesis-results/information/README.md) | [`experiments.information_validation`](experiments/information_validation.py) |
| 3. Transfer entropy | [Transfer entropy estimation](thesis-results/transfer-entropy/README.md) | [`experiments.transfer_validation`](experiments/transfer_validation.py) |
| 4. Ablation | [Ablation study](thesis-results/ablation/README.md) | [`experiments.ablation_validation`](experiments/ablation_validation.py) |

The [complete result archives](https://github.com/vdb1099-gif/njee-thesis/releases/tag/thesis-results-v1) contain the full figures and numerical tables for each study.

## Review path

1. Open a study above and choose a figure. Its CSV gives the exact plotted values, and the study tables provide the scenario-level summaries.
2. Follow the study's protocol and entry point to see how the generators, references, estimators and comparisons were configured.
3. Follow the [estimators](src/estimators/) and [reporting code](src/reporting/) to trace how estimates become the reported quantities.

## How estimation works

NJEE applies the entropy chain rule:

```text
H(X_1, ..., X_d) = H(X_1) + sum(j=2..d) H(X_j | X_1, ..., X_(j-1))
```

The first marginal is estimated from training data using plug-in. For the remaining terms, a classifier predicts the current variable from preceding variables. Validation selects the checkpoint and held-out test cross-entropy supplies each contribution. C-NJEE fits a classifier for every term conditioned on the given variables. MI and CMI use differences of entropy estimates, and TE applies CMI after temporal embedding.

The IID studies use separate training, validation and test sets. The temporal studies use chronological splits and purging. Classical `full` variants use all observations, while `train` variants use the training partition. All information quantities are expressed in **nats**.

## Repository map

| Area | Relevant code |
| --- | --- |
| Study entry points and frozen protocols | [experiments/](experiments/) |
| Generators and reference values | [src/simulation/](src/simulation/) |
| Classical and neural estimators | [src/estimators/](src/estimators/) |
| Neural models, encoding and data splits | [src/models/](src/models/) and [src/utils/](src/utils/) |
| Error metrics, inference, tables and figures | [src/metrics/](src/metrics/) and [src/reporting/](src/reporting/) |
| Unit and integration tests | [tests/](tests/) |

See [Reproducing the thesis experiments](REPRODUCING.md) for environment setup, dry runs, tests, execution and report verification.
