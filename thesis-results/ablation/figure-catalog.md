# Catalogo delle figure · Ablation

36 figure. I nomi corrispondono ai file PNG e SVG del pacchetto scaricabile.

## RMSE per configurazione di architettura

- `architecture-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `architecture-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`

## RMSE per configurazione di condivisione dei parametri

- `sharing-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `sharing-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`

## RMSE per configurazione di dimensione del training set

- `training-size-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `training-size-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`

## RMSE per configurazione di early stopping

- `early-stopping-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `early-stopping-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`

## RMSE per configurazione di discretizzazione

- `discretization-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `discretization-rmse--group-discretization` · gruppo = discretization · origine `tables/summaries.csv`
- `discretization-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`

## RMSE per configurazione di embedding

- `embedding-rmse--group-binary-chain` · gruppo = binary_chain · origine `tables/summaries.csv`
- `embedding-rmse--group-independent` · gruppo = independent · origine `tables/summaries.csv`
- `embedding-rmse--group-temporal` · gruppo = temporal · origine `tables/summaries.csv`

## Variabilità tra ordini entro replica

- `ordering-variability--group-binary-chain` · gruppo = binary_chain · origine `tables/ordering.csv`
- `ordering-variability--group-independent` · gruppo = independent · origine `tables/ordering.csv`

## Differenze appaiate di errore assoluto

- `paired-error-differences--group-binary-chain--same-estimand-true` · gruppo = binary_chain, stessa quantità stimata = True · origine `tables/paired_comparisons.csv`
- `paired-error-differences--group-discretization--same-estimand-false` · gruppo = discretization, stessa quantità stimata = False · origine `tables/paired_comparisons.csv`
- `paired-error-differences--group-independent--same-estimand-true` · gruppo = independent, stessa quantità stimata = True · origine `tables/paired_comparisons.csv`
- `paired-error-differences--group-temporal--same-estimand-false` · gruppo = temporal, stessa quantità stimata = False · origine `tables/paired_comparisons.csv`

## Numero di parametri

- `parameter-count--group-binary-chain` · gruppo = binary_chain · origine `tables/costs.csv`
- `parameter-count--group-discretization` · gruppo = discretization · origine `tables/costs.csv`
- `parameter-count--group-independent` · gruppo = independent · origine `tables/costs.csv`
- `parameter-count--group-temporal` · gruppo = temporal · origine `tables/costs.csv`

## Aggiornamenti dell’ottimizzatore

- `optimizer-updates--group-binary-chain` · gruppo = binary_chain · origine `tables/costs.csv`
- `optimizer-updates--group-discretization` · gruppo = discretization · origine `tables/costs.csv`
- `optimizer-updates--group-independent` · gruppo = independent · origine `tables/costs.csv`
- `optimizer-updates--group-temporal` · gruppo = temporal · origine `tables/costs.csv`

## Valutazioni termine-batch

- `term-batch-evaluations--group-binary-chain` · gruppo = binary_chain · origine `tables/costs.csv`
- `term-batch-evaluations--group-discretization` · gruppo = discretization · origine `tables/costs.csv`
- `term-batch-evaluations--group-independent` · gruppo = independent · origine `tables/costs.csv`
- `term-batch-evaluations--group-temporal` · gruppo = temporal · origine `tables/costs.csv`

## Tempo di esecuzione

- `runtime-seconds--group-binary-chain` · gruppo = binary_chain · origine `tables/costs.csv`
- `runtime-seconds--group-discretization` · gruppo = discretization · origine `tables/costs.csv`
- `runtime-seconds--group-independent` · gruppo = independent · origine `tables/costs.csv`
- `runtime-seconds--group-temporal` · gruppo = temporal · origine `tables/costs.csv`
