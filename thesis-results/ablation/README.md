# Ablation

Come cambiano errore e costo al variare di architettura, condivisione dei parametri, dimensione del training set, early stopping, discretizzazione, embedding e ordine delle variabili? Quando cambia la quantità stimata, l’RMSE usa il riferimento del singolo scenario.

Il manifest finale registra **39 scenari** e **1.170 record completati**.
Le quantità di informazione e i relativi errori sono espressi in nat.

[Esecuzione dello studio](../../experiments/ablation_validation.py) · [Protocollo](../../experiments/ablation_protocol.py) · [Generazione delle figure](../../src/reporting/figures.py) · [Esportazione delle tabelle](../../src/reporting/tables.py)

Il protocollo usa 30 repliche per configurazione. Per la catena binaria, le configurazioni di architettura mostrate hanno RMSE fra 0,0618 e 0,0658 nat, con 0,0617 nat per la configurazione di riferimento. Per leggere questi scarti insieme all’incertezza, si possono consultare i [confronti appaiati](tables/paired_comparisons.csv).

[Scarica figure e tabelle complete (ZIP)](https://github.com/vdb1099-gif/njee-thesis/releases/download/thesis-results-v1/ablation-results.zip).

## Tabelle

- [Riepiloghi per scenario e stimatore (RMSE, bias, deviazione standard e numerosità)](tables/summaries.csv)
- [Tempi e costi per replica](tables/costs.csv)
- [Confronti appaiati](tables/paired_comparisons.csv)
- [Intervalli di incertezza](tables/uncertainty.csv)
- [Valori di riferimento](tables/references.csv)
- [Sensibilità all’ordine delle variabili](tables/ordering.csv)

## Figure

La galleria mostra una selezione per tipo di grafico. L’[indice completo](figure-catalog.md) elenca tutti i 36 grafici, con i parametri fissati e i CSV di origine.

### RMSE per configurazione di architettura

**gruppo = binary_chain**

![RMSE per configurazione di architettura — gruppo = binary_chain](figures/architecture-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/architecture-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di architettura — gruppo = independent](figures/architecture-rmse--group-independent.png)

[CSV dei punti del grafico](tables/architecture-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

### RMSE per configurazione di condivisione dei parametri

**gruppo = binary_chain**

![RMSE per configurazione di condivisione dei parametri — gruppo = binary_chain](figures/sharing-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/sharing-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di condivisione dei parametri — gruppo = independent](figures/sharing-rmse--group-independent.png)

[CSV dei punti del grafico](tables/sharing-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

### RMSE per configurazione di dimensione del training set

**gruppo = binary_chain**

![RMSE per configurazione di dimensione del training set — gruppo = binary_chain](figures/training-size-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/training-size-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di dimensione del training set — gruppo = independent](figures/training-size-rmse--group-independent.png)

[CSV dei punti del grafico](tables/training-size-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

### RMSE per configurazione di early stopping

**gruppo = binary_chain**

![RMSE per configurazione di early stopping — gruppo = binary_chain](figures/early-stopping-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/early-stopping-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di early stopping — gruppo = independent](figures/early-stopping-rmse--group-independent.png)

[CSV dei punti del grafico](tables/early-stopping-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

### RMSE per configurazione di discretizzazione

**gruppo = binary_chain**

![RMSE per configurazione di discretizzazione — gruppo = binary_chain](figures/discretization-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/discretization-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = discretization**

![RMSE per configurazione di discretizzazione — gruppo = discretization](figures/discretization-rmse--group-discretization.png)

[CSV dei punti del grafico](tables/discretization-rmse--group-discretization-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di discretizzazione — gruppo = independent](figures/discretization-rmse--group-independent.png)

[CSV dei punti del grafico](tables/discretization-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

### RMSE per configurazione di embedding

**gruppo = binary_chain**

![RMSE per configurazione di embedding — gruppo = binary_chain](figures/embedding-rmse--group-binary-chain.png)

[CSV dei punti del grafico](tables/embedding-rmse--group-binary-chain-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = independent**

![RMSE per configurazione di embedding — gruppo = independent](figures/embedding-rmse--group-independent.png)

[CSV dei punti del grafico](tables/embedding-rmse--group-independent-points.csv) · [Tabella di origine](tables/summaries.csv).

**gruppo = temporal**

![RMSE per configurazione di embedding — gruppo = temporal](figures/embedding-rmse--group-temporal.png)

[CSV dei punti del grafico](tables/embedding-rmse--group-temporal-points.csv) · [Tabella di origine](tables/summaries.csv).

### Variabilità tra ordini entro replica

**gruppo = binary_chain**

![Variabilità tra ordini entro replica — gruppo = binary_chain](figures/ordering-variability--group-binary-chain.png)

[CSV dei punti del grafico](tables/ordering-variability--group-binary-chain-points.csv) · [Tabella di origine](tables/ordering.csv).

**gruppo = independent**

![Variabilità tra ordini entro replica — gruppo = independent](figures/ordering-variability--group-independent.png)

[CSV dei punti del grafico](tables/ordering-variability--group-independent-points.csv) · [Tabella di origine](tables/ordering.csv).

### Differenze appaiate di errore assoluto

**gruppo = binary_chain, stessa quantità stimata = True**

![Differenze appaiate di errore assoluto — gruppo = binary_chain, stessa quantità stimata = True](figures/paired-error-differences--group-binary-chain--same-estimand-true.png)

[CSV dei punti del grafico](tables/paired-error-differences--group-binary-chain--same-estimand-true-points.csv) · [Tabella di origine](tables/paired_comparisons.csv).

**gruppo = discretization, stessa quantità stimata = False**

![Differenze appaiate di errore assoluto — gruppo = discretization, stessa quantità stimata = False](figures/paired-error-differences--group-discretization--same-estimand-false.png)

[CSV dei punti del grafico](tables/paired-error-differences--group-discretization--same-estimand-false-points.csv) · [Tabella di origine](tables/paired_comparisons.csv).

### Numero di parametri

**gruppo = binary_chain**

![Numero di parametri — gruppo = binary_chain](figures/parameter-count--group-binary-chain.png)

[CSV dei punti del grafico](tables/parameter-count--group-binary-chain-points.csv) · [Tabella di origine](tables/costs.csv).

**gruppo = discretization**

![Numero di parametri — gruppo = discretization](figures/parameter-count--group-discretization.png)

[CSV dei punti del grafico](tables/parameter-count--group-discretization-points.csv) · [Tabella di origine](tables/costs.csv).

### Aggiornamenti dell’ottimizzatore

**gruppo = binary_chain**

![Aggiornamenti dell’ottimizzatore — gruppo = binary_chain](figures/optimizer-updates--group-binary-chain.png)

[CSV dei punti del grafico](tables/optimizer-updates--group-binary-chain-points.csv) · [Tabella di origine](tables/costs.csv).

**gruppo = discretization**

![Aggiornamenti dell’ottimizzatore — gruppo = discretization](figures/optimizer-updates--group-discretization.png)

[CSV dei punti del grafico](tables/optimizer-updates--group-discretization-points.csv) · [Tabella di origine](tables/costs.csv).

### Valutazioni termine-batch

**gruppo = binary_chain**

![Valutazioni termine-batch — gruppo = binary_chain](figures/term-batch-evaluations--group-binary-chain.png)

[CSV dei punti del grafico](tables/term-batch-evaluations--group-binary-chain-points.csv) · [Tabella di origine](tables/costs.csv).

**gruppo = discretization**

![Valutazioni termine-batch — gruppo = discretization](figures/term-batch-evaluations--group-discretization.png)

[CSV dei punti del grafico](tables/term-batch-evaluations--group-discretization-points.csv) · [Tabella di origine](tables/costs.csv).

### Tempo di esecuzione

**gruppo = binary_chain**

![Tempo di esecuzione — gruppo = binary_chain](figures/runtime-seconds--group-binary-chain.png)

[CSV dei punti del grafico](tables/runtime-seconds--group-binary-chain-points.csv) · [Tabella di origine](tables/costs.csv).

**gruppo = discretization**

![Tempo di esecuzione — gruppo = discretization](figures/runtime-seconds--group-discretization.png)

[CSV dei punti del grafico](tables/runtime-seconds--group-discretization-points.csv) · [Tabella di origine](tables/costs.csv).
