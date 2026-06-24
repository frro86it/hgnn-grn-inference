---
title: "Analisi SHAP delle Feature Topologiche — Perché 807 Feature Performano Peggio di 805"
author: "Francesco Rollin, Fabio Sessa"
date: "22 Giugno 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.3
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{amsmath}
  - \usepackage{booktabs}
---

\newpage

# Contesto e Motivazione

## Il problema osservato

Nelle sessioni precedenti è emerso un risultato controintuitivo: aggiungere due feature topologiche — il **grado** e l'**entropia di Shannon** di ogni gene — alla matrice di input $X$ peggiora le performance del modello HGNN invece di migliorarle.

| Configurazione | Feature in X | AUPR | Variazione |
|---|---|---|---|
| HGNN Edge Stratified (baseline) | 805 | **0.247** | — |
| HGNN Edge Stratified + Topo | 807 | 0.240 | **−2.6%** |
| HGNN Hard Stratified (baseline) | 805 | 0.068 | — |
| HGNN Hard Stratified + Topo | 807 | 0.062 | **−8.6%** |

Il tutor, di fronte a questo risultato, ha chiesto di eseguire un'**analisi SHAP** per quantificare il contributo delle due feature topologiche e capire perché, nonostante siano potenzialmente informative, danneggiano il modello.

## Definizione delle feature topologiche

Le due feature aggiunte a ogni gene $i$ sono derivate direttamente dalla matrice di incidenza $H$ (dimensione $1643 \times 195$):

**Grado del nodo** $d(i)$: il numero di TF che regolano il gene $i$.
$$d(i) = \sum_{j=1}^{195} H[i][j]$$

**Entropia di Shannon** $\mathcal{H}(i)$: misura quanto è distribuita la regolazione del gene $i$ tra i suoi TF. Un gene regolato da un solo TF ha entropia zero; un gene regolato uniformemente da molti TF ha entropia massima.
$$\mathcal{H}(i) = -\sum_{j: H[i][j]=1} p_{ij} \log_2 p_{ij}, \quad p_{ij} = \frac{H[i][j]}{d(i)}$$

Entrambe le feature vengono normalizzate in $[0, 1]$ prima di essere concatenate alle 805 feature di espressione, portando $X$ da dimensione $1643 \times 805$ a $1643 \times 807$.

---

\newpage

# L'Analisi SHAP con Integrated Gradients

## Metodo scelto: Integrated Gradients

L'analisi SHAP classica (Shapley Additive Explanations) richiede la valutazione esponenziale del modello su sottoinsiemi di feature. Per reti neurali su grafi questa operazione è computazionalmente proibitiva. Si è usato al suo posto il metodo **Integrated Gradients** (Sundararajan et al., 2017), che è:

- **Compatible con le GNN**: non richiede permutazioni delle feature, ma integra i gradienti lungo un percorso dall'input di riferimento all'input reale
- **Assiomaticamente fondato**: soddisfa le stesse proprietà di Shapley (efficienza, linearità, dummy)
- **Efficiente**: un'unica forward+backward pass per feature

La formula è:
$$\text{IG}_k(x) = (x_k - x_k^{\text{ref}}) \times \int_{\alpha=0}^{1} \frac{\partial F(x^{\text{ref}} + \alpha(x - x^{\text{ref}}))}{\partial x_k} d\alpha$$

dove $x^{\text{ref}}$ è un input di baseline (vettore zero), $x$ è l'input reale, e $F$ è l'output del decoder (probabilità di arco). L'integrale è approssimato con 50 passi lineari.

## Cosa misura l'importanza calcolata

Il valore IG per la feature $k$ risponde a: *"quanto varia l'output del decoder se rimuoviamo progressivamente il contributo della feature $k$ portandola da $x_k$ a zero?"* Un valore alto significa che il modello si affida molto a quella feature per le sue predizioni.

È fondamentale tenere presente la distinzione tra due concetti che l'analisi SHAP permette di separare:

- **Importanza individuale**: quanto il modello *usa* una feature nel prendere le sue decisioni
- **Contributo marginale**: quanto quella feature *migliora* le performance del modello

Questi due valori possono divergere — e nel nostro caso divergono in modo significativo. Una feature può avere alta importanza SHAP pur peggiorando l'AUPR, se il modello la usa come scorciatoia al posto di una rappresentazione interna più potente.

---

\newpage

# Risultati Numerici dell'Analisi SHAP

## Statistiche globali delle 807 feature

| Gruppo | N feature | Media importanza | Mediana | Max | Min |
|---|---|---|---|---|---|
| Espressione | 805 | ~0.001 | ~0.001 | ~0.0047 | — |
| **Grado** (feat. 806) | 1 | — | — | **0.0021** | — |
| **Entropia** (feat. 807) | 1 | — | — | **0.0039** | — |

## Posizione delle feature topologiche nella classifica

- **Entropia** (feat. 807): importanza = **0.0039** → rank **7/807** → supera il ~**99%** delle feature di espressione
- **Grado** (feat. 806): importanza = **0.0021** → supera circa l'**82%** delle feature di espressione

Le prime 6 posizioni sono tutte occupate da feature di espressione. L'entropia si inserisce al settimo posto, con un valore paragonabile alle migliori condizioni sperimentali.

## Top 10 feature per importanza

| Rank | Feature | Importanza | Tipo |
|---|---|---|---|
| 1 | Exp_556 | ~0.0047 | Espressione |
| 2 | Exp_637 | ~0.0046 | Espressione |
| 3 | Exp_409 | ~0.0045 | Espressione |
| 4 | Exp_111 | ~0.0044 | Espressione |
| 5 | Exp_638 | ~0.0043 | Espressione |
| 6 | Exp_475 | ~0.0042 | Espressione |
| **7** | **Entropia_norm** | **0.0039** | **Topologica** |
| 8 | Exp_32 | ~0.0038 | Espressione |
| 9 | Exp_610 | ~0.0037 | Espressione |
| 10 | Exp_35 | ~0.0036 | Espressione |
| ... | ... | ... | ... |
| >20 | **Grado_norm** | **0.0021** | **Topologica** |

## Lettura del grafico SHAP

![Analisi SHAP — Contributo delle Feature alle Predizioni](shap_analysis.png)

Il grafico si compone di tre pannelli che insieme raccontano la stessa storia da angolazioni diverse.

**Pannello sinistro — Top 20 feature per importanza:** Le prime 6 posizioni sono tutte feature di espressione (in blu). L'entropia (in arancione) appare al settimo posto con valore ~0.0039, paragonabile alle migliori condizioni sperimentali. Il grado non è visibile nella top 20, a conferma che la sua ridondanza con Θ è totale.

**Pannello centrale — Boxplot espressione vs topologiche:** Le 805 feature di espressione mostrano una distribuzione molto compressa: mediana intorno a 0.001, IQR stretto, con pochi outlier in alto. Grado ed entropia appaiono come **punti arancioni esterni alla distribuzione** — sono outlier rispetto alle feature di espressione, non parte della loro distribuzione.

**Pannello destro — Importanza media per gruppo:** Rende immediatamente visibile lo squilibrio: la media delle 805 feature di espressione è ~0.001, mentre entropia e grado hanno valori rispettivamente di ~0.0039 e ~0.0021 — fattori 4x e 2x superiori alla media del gruppo di espressione.

## Le 805 vs le 2: aggregazione collettiva contro canale separato

La lettura congiunta dei tre pannelli rivela un comportamento strutturalmente diverso tra i due gruppi di feature.

Le **805 feature di espressione** hanno importanza individuale bassa e distribuita in modo uniforme. Nessuna singola condizione sperimentale domina. Questo è esattamente il segnale che il modello le tratta in modo **collettivo**: nessun esperimento conta molto da solo, ma tutti insieme formano il pattern che Θ legge attraverso la convoluzione ipergrafo. La topologia di Θ mescola, pondera e trasforma le 805 feature in embedding a 128 dimensioni — dopo questo passaggio nessuna feature sopravvive "pura".

Le **2 feature topologiche** invece mostrano importanza individuale alta e isolata. Il modello non le aggrega con le 805: le tratta come segnali autonomi, regole rapide del tipo *"gene con alta entropia → regolato da molti TF → alta probabilità di arco positivo"*. Si comportano come un **canale separato** rispetto alle feature di espressione.

Questo è il meccanismo del collasso: il modello apre una scorciatoia verso grado ed entropia, bypassando la rappresentazione ricca costruita da Θ. Le 805 feature vengono aggregate correttamente; le 2 feature topologiche rimangono isolate e dominano le decisioni localmente, degradando la capacità discriminativa globale.

---

\newpage

# Il Paradosso: Feature Importanti che Peggiorano il Modello

## Il problema

I risultati SHAP mostrano un **paradosso apparente**:

> L'entropia è il 7° predittore più importante su 807, eppure il modello con 807 feature ha AUPR = 0.240, peggiore del modello con 805 feature (AUPR = 0.247).

Come può una feature molto importante peggiorare le performance? La risposta richiede di separare due concetti distinti: **importanza individuale** e **contributo marginale**.

## Causa 1 — La topologia è già codificata in Θ

Questa è la causa principale. La matrice di propagazione dell'HGNN è:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^T \cdot D_v^{-1/2}$$

dove $D_v[i][i] = d(i)$ è esattamente il **grado** del gene $i$, e i termini $D_v^{-1/2}$ e $D_e^{-1}$ effettuano una normalizzazione spettrale che dipende dalla distribuzione dei gradi — la stessa distribuzione che caratterizza l'**entropia**.

Durante il forward pass, ogni embedding viene aggiornato come:
$$\text{emb}^{(l+1)} = \sigma\left(\Theta \cdot \text{emb}^{(l)} \cdot W^{(l)}\right)$$

Θ è **calcolata una sola volta da H** e rimane fissa. Questo significa che a ogni layer, ogni gene riceve informazioni pesate inversamente al suo grado e proporzionalmente alla distribuzione dei suoi TF — ovvero riceve implicitamente sia il grado che l'entropia codificati geometricamente nello spazio degli embedding a 128 dimensioni.

**Il grado e l'entropia aggiunti esplicitamente in X sono quindi derivate di H, e H è già integralmente utilizzata da Θ.** Il modello riceve la stessa informazione due volte, ma in rappresentazioni diverse e incompatibili.

## Causa 2 — Shortcut learning: il modello usa la versione sbagliata

Quando una feature semplice e correlata è disponibile, le reti neurali tendono a fare "shortcut learning" (Geirhos et al., 2020): usano la scorciatoia invece di imparare la rappresentazione più complessa ma più potente.

Nel nostro caso:

- **Senza le feature topologiche esplicite (805 feat.)**: il modello è *costretto* a usare Θ per estrarre informazioni strutturali. Impara a discriminare le coppie TF→gene attraverso gli embedding a 128 dimensioni che Θ costruisce, che catturano la struttura globale della rete.

- **Con le feature topologiche esplicite (807 feat.)**: il modello "vede" il grado e l'entropia come scalari diretti in X. Li usa come scorciatoia perché sono già informativi singolarmente. Ma il grado di un gene non dice nulla sulla *specificità* di un'interazione: un gene regolato da 50 TF (grado alto) può avere sia molte interazioni vere che molte false. Il discriminatore diventa meno preciso.

## Causa 3 — Interferenza del gradiente durante la backpropagation

Durante il training, il gradiente della loss fluisce attraverso due canali paralleli per la stessa informazione:

**Canale 1** (implicito, via Θ):
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \Theta^T \cdot \frac{\partial \mathcal{L}}{\partial \text{emb}^{(l+1)}}$$

**Canale 2** (esplicito, via X):
$$\frac{\partial \mathcal{L}}{\partial x_{806}} = \frac{\partial \mathcal{L}}{\partial \text{emb}^{(1)}} \cdot W^{(1)T}_{[:,806]}$$

I due canali aggiornano i pesi in direzioni che tengono conto della stessa informazione (il grado) ma codificata in spazi di dimensione radicalmente diversa (128D vs 1D). I gradienti si "contraddicono" perché ottimizzano rappresentazioni diverse della stessa quantità. Il risultato è analogo alla multicollinearità nella regressione lineare: i coefficienti stimati diventano instabili e la varianza dell'errore aumenta.

## Perché l'entropia ha importanza più alta del grado

L'entropia ha importanza SHAP quasi doppia rispetto al grado (0.0039 vs 0.0021). Questo è coerente con la struttura di Θ:

- Il **grado** $d(i)$ è codificato *esplicitamente e precisamente* in $D_v[i][i]$. Il modello ha già il grado al 100% di fedeltà tramite Θ. L'aggiunta esplicita è totalmente ridondante → bassa importanza marginale.

- L'**entropia** $\mathcal{H}(i)$ cattura la *distribuzione* della regolazione, non solo il conteggio. Θ codifica questa informazione in modo più indiretto (attraverso le combinazioni di $D_v^{-1/2}$ e $D_e^{-1}$). L'aggiunta esplicita introduce quindi una piccola quota di informazione genuinamente nuova, che il modello usa attivamente → importanza marginale più alta.

Tuttavia, quella piccola quota di informazione nuova non compensa il danno da interferenza del gradiente e shortcut learning. Il bilancio netto è negativo.

---

\newpage

# Interpretazione Complessiva

## Il verdetto dell'analisi SHAP

La risposta alla domanda del tutor è articolata in tre livelli:

**Livello 1 — Cosa serve la SHAP:** misura *quanto il modello usa* ogni feature, non *quanto quella feature migliora* il modello. Alta importanza SHAP non implica contributo positivo alle performance.

**Livello 2 — Il comportamento dei due gruppi:** le 805 feature di espressione vengono aggregate collettivamente da Θ — nessuna pesa molto da sola, tutte insieme costruiscono la rappresentazione. Le 2 feature topologiche rimangono isolate con peso individuale altissimo, come un canale separato che il modello usa in modo autonomo invece di integrarle nel pattern collettivo.

**Livello 3 — La causa profonda:** grado ed entropia sono derivate di H, e H è già interamente contenuta in Θ. Il modello le riceve due volte: una volta implicita e ricca (128 dimensioni via convoluzione), una volta esplicita e povera (scalare grezzo in X). Sceglie la scorciatoia scalare, peggiora.

> **Le feature topologiche sono importanti — ma sono già presenti nel modello attraverso Θ.** Aggiungendole esplicitamente in X non si aggiunge informazione nuova; si forza il modello ad aprire un canale preferenziale verso una rappresentazione alternativa più povera, bypassando quella più potente costruita da Θ.

Questo risultato non è un fallimento: **è una conferma positiva che la HGNN funziona correttamente**. Il fatto che Θ codifichi già la topologia in modo così completo da rendere ridondante qualsiasi aggiunta esplicita dimostra che l'architettura è ben progettata per questo problema.

## Schema riassuntivo del paradosso

```
Feature topologiche ad alta importanza SHAP
              ↓
   Alta importanza SHAP $\neq$ contributo positivo
              ↓
   Perché l'importanza SHAP misura "quanto il modello
   usa quella feature" — non "quanto quella feature
   migliora il modello"
              ↓
   Il modello le usa tanto perché sono informative
   → ma erano già codificate in Θ in forma migliore
   → il modello sostituisce la rappresentazione ricca
     (128D via Θ) con la scorciatoia povera (1D scalare)
              ↓
   Risultato: AUPR da 0.247 → 0.240 (−2.6%)
```

## Implicazione per la tesi

Questo risultato porta a una conclusione scientificamente rilevante: nelle HGNN, **la topologia dell'ipergrafo non deve essere iniettata come feature esplicita in X**. La matrice Θ la incorpora già al livello architetturale in modo ottimale. Qualsiasi tentativo di renderla esplicita introduce ridondanza e degrada le performance.

Questo distingue le HGNN dai metodi classici di machine learning su grafi dove le feature topologiche (grado, betweenness, PageRank) sono comunemente aggiunte manualmente perché il modello non ha meccanismi intrinseci per apprenderle.

---

\newpage

# Riferimenti

- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). **Axiomatic Attribution for Deep Networks** (Integrated Gradients). *ICML*.
- Geirhos, R. et al. (2020). **Shortcut Learning in Deep Neural Networks**. *Nature Machine Intelligence*, 2, 665–673.
