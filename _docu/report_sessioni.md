---
title: "Sintesi delle Sessioni di Lavoro — GRN-Hypergraph-Classifier"
author: "Francesco Rollin, Fabio"
date: "22 Giugno 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.3
toc: true
toc-depth: 3
numbersections: true
---

\newpage

# Introduzione

Questo documento riassume il percorso di sviluppo del progetto **GRN-Hypergraph-Classifier**, dalla comprensione iniziale del dataset DREAM5 fino alle ultime sperimentazioni con il curriculum learning. Il lavoro è suddiviso in tre macro-sessioni che corrispondono ai tre momenti di avanzamento principali documentati nelle chat di lavoro.

Il progetto, sviluppato per il corso di Bioinformatica dell'Università di Salerno, ha come obiettivo l'inferenza di reti di regolazione genica (GRN) su *Escherichia coli* usando Hypergraph Neural Networks (HGNN), confrontando diversi approcci di costruzione e training rispetto ai metodi benchmark di DREAM5.

---

\newpage

# Sessione 1 — Comprensione del Dominio e Costruzione della Pipeline

## Contesto

La prima sessione ha riguardato l'acquisizione delle basi teoriche e la costruzione della pipeline sperimentale. Il punto di partenza è stato l'email del tutor (Gerardo), che indicava tre pilastri di studio: il dataset DREAM5, i fattori di trascrizione, e gli ipergrafi.

## I tre paper fondamentali

**Paper 1 — Latchman (1997): Transcription Factors: An Overview**
I fattori di trascrizione (TF) sono proteine che si legano a specifiche sequenze di DNA e regolano la trascrizione genica. Funzionano come attivatori o repressori. Nel progetto, ogni TF corrisponde a un **iperarco** nella matrice di incidenza H: un singolo TF può regolare decine o centinaia di geni contemporaneamente, rendendo l'ipergrafo lo strumento naturale per modellare questa relazione uno-a-molti.

**Paper 2 — Marbach et al. (2012): DREAM5**
Il dataset DREAM5 raccoglie oltre 30 metodi di inferenza delle GRN valutati su quattro organismi. I metodi si dividono in due categorie: metodi statistici (mutual information, regressione, reti bayesiane) e il metodo ensemble (Community Network). Nessun singolo metodo domina su tutti i dataset, mentre l'ensemble è il più robusto. Il nostro progetto si propone di confrontarsi con questi metodi usando un approccio supervisionato basato su deep learning.

**Paper 3 — Feng et al. AAAI 2019: Hypergraph Neural Networks**
La HGNN estende le GCN agli ipergrafi tramite la matrice di propagazione:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^T \cdot D_v^{-1/2}$$

dove H è la matrice di incidenza, W è la matrice dei pesi degli iperarchi, $D_v$ e $D_e$ sono le matrici diagonali dei gradi di nodi e iperarchi. Θ rimane fissa durante il training; solo i pesi della rete vengono aggiornati.

## Il dataset DREAM5 Network 1

Il dataset contiene:
- **1643 geni** misurati in **805 esperimenti** di espressione
- **195 fattori di trascrizione (TF)**
- **4012 interazioni vere** (positive) nel gold standard
- **274380 interazioni false** (negative)
- Sbilanciamento di **1:68** — ogni coppia vera è accompagnata da 68 false

## La pipeline implementata

La pipeline è strutturata in quattro step modulari, orchestrati da `main.py`:

- **step1.py** — Carica i tre file DREAM5 (expression data, TF list, gold standard) e normalizza la matrice di espressione X
- **step2.py** — Costruisce H (HGNN) o A (GCN) e gestisce il masking train/test con le diverse strategie di campionamento
- **step3.py** — Allena HGNN o GCN con early stopping su AUPR di validazione
- **step4.py** — Calcola AUPR, AUROC, precision, recall e F1 sul test set

## Confronto tra approcci

Una scoperta chiave di questa fase è stata la distinzione tra tre tipi di costruzione della struttura del grafo/ipergrafo:

| Approccio | Struttura costruita da | Tipo |
|---|---|---|
| **Gold standard** | Interazioni biologiche certificate | Supervisionato |
| **Similarità statistica (Rosario-like)** | Correlazione di Pearson + KNN | Statistico |
| **Edge prediction** | Gold standard completo (H=100%) + masking test | Supervisionato |

L'approccio **edge prediction** si è rivelato superiore: invece di rimuovere archi per costruire H, usa il gold standard completo e maschera solo le coppie da predire nel test set. Questo produce una Θ con rank effettivo più alto e propagazione più ricca.

## Incontro con il tutor

Il tutor ha confermato l'obiettivo: replicare il lavoro di Rosario sulla GCN, ma con gli ipergrafi. Ha dato tre indicazioni operative:
1. Partire da Network 1 per allenare, usare Network 2 per testare
2. Tenere un diario del lavoro
3. Non preoccuparsi se i risultati non sono perfetti: il contributo metodologico è l'uso degli ipergrafi

---

\newpage

# Sessione 2 — Analisi Topologica dell'Ipergrafo

## Motivazione

Dopo aver ottenuto i risultati della prima ondata di esperimenti, il tutor ha posto tre domande scientifiche specifiche:

1. Quali caratteristiche topologiche dei nodi spiegano le differenze di performance tra stratified e hard negative?
2. Come si propaga l'informazione attraverso Θ nei diversi approcci?
3. Le metriche topologiche, aggiunte come feature esplicite, migliorano le performance?

## Struttura matematica dell'ipergrafo

La matrice di incidenza H è definita come:

$$H[i][j] = \begin{cases} 1 & \text{se TF}_j \text{ regola il gene}_i \\ 0 & \text{altrimenti} \end{cases}$$

Per il Network 1 di DREAM5: H ha dimensione 1643 × 195, con densità del 1.25%.

Θ viene calcolata una volta sola da H e rimane fissa durante tutto il training. Il forward pass a ogni epoca è:

$$\text{emb}^{(l+1)} = \sigma\left(\Theta \cdot \text{emb}^{(l)} \cdot W^{(l)}\right)$$

## Risultati dell'analisi topologica

### 1. Bottleneck strutturale: il TF G120

L'analisi del grado dei TF ha rilevato una distribuzione estremamente sbilanciata. Il TF G120 regola il **26.3% di tutti i geni target** (432 su 1643). Questo crea un bottleneck nella propagazione di Θ: il modello rischia di imparare prevalentemente dalla struttura di G120, ignorando i TF con pochi target.

Soluzione proposta: **pesaggio inversamente proporzionale al grado** degli iperarchi nella matrice W di Θ, cioè $W_{jj} = 1/\sqrt{\text{deg}(j)}$. Sperimentata nel Blocco 5 con AUPR ≈ 0.065: l'AUROC rimane alto (0.82) ma la calibrazione del decoder si rompe, suggerendo che il problema non è nella propagazione ma nella fase di scoring.

### 2. Analisi spettrale di Θ

Confrontando H costruita con il 100% degli archi (edge prediction) vs H costruita con l'80% (gold standard):

- H completa: **447** componenti connesse effettive, autovalori più distribuiti
- H ridotta: **434** componenti connesse, autovalori più concentrati

Il rank effettivo superiore spiega il miglioramento da AUPR 0.068 (gold standard) a AUPR 0.247 (edge prediction): H completa propaga informazioni più ricche tra geni co-regolati.

### 3. Il vantaggio di stratified è metodologico, non topologico

Confrontando le proprietà dei nodi mascherati in stratified vs random:
- Grado medio nodi mascherati: 3.91 (stratified) vs 3.92 (random) — differenza +0.3%
- Entropia media: sostanzialmente identica

Il vantaggio di stratified (AUPR 0.247 vs 0.227) non deriva da differenze topologiche nei nodi scelti per il test, ma dal **bilanciamento per TF**: in stratified, ogni TF contribuisce proporzionalmente al training set, evitando che i TF con molti target dominino il gradiente.

### 4. Hard negative: un problema sistemico

Senza curriculum learning, l'hard negative sampling porta AUPR da 0.247 a 0.068. L'analisi degli embeddings mostra che la perturbazione è **uniforme su tutti i geni**, non correlata con il grado o l'entropia. Anche geni con un solo TF regolatore mostrano grandi differenze di embedding rispetto a stratified. I negativi difficili rendono il problema di ottimizzazione instabile fin dall'inizio.

### 5. Θ codifica già la topologia

Le feature topologiche esplicite (grado ed entropia) aggiunte in X (805→807 feature) non migliorano le performance:
- HGNN + topo features: AUPR 0.240 (−2.6% rispetto a baseline 0.247)
- HGNN hard + topo features: AUPR 0.062 (−8.6% rispetto a hard senza topo)

**Conclusione:** Θ codifica già implicitamente la struttura topologica durante la convoluzione ipergrafo. Aggiungere le stesse informazioni in X introduce ridondanza che disturba il training.

### 6. Analisi SHAP (feature importance)

Usando Integrated Gradients (Sundararajan 2017), compatibile con le GNN, è stato analizzato quali degli 805 esperimenti di espressione genica influenzano maggiormente la predizione. I risultati mostrano che le feature più importanti corrispondono agli esperimenti con condizioni di stress biologico (antibiotici, variazioni di temperatura), coerente con la letteratura sui TF in *E. coli*.

---

\newpage

# Sessione 3 — Curriculum Learning (22 Giugno 2026)

## Motivazione

Dall'analisi topologica era emerso chiaramente che il problema dell'hard negative non è topologico ma di ottimizzazione: presentare subito negativi molto difficili impedisce al modello di imparare le rappresentazioni di base. La soluzione naturale è il **curriculum learning** (Bengio et al., 2009): presentare esempi in ordine crescente di difficoltà.

## Implementazione

Il curriculum learning implementato in `step3.py` divide il training in **tre fasi**:

| Fase | Epoche | Tipo di negativi | Obiettivo |
|---|---|---|---|
| **Fase 1** | 0 → ratio × epochs | Random (facili) | Il modello impara le rappresentazioni di base |
| **Fase 2** | ratio → (ratio+0.5)/2 × epochs | 50% random + 50% hard | Transizione graduale |
| **Fase 3** | resto | Hard stratified (difficili) | Fine-tuning con negativi sfidanti |

Il parametro chiave è `curriculum_phase1_ratio`: la frazione di epoche totali dedicata alla fase 1.

## Esperimenti e risultati

### Grid search sul ratio

| Esperimento | Phase1 ratio | Epoche totali | AUPR | vs Hard (no curriculum) | vs Baseline |
|---|---|---|---|---|---|
| hgnn_curriculum (default) | 0.33 | 100 | 0.143 | +112% | −42% |
| hgnn_curriculum_60 | 0.60 | 150 | 0.240 | +255% | −3% |
| hgnn_curriculum_70 | **0.70** | **150** | **0.266** | **+293%** | **+8%** |
| hgnn_curriculum_80 | 0.80 | 150 | 0.266 | +293% | +8% |
| hgnn_curriculum_60_ep200 | 0.60 | 200 | 0.220 | — | −11% |

**Trovata:** ratio=0.70 e ratio=0.80 producono risultati identici (plateau). Aumentare le epoche a 200 con ratio=0.60 peggiora per overfitting nella Fase 3.

### Varianti testate

**Patience variabile per fase** (`--curriculum_patience 10,25,20`):
- AUPR = 0.253 — peggiora di 4.9% rispetto a curriculum_70
- La patience non è il fattore limitante; il problema è altrove

**Curriculum + arc features** (`--use_arc_features`):
- AUPR = 0.228 — i due metodi interferiscono
- Arc features e curriculum learning mirano entrambi a migliorare il decoder; la combinazione sovrascrive i segnali

**Curriculum sigmoid** (transizione continua invece di a fasi):
- AUPR = 0.224 — la gradualità continua non aiuta
- Le fasi discrete sono efficaci: il salto netto alla Fase 3 è utile

### Online Hard Negative Mining

Alternativa al curriculum: ricalcolare dinamicamente i negativi più difficili per il modello attuale ogni N epoche.

| Esperimento | Warmup | AUPR | Note |
|---|---|---|---|
| hgnn_online_mining | 0 ep | 0.024 | Crolla: ep.10 il modello è ancora casuale |
| hgnn_online_mining_w60 | 60 ep | 0.208 | Migliora ma si ferma a ep.60 |

Il mining senza warmup è inutilizzabile (gli embeddings casuali selezionano negativi casuali, non difficili). Con warmup=60, il modello si ferma all'inizio del mining senza imparare ulteriormente.

## Risultato finale: superamento di CLR

Il curriculum_70 (AUPR = 0.266) supera per la prima volta un metodo DREAM5:

```
DREAM5 CLR (Mutual Information):  AUPR = 0.255
HGNN Curriculum 70%:               AUPR = 0.266  ← +4.3% vs CLR ✅
```

---

\newpage

# Quadro Riassuntivo Complessivo

## Classifica finale per AUPR

| Posizione | Modello | AUPR | Tipo |
|---|---|---|---|
| 1 | DREAM5 Community (ensemble) | 0.327 | Benchmark |
| 2 | DREAM5 Best Regression | 0.313 | Benchmark |
| 3 | DREAM5 TIGRESS | 0.301 | Benchmark |
| 4 | DREAM5 GENIE3 | 0.291 | Benchmark |
| **5** | **HGNN Curriculum 70% / 80%** | **0.266** | **← Nostro miglior risultato** |
| 6 | DREAM5 CLR | 0.255 | Benchmark (superato) |
| 7 | HGNN Edge Stratified | 0.247 | Nostro baseline |
| 8 | HGNN Edge Stratified + Topo | 0.240 | Topologia ridondante |
| 9 | HGNN Curriculum 60% | 0.240 | — |
| 10 | GCN Statistical (Rosario) | 0.203 | Struttura da similarità |
| 11 | HGNN Statistical (Rosario) | 0.184 | — |
| 12 | GCN Gold Standard | 0.151 | — |
| 13 | HGNN Hard Stratified (no CL) | 0.068 | Collasso senza curriculum |
| 14 | HGNN Gold Standard (baseline) | 0.068 | — |

## Matrice 2×2: struttura vs modello

|  | Gold Standard / Edge prediction | Similarità (Rosario-like) |
|---|---|---|
| **HGNN** | 0.266 (curriculum) | 0.184 |
| **GCN** | 0.151 (gold) | 0.203 |

**Lettura:** la struttura biologica (gold standard + edge prediction) supera la struttura statistica per HGNN. Curiosamente, per GCN è l'opposto: la similarità (0.203) supera il gold standard (0.151). Questo suggerisce che la GCN, operando su grafi classici, sfrutta meglio la co-espressione rispetto all'HGNN, che invece beneficia della struttura a iperarchi biologici.

## Progressione temporale

| Versione | Data | Novità | AUPR best |
|---|---|---|---|
| src_4 | Aprile 2026 | Pipeline, ablation, GCN/Rosario, edge prediction | 0.247 |
| src_5 | 17 Giugno 2026 | Hard negative sampling, analisi topologica | 0.247 |
| src_XIV | 22 Giugno 2026 | Curriculum learning, SHAP, online mining | **0.266** |

---

\newpage

# Sviluppi Futuri

In ordine di priorità suggerito dall'analisi:

**1. Validazione su Network 2 e 3 (S. aureus, S. cerevisiae)**
Il modello è stato sviluppato interamente su Network 1. Verificare la generalizzazione su altri organismi è il passo successivo più importante per la tesi.

**2. Curriculum learning sul ratio della Fase 2**
Il plateau a ratio=0.70/0.80 suggerisce che la Fase 2 (mixed) potrebbe essere ottimizzata: durata, composizione (30/70 invece di 50/50), o uso di un diverso tipo di negativi.

**3. Confronto con GCN di Rosario (metodo originale)**
Implementare il codice esatto di Rosario e rieseguirlo sui dati DREAM5 per avere un confronto diretto, non solo una replica semplificata.

**4. Curriculum learning con warmup + mining**
Combinare curriculum (Fase 1 = warmup) con online mining in Fase 3, invece di negativi hard statici. Il mining con warmup=60 ha mostrato AUPR 0.208: con un curriculum più lungo come warmup potrebbe migliorare ulteriormente.

**5. Feature a livello di arco**
Invece di aggiungere il grado al nodo (arc features nel decoder), aggiungere proprietà dell'iperarco (peso calcolato dalla mutual information tra TF e target) direttamente nella matrice W di Θ.

**6. Integrazione di dati genomici aggiuntivi**
Le 805 feature attuali sono esperimenti di microarray. Aggiungere sequenze promotore, dati ChIP-seq, o struttura cromatinica potrebbe migliorare significativamente le performance superando il tetto di GENIE3 (0.291).

---

\newpage

# Riferimenti

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). **Curriculum Learning**. *ICML*.
- Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with GCN**. *ICLR*.
- Latchman, D.S. (1997). **Transcription Factors: An Overview**. *Int. J. Biochem. Cell Biol.*, 29(12), 1305–1312.
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). **Axiomatic Attribution for Deep Networks** (Integrated Gradients). *ICML*.
