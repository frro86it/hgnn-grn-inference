---
title: "Analisi Topologica della Rete di Regolazione Genica — HGNN su DREAM5 Network 1"
author: "Francesco Rollin, Fabio"
date: "Giugno 2026"
geometry: margin=2.5cm
fontsize: 11pt
linestretch: 1.3
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
---

\newpage

# Introduzione e Motivazione

## Contesto del progetto

Questo documento descrive le analisi topologiche condotte sulla rete di regolazione genica (GRN) di *Escherichia coli* del dataset DREAM5 Network 1, in seguito ai risultati sperimentali ottenuti con le Hypergraph Neural Networks (HGNN) e le Graph Convolutional Networks (GCN).

Il dataset DREAM5 Network 1 è composto da:

- **1643 geni** misurati in 805 esperimenti di espressione
- **195 fattori di trascrizione (TF)**
- **4012 interazioni vere** (Label=1) certificate biologicamente
- **274380 interazioni false** (Label=0)
- **Sbilanciamento**: 68 interazioni false per ogni interazione vera

## Problema posto dal tutor

Avendo osservato differenze significative nelle performance tra i vari approcci — in particolare tra l'edge prediction con stratified masking (AUPR = 0.247) e l'hard negative sampling (AUPR = 0.068) — il tutor ha chiesto di rispondere a tre domande scientifiche:

1. **Quali caratteristiche topologiche dei nodi** (geni e TF) spiegano le differenze di performance?
2. **Come si propaga l'informazione** attraverso la matrice $\Theta$ nei diversi approcci?
3. **Le metriche topologiche**, se aggiunte come feature esplicite al modello, **migliorano le performance**?

\newpage

# Struttura Matematica dell'Ipergrafo

## Definizioni fondamentali

Prima di descrivere le analisi, è necessario definire le strutture matematiche utilizzate.

### La matrice di incidenza H

L'ipergrafo biologico è rappresentato dalla **matrice di incidenza** $H$ di dimensione $n \times m$, dove $n = 1643$ è il numero di geni e $m = 195$ è il numero di TF (iperarchi):

$$H[i][j] = \begin{cases} 1 & \text{se il TF}_j \text{ regola il gene}_i \\ 0 & \text{altrimenti} \end{cases}$$

**Esempio concreto**: Se TF_A (colonna 3) regola G119 (riga 5):

$$H[5][3] = 1, \quad H[5][k] = 0 \text{ per } k \neq 3 \text{ (se G119 fosse regolato solo da TF\_A)}$$

### La matrice di propagazione Θ

La matrice $\Theta$ è il cuore della HGNN. Viene calcolata una sola volta da $H$ e rimane fissa durante tutto il training. È definita come [@feng2019hypergraph]:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^T \cdot D_v^{-1/2}$$

dove:

- $D_v \in \mathbb{R}^{n \times n}$ = matrice diagonale dei gradi dei nodi: $D_v[i][i] = \sum_j H[i][j]$
- $D_e \in \mathbb{R}^{m \times m}$ = matrice diagonale dei gradi degli iperarchi: $D_e[j][j] = \sum_i H[i][j]$
- $W \in \mathbb{R}^{m \times m}$ = matrice diagonale dei pesi degli iperarchi (identità nel nostro caso)

**Ruolo di Θ**: $\Theta[i][j]$ indica quanto il gene $j$ influenza l'embedding del gene $i$ durante la propagazione. È l'analogo del **Laplaciano normalizzato** dei grafi standard, esteso agli ipergrafi.

### Il meccanismo di forward pass

Ad ogni epoca del training, l'embedding di ogni gene viene calcolato come:

$$\text{emb}^{(l+1)} = \sigma\left(\Theta \cdot \text{emb}^{(l)} \cdot W^{(l)}\right)$$

dove $W^{(l)}$ è la matrice dei pesi apprendibile al layer $l$ e $\sigma$ è la funzione di attivazione ReLU. **Θ è fissa**, solo $W$ viene aggiornata durante il training.

\newpage

# Analisi 1 — Metriche Topologiche per Nodo

## Motivazione

La prima domanda del tutor è: **i geni hanno caratteristiche strutturali diverse che li rendono più o meno difficili da classificare?** Per rispondere, calcoliamo due metriche per ogni gene: il **grado** e l'**entropia topologica**.

## Il Grado del Nodo

### Definizione

Il **grado** $d(v_i)$ di un gene $i$ è il numero di TF che lo regolano:

$$d(v_i) = \sum_{j=1}^{m} H[i][j]$$

**Esempio su G119**: Il gene G119 è regolato da 12 TF diversi. Nella sua riga di $H$, 12 colonne hanno valore 1 e le restanti 183 hanno valore 0:

$$d(G119) = \sum_{j=1}^{195} H[\text{G119}][j] = 12$$

### Interpretazione biologica

| Grado | Significato biologico | Difficoltà classificazione |
|-------|----------------------|---------------------------|
| $d = 0$ | Gene non regolato nel gold standard | Non classificabile |
| $d = 1$ | Regolato da 1 solo TF (molto specifico) | Facile |
| $d = 2-3$ | Regolazione moderata | Media |
| $d \geq 8$ | Gene hub, regolato da molti TF | Difficile |

### Risultati ottenuti

![Distribuzione del grado e dell'entropia dei nodi nell'ipergrafo HGNN. In alto a sinistra: distribuzione del grado per gene. In alto al centro: distribuzione dell'entropia topologica. In alto a destra: correlazione grado-entropia.](./output/topology_analysis/topology_analysis.png)

**Tabella 1**: Statistiche del grado per gene

| Metrica | Valore |
|---------|--------|
| Geni totali | 1643 |
| Geni con almeno 1 TF | 1498 (91.2%) |
| Geni non regolati (d=0) | 145 (8.8%) |
| Grado medio | 2.68 TF/gene |
| Grado massimo | 12 (gene G119) |

I geni con grado 0 rappresentano un caso speciale: il modello non riceve nessuna informazione strutturale per questi geni tramite $\Theta$, e il loro embedding dipende esclusivamente dalle feature di espressione $X$.

## L'Entropia Topologica

### Motivazione

Il grado misura **quanti** TF regolano un gene, ma non dice **quanto uniformemente** è distribuita quella regolazione. Due geni con grado 3 potrebbero avere regolazioni molto diverse:

- Gene A: regolato da TF_1 con peso dominante (90%) e TF_2, TF_3 marginali
- Gene B: regolato da TF_1, TF_2, TF_3 con peso uguale (33% ciascuno)

Per catturare questa differenza usiamo l'**entropia di Shannon** [@shannon1948mathematical].

### Definizione

Per ogni gene $i$, definiamo la distribuzione di probabilità tra i suoi TF regolatori:

$$p_{ij} = \frac{H[i][j]}{\sum_{k=1}^{m} H[i][k]} = \frac{H[i][j]}{d(v_i)}$$

L'entropia topologica è quindi:

$$\mathcal{H}(v_i) = -\sum_{j=1}^{m} p_{ij} \cdot \log_2(p_{ij})$$

con la convenzione $0 \cdot \log_2(0) = 0$.

### Esempio di calcolo su G119

G119 è regolato da 12 TF. Nella nostra implementazione, tutti i valori in $H$ sono binari (0 o 1), quindi ogni TF regolatore ha lo stesso peso:

$$p_{ij} = \frac{1}{12} \approx 0.0833 \quad \text{per ciascuno dei 12 TF}$$

$$\mathcal{H}(G119) = -12 \times \left(\frac{1}{12} \times \log_2\frac{1}{12}\right) = \log_2(12) \approx 3.585 \text{ bit}$$

### Esempio di calcolo su un gene con grado 1

Un gene regolato da un solo TF ha:

$$p_{ij} = 1 \text{ per il TF regolatore}, \quad p_{ij} = 0 \text{ per tutti gli altri}$$

$$\mathcal{H}(v_i) = -(1 \times \log_2 1) = 0 \text{ bit}$$

Entropia zero significa **certezza totale**: sappiamo esattamente quale TF regola il gene.

### Interpretazione

$$\mathcal{H}(v_i) \in [0, \log_2(d(v_i))]$$

- **$\mathcal{H} = 0$ bit**: gene regolato da un solo TF → embedding guidato da una sola fonte → classificazione più semplice
- **$\mathcal{H}$ alta**: gene regolato da molti TF con pesi simili → embedding deve mediare segnali multipli → classificazione più difficile → i negativi hard hanno più impatto

### Risultati ottenuti

**Top 10 geni più regolati** (e più ambigui):

| Gene | Grado $d(v)$ | Entropia $\mathcal{H}(v)$ [bit] |
|------|-------------|--------------------------------|
| G119 | 12 | 3.585 |
| G115 | 11 | 3.459 |
| G796 | 11 | 3.459 |
| G1203 | 10 | 3.322 |
| G814 | 10 | 3.322 |
| G851 | 10 | 3.322 |
| G1230 | 9 | 3.170 |
| G394 | 9 | 3.170 |
| G934 | 9 | 3.170 |
| G129 | 9 | 3.170 |

## Il Grado dei TF (Iperarchi)

### Definizione

Il grado di un TF (iperarco) misura quanti geni regola:

$$\delta(e_j) = \sum_{i=1}^{n} H[i][j]$$

### Risultato critico — il bottleneck G120

La distribuzione del grado dei TF è **estremamente sbilanciata**:

| Fascia di grado | N° TF | % sul totale TF |
|----------------|-------|----------------|
| 1–5 geni | 55 | 28.2% |
| 6–10 geni | 54 | 27.7% |
| 11–20 geni | 46 | 23.6% |
| 21–50 geni | 32 | 16.4% |
| 51–100 geni | 0 | 0% |
| > 100 geni | 8 | 4.1% |

Il TF G120 regola **432 geni** su 1643 totali, pari al **26.3%** di tutti i target. Questo crea un **bottleneck biologico**: la propagazione in $\Theta$ è dominata da questo singolo TF, e il modello tende a imparare principalmente da esso.

**Implicazione pratica**: una possibile estensione del modello sarebbe pesare i TF inversamente al loro grado nella matrice $W$ di $\Theta$, per ridurre il dominio dei TF con molti target.

\newpage

# Analisi 2 — Analisi Spettrale di Θ

## Motivazione

La seconda domanda del tutor riguarda **come l'informazione si propaga** attraverso la rete. La teoria spettrale dei grafi ci fornisce gli strumenti per rispondere: gli **autovalori** di $\Theta$ descrivono le modalità di propagazione dell'informazione.

## Teoria spettrale applicata a Θ

Poiché $\Theta$ è simmetrica reale, ammette la decomposizione spettrale:

$$\Theta = U \Lambda U^T$$

dove $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ con $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_n$.

Ogni autovalore $\lambda_k$ descrive una **modalità di propagazione**:

- **$\lambda_k$ grande** (vicino a 1): modalità in cui l'informazione si propaga velocemente e con intensità
- **$\lambda_k$ piccolo** (vicino a 0): modalità debole, poca propagazione

Il **rango effettivo** della matrice è il numero di autovalori significativi ($\lambda_k > 0.01$), e misura la **ricchezza informativa** della propagazione.

## Lo Spectral Gap

Il **spectral gap** è definito come:

$$\Delta\lambda = \lambda_1 - \lambda_2$$

Nei grafi connessi, un gap grande indica propagazione efficiente. Nella nostra rete:

$$\lambda_1 = \lambda_2 = 1.0 \implies \Delta\lambda = 0$$

Questo è **matematicamente atteso**: la nostra rete è composta da **componenti disconnesse** (sottografi biologici separati), quindi ci sono più autovalori pari a 1. Questo non è un errore — riflette la struttura biologica reale di E. coli.

## Confronto Gold Standard vs Edge Prediction

| Metrica | H gold standard (80%) | H edge prediction (100%) |
|---------|----------------------|-------------------------|
| Spectral gap | 0.0000 | 0.0000 |
| Rank effettivo | 434 | **447** |
| $\lambda_1$ | 1.0000 | 1.0000 |
| $\lambda_{\min}$ | −0.3623 | −0.4042 |

**Interpretazione**: H completa ha **13 componenti connesse in più** rispetto a H parziale. Questo significa che la propagazione di $\Theta$ calcolata da H completa raggiunge più sottoreti biologiche, producendo embedding più informativi. Questo risultato spiega **parzialmente** il vantaggio dell'edge prediction rispetto all'approccio gold standard standard.

Dalla figura si osserva inoltre che lo spettro di $\Theta$ decade più rapidamente per H edge prediction (curva arancione) rispetto a H gold standard (curva blu). Questo indica che in H completa la propagazione è più **distribuita** tra molte modalità, mentre in H parziale è più **concentrata** su poche modalità dominanti.

## Confronto Random vs Stratified Masking — topologia dei nodi mascherati

Un risultato sorprendente emerge dal confronto delle proprietà topologiche dei nodi mascherati nei due approcci:

| Metrica | Random test | Stratified test |
|---------|------------|-----------------|
| N° archi mascherati | 803 | 760 |
| Grado medio | **3.91** | **3.92** (+0.3%) |
| Entropia media [bit] | **1.702** | **1.722** (+1.2%) |

Le distribuzioni di grado ed entropia sono **praticamente identiche**. Questo dimostra che la differenza di performance tra random (AUPR = 0.227) e stratified (AUPR = 0.247) **non è dovuta alla topologia dei nodi mascherati**, ma è di natura **metodologica**: stratified garantisce che ogni TF sia rappresentato proporzionalmente nel training set, permettendo alla rete di imparare da tutti i TF invece che solo da quelli sovra-campionati casualmente.

\newpage

# Analisi 3 — Confronto degli Embeddings

## Motivazione

Avendo osservato che l'hard negative sampling crolla nelle performance (AUPR = 0.068 vs 0.247), vogliamo capire **come questo approccio modifica la rappresentazione interna dei geni**. Gli embeddings sono i vettori a 128 dimensioni che il modello costruisce per ogni gene dopo il training:

$$\text{emb}_i \in \mathbb{R}^{128}$$

## Metodologia

Per confrontare gli embeddings dei due approcci (stratified vs hard stratified), abbiamo:

1. Riallenato entrambi i modelli salvando i checkpoint (`--save_checkpoints`)
2. Per ogni gene $i$, calcolato la **distanza euclidea**:

$$\text{diff}(i) = \|\text{emb}^{\text{strat}}_i - \text{emb}^{\text{hard}}_i\|_2 = \sqrt{\sum_{k=1}^{128}\left(\text{emb}^{\text{strat}}_{ik} - \text{emb}^{\text{hard}}_{ik}\right)^2}$$

3. Calcolato la **similarità coseno**:

$$\cos(i) = \frac{\text{emb}^{\text{strat}}_i \cdot \text{emb}^{\text{hard}}_i}{\|\text{emb}^{\text{strat}}_i\| \cdot \|\text{emb}^{\text{hard}}_i\|}$$

4. Correlato $\text{diff}(i)$ con grado $d(v_i)$ ed entropia $\mathcal{H}(v_i)$

## Risultati

![Confronto degli embeddings tra approccio stratified e hard stratified. A sinistra: distribuzione delle differenze euclidee. Al centro: grado vs differenza embedding. A destra: entropia vs differenza embedding.](./output/topology_analysis/embedding_comparison.png)

**Statistiche globali**:

| Metrica | Valore |
|---------|--------|
| Differenza media | 8.86 |
| Similarità coseno media | 0.296 |

**Top 10 geni con embedding più diversi**:

| Gene | Differenza | Cos. sim. | Grado |
|------|-----------|-----------|-------|
| G810 | 28.29 | 0.311 | basso |
| G145 | 23.92 | 0.297 | basso |
| G1163 | 20.74 | 0.162 | — |
| G170 | 20.74 | 0.162 | — |
| G115 | 19.81 | 0.309 | 11 |
| G119 | 19.46 | 0.326 | 12 |

## La scoperta sorprendente

Osservando i grafici "Grado vs Differenza" e "Entropia vs Differenza", emerge un risultato **inatteso**: la differenza di embedding **non è correlata** con grado o entropia. Anche geni con grado basso ($d = 1, 2$) mostrano differenze molto elevate.

**Interpretazione**: I negativi hard non confondono preferenzialmente i geni hub (alto grado, alta entropia), ma perturbano **tutti i geni in modo relativamente uniforme**. Il problema non è locale (specifico di certi nodi) ma **sistemico**: la distribuzione dei negativi difficili rende il problema di ottimizzazione instabile sin dall'inizio del training, impedendo la convergenza a una rappresentazione utile per qualsiasi gene.

Questa osservazione supporta l'idea che la soluzione all'hard negative non sia una modifica topologica, ma un approccio di **curriculum learning** [@bengio2009curriculum]: iniziare il training con negativi facili (random) e aumentare gradualmente la difficoltà man mano che il modello converge.

\newpage

# Analisi 4 — Feature Topologiche come Input del Modello

## Motivazione

La terza domanda del tutor è: se grado ed entropia descrivono la difficoltà di classificazione di un gene, il modello migliora se queste informazioni gli vengono fornite esplicitamente come input?

Attualmente ogni gene è descritto da $X \in \mathbb{R}^{n \times 805}$ (livelli di espressione in 805 esperimenti). L'idea è di aggiungere le metriche topologiche:

$$X_{\text{new}} = [X \mid d_{\text{norm}} \mid \mathcal{H}_{\text{norm}}] \in \mathbb{R}^{n \times 807}$$

dove $d_{\text{norm}} = d(v) / \max d$ e $\mathcal{H}_{\text{norm}} = \mathcal{H}(v) / \max \mathcal{H}$ sono versioni normalizzate nell'intervallo $[0, 1]$.

La normalizzazione è necessaria per rendere le nuove feature compatibili con le feature di espressione (già in scala Z-score con media 0 e deviazione standard 1).

## Implementazione

La modifica è stata implementata come flag opzionale `--use_topo_features` in `main.py`, disattivato per default:

- **Senza flag**: $X \in \mathbb{R}^{n \times 805}$, comportamento identico agli esperimenti precedenti
- **Con flag**: $X_{\text{new}} \in \mathbb{R}^{n \times 807}$, nuovi esperimenti in cartelle separate

Questo garantisce la piena riproducibilità degli esperimenti precedenti.

## Risultati

| Esperimento | Feature | AUPR | $\Delta$ AUPR |
|-------------|---------|------|---------------|
| edge_hgnn_stratified | solo espressione | 0.2467 | — |
| edge_hgnn_stratified_topo | + grado + entropia | 0.2349 | −4.8% |
| hgnn_edge_hard_stratified | solo espressione | 0.0677 | — |
| hgnn_edge_hard_stratified_topo | + grado + entropia | 0.0619 | −8.6% |

## Interpretazione

Le feature topologiche **non migliorano** le performance — anzi, le peggiorano leggermente. Questo risultato, apparentemente negativo, è in realtà **scientificamente informativo**:

La matrice $\Theta$ propaga già implicitamente la struttura topologica negli embeddings. Formalmente, al primo layer della HGNN:

$$\text{emb}^{(1)}_i = \sigma\left(\sum_j \Theta[i][j] \cdot X_j \cdot W\right)$$

Il termine $\sum_j \Theta[i][j] \cdot X_j$ è una **media pesata delle feature dei vicini**, dove i pesi $\Theta[i][j]$ dipendono direttamente dal grado e dalla struttura di $H$. Quindi il modello riceve già informazioni topologiche attraverso $\Theta$, rendendo ridondante l'aggiunta esplicita di grado ed entropia in $X$.

Inoltre, con 805 feature di espressione e solo 2 feature topologiche, la rete $W$ ha difficoltà a "trovare" il segnale utile nelle 2 colonne aggiuntive tra le 807 totali.

\newpage

# Risultati Complessivi

## Confronto di tutti gli esperimenti

![Confronto AUPR di tutti gli esperimenti rispetto alle baseline DREAM5. La linea tratteggiata rossa indica la baseline random (AUPR = 0.0144, proporzione 1:68 come DREAM5).](./output/comparison.png)

**Tabella riepilogativa completa** (ordinata per AUPR decrescente):

| Esperimento | Approccio | AUPR | Miglioramento |
|-------------|-----------|------|---------------|
| DREAM5_Community | Ensemble (unsupervised) | 0.327 | 22.7x |
| DREAM5_Best_Regression | Regression | 0.313 | 21.7x |
| DREAM5_TIGRESS | Regression | 0.301 | 20.9x |
| DREAM5_GENIE3 | Random Forest | 0.291 | 20.2x |
| DREAM5_CLR | Mutual Info | 0.255 | 17.7x |
| edge_hgnn_stratified | HGNN edge pred. | **0.247** | **17.1x** |
| edge_hgnn_stratified_topo | HGNN + topo feat. | 0.235 | 16.3x |
| edge_hgnn_random | HGNN edge pred. | 0.227 | 15.8x |
| gcn_rosario_k10 | GCN statistico | 0.203 | 14.1x |
| hgnn_rosario_k10 | HGNN statistico | 0.184 | 12.8x |
| gcn_baseline | GCN gold std. | 0.151 | 10.5x |
| edge_gcn_stratified | GCN edge pred. | 0.151 | 10.5x |
| edge_gcn_random | GCN edge pred. | 0.135 | 9.4x |
| hgnn_edge_hard_stratified | HGNN hard neg. | 0.068 | 4.7x |
| hgnn_edge_hard_strat_topo | HGNN hard + topo | 0.062 | 4.3x |
| hgnn_edge_hard_random | HGNN hard neg. | 0.054 | 3.7x |
| gcn_edge_hard_random | GCN hard neg. | 0.030 | 2.1x |
| gcn_edge_hard_stratified | GCN hard neg. | 0.027 | 1.9x |
| DREAM5_Random | Baseline random | 0.014 | 1.0x |

## Nota metodologica sul confronto con DREAM5

Tutti i risultati utilizzano la **proporzione 1:68** tra positivi e negativi nel test set, uguale a quella del gold standard DREAM5 originale. Questo garantisce una baseline AUPR di circa 1.44-1.45%, confrontabile con quella di DREAM5.

Il confronto con i metodi DREAM5 è tuttavia **indicativo**: i metodi DREAM5 sono completamente unsupervised e valutati sull'intero gold standard, mentre i nostri metodi supervisionati utilizzano l'80% del gold standard per il training e il 20% per il test.

\newpage

# Conclusioni

## Sintesi delle scoperte

Le analisi topologiche condotte hanno prodotto quattro risultati principali:

**1. Bottleneck strutturale nei TF**

Il TF G120 regola il 26.3% di tutti i geni target (432 su 1643). Questa asimmetria estrema nella distribuzione dei gradi dei TF crea un bottleneck nella propagazione di $\Theta$: il modello tende ad imparare prevalentemente dalla struttura di regolazione di questo singolo TF. Una possibile estensione è il **pesaggio inversamente proporzionale al grado** degli iperarchi nella matrice $W$ di $\Theta$.

**2. H completa spiega il vantaggio dell'edge prediction**

L'analisi spettrale mostra che H costruita con il 100% degli archi ha un rank effettivo superiore (447 vs 434 componenti connesse). Questo produce autovalori di $\Theta$ più distribuiti e una propagazione dell'informazione più ricca, spiegando il miglioramento di AUPR da 0.068 (approccio gold standard) a 0.247 (edge prediction).

**3. Il vantaggio di stratified è metodologico, non topologico**

Il confronto delle proprietà dei nodi mascherati (grado medio 3.91 vs 3.92, differenza +0.3%) dimostra che stratified e random mascherano geni con proprietà topologiche quasi identiche. Il vantaggio di stratified (AUPR 0.247 vs 0.227) deriva dal **bilanciamento per TF**: ogni TF è rappresentato proporzionalmente nel training set.

**4. Hard negative è un problema sistemico**

La differenza degli embeddings tra stratified e hard stratified non è correlata con grado o entropia: anche geni con grado basso mostrano differenze elevate. I negativi hard perturbano uniformemente tutti i geni, rendendo il problema di ottimizzazione instabile fin dall'inizio. Una soluzione naturale è il **curriculum learning** [@bengio2009curriculum].

**5. Θ codifica già la topologia**

Le feature topologiche esplicite (grado ed entropia) non migliorano le performance (−4.8% e −8.6%). Questo conferma che $\Theta$ codifica già implicitamente la struttura topologica negli embeddings attraverso la convoluzione ipergrafo.

## Sviluppi futuri

1. **Curriculum learning per hard negative**: iniziare con negativi facili e aumentare gradualmente la difficoltà
2. **Peso inversamente proporzionale per TF dominanti**: ridurre l'influenza di G120 nella propagazione
3. **Feature topologiche a livello di arco**: invece di aggiungere grado al nodo, aggiungere proprietà degli archi (es. peso dell'iperarco nel decoder)
4. **Validazione su Network 2 e 3**: verificare la generalizzazione dei risultati su altri organismi
5. **Integrazione di dati genomici aggiuntivi**: sequenze, struttura cromatinica, ChIP-seq per superare i limiti dell'expression data

\newpage

# Riferimenti

[@feng2019hypergraph]: Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). **Hypergraph Neural Networks**. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 3558–3565.

[@kipf2017semi]: Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *International Conference on Learning Representations (ICLR)*.

[@shannon1948mathematical]: Shannon, C. E. (1948). **A Mathematical Theory of Communication**. *Bell System Technical Journal*, 27(3), 379–423.

[@marbach2012wisdom]: Marbach, D., Costello, J. C., Küffner, R., et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.

[@bengio2009curriculum]: Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). **Curriculum Learning**. *Proceedings of the 26th Annual International Conference on Machine Learning (ICML)*, 41–48.

[@milo2002network]: Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., & Alon, U. (2002). **Network motifs: simple building blocks of complex networks**. *Science*, 298(5594), 824–827.

