---
title: "Approccio alla Rosario — Costruzione della Struttura dalla Similarità Genica"
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

# Il Progetto Originale di Rosario

## Contesto e obiettivo

Il progetto di Rosario — disponibile pubblicamente su GitHub al repository `FLaTNNBio/GRN-Hypergraph-Classifier` — si propone di costruire un framework per l'inferenza di reti di regolazione genica usando ipergrafi come struttura dati. Il suo obiettivo di partenza è identico al nostro: capire quali fattori di trascrizione controllano quali geni in *Escherichia coli*, usando il dataset DREAM5.

La differenza fondamentale rispetto al nostro approccio sta nel punto di partenza: Rosario non usa il gold standard biologico per costruire la struttura del grafo. Costruisce invece una rete di connessioni basandosi esclusivamente sulla **similarità statistica** tra i profili di espressione genica — la misura di quanto due geni si comportano in modo simile tra i diversi esperimenti di microarray.

## L'architettura originale

Il framework di Rosario è organizzato in sette fasi. Nelle prime due prepara i dati calcolando per ogni gene delle statistiche aggiuntive: varianza (quanto cambia l'attività del gene tra gli esperimenti), skewness (se il gene tende a essere sempre attivo o sempre inattivo) e autocorrelazione (se l'attività è stabile nel tempo). Queste feature statistiche vengono aggiunte ai profili di espressione come informazioni supplementari per ogni nodo.

Nella terza fase costruisce la struttura del grafo usando K nearest neighbors (K=5): per ogni gene, trova i 5 geni con il profilo di espressione più simile e li connette con un arco. La similarità viene calcolata combinando correlazione di Pearson e similarità coseno.

Il modello neurale scelto è una **ResidualGAT** (Graph Attention Network con connessioni residue), che a differenza di una semplice GCN assegna pesi diversi ai vicini durante l'aggregazione — alcuni vicini vengono considerati più informativi di altri tramite un meccanismo di attenzione. Il decoder è un MLP a tre livelli che prende gli embedding di una coppia (TF, gene) e predice la probabilità dell'interazione.

Per il training usa una **focal loss** personalizzata, che penalizza maggiormente gli esempi difficili rispetto alla Binary Cross-Entropy standard, e include un meccanismo di hard negative sampling per esporre il modello agli esempi più insidiosi.

## I limiti riconosciuti dal progetto stesso

Il README del repository di Rosario identifica esplicitamente quattro problemi aperti nel suo approccio:

Il primo e più importante è strutturale: il framework produce in realtà grafi normali, dagli iperarchi vengono poi estratti *deterministicamente* — non appresi. Questo significa che la struttura a iperarchi non è un vero risultato del modello ma una post-elaborazione della struttura a grafo.

Il secondo problema è computazionale: l'hard negative sampling, che per ogni TF calcola i punteggi di similarità su tutti i geni, è troppo lento per dataset grandi.

Il terzo riguarda la capacità espressiva del decoder: un MLP semplice potrebbe non catturare interazioni complesse tra TF e geni nei loro embedding.

Il quarto riguarda la valutazione: le metriche misurano solo la qualità delle coppie TF-gene, non la qualità degli iperarchi come strutture collettive.

---

\newpage

# Motivazione del Confronto

## Perché replicare l'approccio Rosario

Quando il nostro progetto aveva già prodotto i primi risultati con l'approccio gold standard, il tutor ha posto una domanda precisa: quanto conta la qualità della struttura di partenza? I nostri modelli usavano il gold standard biologico per costruire H — una scelta metodologicamente corretta ma che produce una struttura sparsa. Rosario usava la similarità statistica — una scelta più libera biologicamente ma che produce strutture più dense.

Replicare l'approccio di Rosario nel nostro framework ci permetteva di rispondere a una domanda scientifica concreta con una **matrice 2×2** di esperimenti:

|  | Struttura biologica (gold standard) | Struttura statistica (similarità) |
|---|---|---|
| **GCN** | exp01–exp07 baseline | gcn\_rosario\_k10 |
| **HGNN** | gcn\_baseline | hgnn\_rosario\_k10 |

Le due domande a cui risponde questa matrice sono indipendenti. La prima: struttura biologica o statistica — quale produce risultati migliori? La seconda: grafo o ipergrafo — quale modello è più adatto al problema?

## La domanda del data leakage

Prima ancora di implementare gli esperimenti, durante la discussione con il tutor è emersa una preoccupazione metodologica importante. La similarità statistica tra profili di espressione — la correlazione di Pearson e la similarità coseno — misura quanto due geni si comportano allo stesso modo nei diversi esperimenti. Ma c'è un principio biologico fondamentale che collega direttamente questa misura al gold standard:

> Geni regolati dallo stesso TF tendono a co-esprimersi.

Questo significa che la matrice di similarità contiene, in modo implicito, informazioni sulle interazioni biologiche che il gold standard codifica in modo esplicito. Quando si costruisce H dalla similarità e poi si valuta il modello sul gold standard, esiste il rischio di un **data leakage implicito**: la struttura H è stata costruita da dati già correlati con la risposta che si vuole predire. Il modello non sta imparando a predire interazioni nuove — sta riconoscendo pattern già codificati nella struttura di partenza.

Questa osservazione non invalida i risultati, ma li contestualizza: performance più alte non indicano necessariamente un modello più capace, ma potrebbero riflettere una struttura di partenza più informativa per ragioni biologiche.

---

\newpage

# Implementazione nel Nostro Framework

## La scelta metodologica: K=10

Rispetto al K=5 del progetto originale di Rosario, si è scelto K=10 come valore del nearest neighbor. La motivazione è che K=5 produce una struttura molto sparsa — ogni gene è connesso a soli 5 vicini — che non differisce molto dall'approccio gold standard in termini di densità. K=10 crea una struttura più densa e permette una propagazione più ricca attraverso Theta, rendendo il confronto più informativo.

## Calcolo della matrice di similarità (`step2.py` con `structure='statistical'`)

Il cuore dell'approccio è la funzione di calcolo della similarità, implementata in `step2.py` quando viene invocata con il parametro `structure='statistical'`. Nella versione attuale del progetto (src\_XIV) tutta la logica di costruzione della struttura — gold standard, statistica ed edge prediction — è unificata in un unico file `step2.py`, selezionabile tramite il flag `--analysis` di `main.py`. Per eseguire gli esperimenti alla Rosario si usa:

```bash
# GCN con struttura statistica
python main.py --network_path /percorso/Network1 \
               --analysis gcn_statistical \
               --exp_name gcn_rosario_k10

# HGNN con struttura statistica
python main.py --network_path /percorso/Network1 \
               --analysis hgnn_statistical \
               --exp_name hgnn_rosario_k10
```

Per ogni coppia di geni (i, j), si calcolano due misure:

**Correlazione di Pearson:** misura quanto i due profili di espressione si muovono nella stessa direzione. Un valore di +1 significa che i due geni sono sempre attivi e inattivi nelle stesse condizioni sperimentali; un valore di -1 indica il contrario.

$$\text{Pearson}(i,j) = \frac{\sum_k (x_{ik} - \bar{x}_i)(x_{jk} - \bar{x}_j)}{\sqrt{\sum_k (x_{ik} - \bar{x}_i)^2 \cdot \sum_k (x_{jk} - \bar{x}_j)^2}}$$

**Similarità coseno:** misura l'angolo tra i due vettori di espressione nello spazio a 805 dimensioni. Complementa la correlazione di Pearson perché è meno sensibile alle differenze di scala assoluta.

Le due misure vengono normalizzate in $[0,1]$ e combinate con pesi 0.6 (Pearson) e 0.4 (coseno), replicando la scelta di Rosario. Il risultato è una matrice di similarità combinata di dimensione $1643 \times 1643$.

## Costruzione della struttura GCN (`--analysis gcn_statistical`)

In modalità grafo, per ogni gene si selezionano i K=10 geni più simili e si creano archi bidirezionali tra loro. La matrice di adiacenza risultante A è quadrata (1643×1643), simmetrica, e contiene circa 27.000 connessioni — densità dell'1.0%. Questa è quasi dieci volte più densa della matrice A del gold standard (circa 8.000 connessioni, densità 0.3%).

## Costruzione della struttura HGNN (`--analysis hgnn_statistical`)

In modalità ipergrafo, per ogni gene i si crea un iperarco che include il gene stesso e i suoi K=10 nearest neighbors. La matrice di incidenza H risultante è quadrata (1643×1643) — diversamente da H del gold standard che è rettangolare (1643×195) — con circa 18.000 connessioni, densità del 0.67%.

Questa è la differenza strutturale più rilevante rispetto all'approccio gold standard: nell'approccio statistico ogni gene diventa il centro del proprio iperarco, mentre nell'approccio gold standard ogni iperarco corrisponde a un TF biologico. Il numero di iperarchi passa da 195 (i TF del gold standard) a 1643 (tutti i geni).

## Confronto della sparsità tra i quattro approcci

| Approccio | Forma di H | Connessioni | Densità |
|---|---|---|---|
| HGNN Gold Standard | 1643 × 195 | 3.209 | 1.00% |
| GCN Gold Standard | 1643 × 1643 | 8.033 | 0.30% |
| **HGNN Rosario (K=10)** | **1643 × 1643** | **18.000** | **0.67%** |
| **GCN Rosario (K=10)** | **1643 × 1643** | **27.000** | **1.00%** |

L'approccio statistico produce strutture significativamente più dense. Questa densità si traduce in una matrice Theta con rank effettivo più alto e quindi embedding più informativi — ma come vedremo, a un costo metodologico preciso.

---

\newpage

# Risultati

## GCN Rosario (`gcn_rosario_k10`)

Il modello GCN con struttura costruita dalla similarità statistica ottiene:

| Metrica | Valore |
|---|---|
| **AUPR** | **0.2034** |
| AUROC | 0.8658 |
| Precision | 0.0747 |
| Recall | 0.7011 |
| Best epoch | 99 |

Con AUPR = 0.203, la GCN Rosario supera significativamente la GCN con gold standard (0.151, +35%) e la HGNN con gold standard (0.074, +174%). Il modello non supera però la HGNN con edge prediction (0.247).

## HGNN Rosario (`hgnn_rosario_k10`)

Il modello HGNN con struttura costruita dalla similarità statistica ottiene:

| Metrica | Valore |
|---|---|
| **AUPR** | **0.1842** |
| AUROC | 0.8604 |
| Precision | 0.0653 |
| Recall | 0.7335 |
| Best epoch | 100 |

Con AUPR = 0.184, l'HGNN Rosario supera il gold standard HGNN (0.074, +148%) ma rimane sotto la GCN Rosario e sotto la HGNN con edge prediction.

## Quadro comparativo completo

| Modello | Struttura | AUPR | AUROC |
|---|---|---|---|
| HGNN Edge Stratified (nostro) | Gold standard completo (masking) | **0.247** | 0.925 |
| **GCN Rosario (K=10)** | **Similarità statistica** | **0.203** | **0.866** |
| **HGNN Rosario (K=10)** | **Similarità statistica** | **0.184** | **0.860** |
| GCN Gold Standard | Gold standard (80%) | 0.151 | 0.802 |
| HGNN Gold Standard (best) | Gold standard (80%) | 0.074 | 0.660 |
| *AUPR casuale* | — | *0.014* | — |
| DREAM5 CLR | — | 0.255 | — |
| DREAM5 GENIE3 | — | 0.291 | — |

## Lettura della matrice 2×2

```
                    Gold standard      Similarità statistica
                    (biologico)        (alla Rosario)

GCN             |   0.151           |   0.203  (+35%)
HGNN            |   0.074           |   0.184  (+148%)
```

Due conclusioni emergono dalla matrice. Prima: la struttura statistica supera quella biologica ridotta in entrambe le configurazioni. Seconda: la GCN supera la HGNN in entrambe le strutture. La combinazione peggiore è HGNN + gold standard (0.074), la migliore tra le quattro celle è GCN + statistico (0.203).

---

\newpage

# Il Problema del Data Leakage

## Perché i risultati sono "gonfiati"

I risultati dell'approccio Rosario sono più alti di quelli del gold standard ridotto, ma questo non significa che il modello sia più capace. La ragione è biologica, non algoritmica.

In *E. coli*, come in tutti gli organismi, esiste una legge di co-espressione: geni che condividono un regolatore comune tendono ad avere profili di espressione simili. Se TF_A regola Gene_1, Gene_2 e Gene_3, nella maggior parte degli esperimenti questi tre geni tendono ad attivarsi e disattivarsi insieme, perché hanno lo stesso interruttore. Di conseguenza, la correlazione di Pearson tra Gene_1, Gene_2 e Gene_3 è alta.

Quando si costruisce H dalla correlazione di Pearson e poi si valuta il modello sul gold standard — che contiene esattamente quelle interazioni TF→gene — si crea un circolo: la struttura H è costruita da dati che già codificano indirettamente le risposte biologiche che si vogliono predire. Il modello non impara a scoprire interazioni nuove — impara a riconoscere pattern che erano già presenti nella struttura di partenza.

Il data leakage non è intenzionale né disonesto: è una conseguenza inevitabile della biologia. Ma rende i risultati non comparabili direttamente con approcci che non sfruttano questa correlazione. Una AUPR di 0.203 con struttura statistica non equivale a una AUPR di 0.203 con struttura biologica.

## Confronto con il metodo edge prediction

L'approccio edge prediction risolve questo problema alla radice: usa il gold standard completo per costruire H — quindi non perde nessuna connessione biologica — ma maschera le interazioni di test in modo che il modello non le "veda" durante il training. Il risultato (AUPR = 0.247) è metodologicamente pulito e strutturalmente ricco, combinando i vantaggi di entrambi gli approcci senza il rischio di leakage.

---

\newpage

# Differenze Rispetto al Progetto Originale di Rosario

Rispetto all'implementazione originale, il nostro adattamento differisce in quattro punti principali.

Il primo riguarda il **modello neurale**: Rosario usa una ResidualGAT con meccanismo di attenzione; noi usiamo una GCN standard e una HGNN standard. Questa scelta è deliberata: l'obiettivo non era replicare esattamente il modello di Rosario ma isolare il contributo della struttura (statistica vs biologica) mantenendo il modello costante.

Il secondo riguarda il **valore di K**: Rosario usa K=5, noi K=10. Come spiegato, K=10 produce una struttura più densa che rende il confronto più significativo.

Il terzo riguarda la **vera gestione degli ipergrafi**: Rosario ammette nel suo README che il suo framework produce in realtà grafi normali dagli iperarchi vengono estratti deterministicamente. Nel nostro caso, l'analisi `hgnn_statistical` (invocata con `--analysis hgnn_statistical`) costruisce una vera matrice di incidenza H e calcola la vera matrice di propagazione Theta secondo la formula di Feng et al. (2019), producendo un ipergrafo autentico.

Il quarto riguarda la **funzione di loss**: Rosario usa una focal loss personalizzata per gestire lo sbilanciamento; noi usiamo Binary Cross-Entropy con class weighting, che è più standard e permette confronti più diretti con gli altri esperimenti del progetto.

---

\newpage

# Conclusioni

L'approccio alla Rosario ha svolto nel progetto un ruolo di **esperimento di controllo**: ci ha permesso di quantificare quanto vale la struttura statistica rispetto a quella biologica, e di capire perché risultati più alti non sempre significano modelli migliori.

I risultati mostrano che la struttura statistica produce AUPR più alte della struttura gold standard ridotta in entrambe le configurazioni GCN e HGNN. Tuttavia, il confronto corretto non è tra gold standard ridotto e statistico, ma tra statistico e edge prediction: e qui la struttura biologica completa (AUPR = 0.247) supera quella statistica (AUPR = 0.203), senza portare il rischio di data leakage.

La conclusione metodologica principale è quindi chiara: la struttura biologica, quando viene usata correttamente senza riduzione, è superiore a quella statistica sia in termini di performance che di correttezza metodologica. L'approccio di Rosario è un punto di riferimento utile, ma non è il punto di arrivo.

---

\newpage

# Riferimenti

- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR*.
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.
- FLaTNNBio (2025). **GRN-Hypergraph-Classifier**. GitHub. [https://github.com/FLaTNNBio/GRN-Hypergraph-Classifier](https://github.com/FLaTNNBio/GRN-Hypergraph-Classifier)
- Veličković, P. et al. (2018). **Graph Attention Networks**. *ICLR*.
