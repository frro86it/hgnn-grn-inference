---
title: "Approccio Gold Standard — Costruzione dell'Ipergrafo da Dati Biologici"
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

# Introduzione e Motivazione

## Il punto di partenza

Ogni modello basato su ipergrafi ha bisogno di una struttura di partenza: la matrice di incidenza H, che definisce quali geni appartengono a quali iperarchi. Nel nostro caso, ogni iperarco corrisponde a un fattore di trascrizione, e H[i][j] = 1 se il TF j regola il gene i. La domanda fondamentale che si pone all'inizio del progetto è: da dove viene questa struttura?

La risposta più naturale, e biologicamente più onesta, è usare direttamente le interazioni certificate dal gold standard di DREAM5. Il gold standard contiene 4.012 interazioni TF→gene vere per *Escherichia coli*, verificate sperimentalmente. Usarle per costruire H significa basarsi su conoscenza biologica reale, non su proxy statistici.

Questo approccio — che chiamiamo **gold standard** — è stato il primo implementato nel progetto, e rappresenta il punto di partenza concettuale di tutto il lavoro successivo.

## Il problema dello sbilanciamento

Prima ancora di costruire H, è necessario fare i conti con la struttura del dataset. Il gold standard contiene:

- **4.012 interazioni vere** (Label = 1), pari all'1.4% del totale
- **274.380 interazioni false** (Label = 0), pari al 98.6% del totale
- **Rapporto di sbilanciamento: 1:68** — per ogni interazione vera esistono 68 false

In presenza di uno squilibrio così estremo, la metrica standard di accuratezza è inutile: un modello che predice tutto come "falso" raggiungerebbe il 98.6% di accuratezza senza imparare nulla. Per questo motivo, la metrica di riferimento adottata in tutto il progetto è l'**AUPR** (Area Under the Precision-Recall Curve), che penalizza severamente i modelli con precision bassa — ovvero quelli che fanno troppi falsi positivi. L'AUPR di un predittore puramente casuale su questo dataset vale 0.014.

---

\newpage

# Costruzione della Matrice H

## Il principio: usare l'80% del gold standard

L'approccio gold standard divide le 4.012 interazioni vere in due insiemi distinti prima di iniziare qualsiasi training:

- **80% (circa 3.209 interazioni)** viene usato per costruire la matrice H e addestrare il modello
- **20% (circa 803 interazioni)** viene riservato per la valutazione finale e non tocca mai il training

Le interazioni false non vengono divise: tutte e 274.380 rimangono nel test set, affiancate alle 803 interazioni vere riservate.

La divisione è casuale, con seed fisso per garantire la riproducibilità. Questo significa che non c'è nessun meccanismo di bilanciamento per TF: alcuni fattori di trascrizione potrebbero perdere più del 20% dei loro archi nel training, altri meno, a seconda del campionamento casuale.

## La matrice di incidenza H

La matrice H viene costruita esclusivamente dalle 3.209 interazioni del training set. Per ogni coppia (TF j, gene i) presente nel training, si pone H[i][j] = 1. La forma finale di H è:

$$H \in \{0, 1\}^{1643 \times 195}$$

dove 1.643 è il numero di geni e 195 è il numero di TF. Con 3.209 connessioni su 1.643 × 195 = 320.385 elementi possibili, la densità di H è circa **1.00%** — una matrice estremamente sparsa.

Questa sparsità è il fattore critico dell'approccio gold standard. Theta, la matrice di propagazione dell'HGNN, viene calcolata da H:

$$\Theta = D_v^{-1/2} \cdot H \cdot W \cdot D_e^{-1} \cdot H^T \cdot D_v^{-1/2}$$

Una H con solo 3.209 connessioni produce un Theta con **rank effettivo basso**: molti geni non regolati da nessun TF nel training set rimangono isolati nella propagazione, e il modello non riesce a costruire embedding significativi per loro.

## Normalizzazione delle feature

Le 805 feature di ogni nodo — corrispondenti ai 805 esperimenti di espressione genica — vengono normalizzate con **Z-score** per ciascun gene:

$$x_{norm}[i, k] = \frac{x[i, k] - \mu_k}{\sigma_k}$$

La normalizzazione è indispensabile perché i profili di espressione possono avere scale molto diverse tra esperimenti. Senza normalizzazione, le feature con varianza alta dominerebbero il calcolo della loss.

---

\newpage

# Architettura del Modello

## HGNN: Hypergraph Neural Network

Il modello HGNN è implementato seguendo il paper di Feng et al. (AAAI 2019). L'architettura è composta da due parti:

**Encoder HGNN (2 layer):** prende in input X normalizzato (1643 × 805) e produce embedding a 128 dimensioni per ogni gene.

$$\text{emb}^{(1)} = \text{ReLU}\left(\text{BN}\left(\Theta \cdot X \cdot W^{(0)}\right)\right)$$
$$\text{emb}^{(2)} = \text{ReLU}\left(\text{BN}\left(\Theta \cdot \text{emb}^{(1)} \cdot W^{(1)}\right)\right)$$

A ogni layer viene applicato dropout e BatchNorm. Theta rimane fissa durante tutto il training.

**Decoder MLP:** prende gli embedding di una coppia (TF, gene) e predice la probabilità che l'interazione esista:

$$P(\text{TF}_j \to \text{gene}_i) = \sigma\left(\text{MLP}\left([\text{emb}_j \| \text{emb}_i]\right)\right)$$

Il decoder è un percettrone a due layer con ReLU e sigmoid finale.

## GCN: Graph Convolutional Network (baseline)

In parallelo alla HGNN, si allena anche una GCN come baseline di confronto. La GCN lavora su una matrice di adiacenza quadrata A (1643 × 1643) costruita dalle stesse interazioni del gold standard, ma rappresentata come grafo classico anziché come ipergrafo. L'architettura è identica a quella dell'HGNN per garantire un confronto equo: stessi due layer, stessa dimensione degli embedding, stesso decoder MLP.

## Training e early stopping

Il modello viene allenato minimizzando la **Binary Cross-Entropy** con class weighting per compensare lo sbilanciamento. L'ottimizzatore è Adam. L'early stopping monitora l'AUPR sul validation set: se non migliora per un numero di epoche pari alla patience, il training si interrompe e si ripristina il miglior checkpoint.

---

\newpage

# Gli Esperimenti di Hyperparameter Tuning

## Motivazione del tuning

La prima versione del modello (exp01\_baseline) usava una configurazione di default: dropout 0.5, learning rate 0.001, patience 10, 100 epoche. I risultati iniziali mostravano segnali di overfitting nella curva di training — l'AUPR sul training set cresceva continuamente mentre quella sul validation si stabilizzava presto. Questo ha motivato una serie di esperimenti sistematici per trovare la configurazione ottimale.

## Esperimento 1 — Baseline (`exp01_baseline`)

Configurazione di riferimento con tutti i parametri di default.

| Parametro | Valore |
|---|---|
| Dropout | 0.5 |
| Learning rate | 0.001 |
| Weight decay | $5 \times 10^{-4}$ |
| Patience | 10 |
| Epoche | 100 |

Risultato: **AUPR = 0.068**, AUROC = 0.664, best\_epoch = 99.

Il best\_epoch = 99 segnala che il modello non ha ancora trovato un plateau stabile alla centesima epoca: l'early stopping non è mai intervenuto e il modello ha girato per tutte le epoche previste. L'AUROC di 0.664 indica che il modello sa ordinare le coppie meglio del caso (0.5), ma la precision è troppo bassa per convertire questo in AUPR alto.

## Esperimento 2 — Dropout aumentato a 0.6 (`exp02_dropout06`)

L'ipotesi era che aumentare il dropout da 0.5 a 0.6 potesse ridurre l'overfitting, forzando il modello a costruire rappresentazioni più robuste.

Risultato: **AUPR = 0.069**, AUROC = 0.665, best\_epoch = 100.

Il miglioramento è marginale (+0.1 punti di AUPR). La struttura sparsa di H limita quanto il modello può imparare indipendentemente dalla regolarizzazione: con pochi archi in H, gli embedding di molti geni rimangono poco informati.

## Esperimento 3 — Dropout aumentato a 0.7 (`exp03_dropout07`)

Spingere il dropout oltre 0.6 dovrebbe aumentare la regolarizzazione ma rischia di eliminare troppa informazione.

Risultato: **AUPR = 0.067**, AUROC = 0.663, best\_epoch = 100.

Come atteso, il dropout 0.7 peggiora leggermente rispetto a 0.6. La conclusione è che il range ottimale di dropout per questo dataset si trova intorno a 0.5–0.6.

## Esperimento 4 — Weight decay aumentato (`exp04_wd1e3`)

Il weight decay penalizza i pesi grandi dell'ottimizzatore, scoraggiando l'overfitting sui dati di training. Si è raddoppiato da $5 \times 10^{-4}$ a $10^{-3}$.

Risultato: **AUPR = 0.060**, AUROC = 0.661, best\_epoch = 98.

Il weight decay più alto peggiora: la regolarizzazione extra impedisce ai pesi di raggiungere i valori necessari per discriminare bene le interazioni scarse. Con solo 3.209 positivi nel training, il modello ha bisogno di libertà sufficiente per memorizzare le strutture biologiche.

## Esperimento 5 — Patience ridotta a 5 (`exp05_patience05`)

Ridurre la patience significa fermare prima il training quando l'AUPR smette di migliorare. L'ipotesi è che il modello raggiunga il suo massimo presto e poi stazioni o peggiori leggermente.

Risultato: **AUPR = 0.074**, AUROC = 0.660, best\_epoch = 98.

Questo è il **miglior risultato** tra tutti gli esperimenti gold standard HGNN. La patience ridotta a 5 suggerisce che effettivamente l'early stopping standard (patience=10) lasciava il modello girare troppo a lungo dopo aver raggiunto il suo massimo, accumulando rumore nel gradiente.

## Esperimento 6 — Learning rate abbassato (`exp06_lr0001`)

Un learning rate più basso (0.0001 invece di 0.001) dovrebbe produrre aggiornamenti più cauti e potenzialmente migliore generalizzazione, a patto di avere abbastanza epoche per convergere.

Risultato: **AUPR = 0.048**, AUROC = 0.656, best\_epoch = 99.

Il learning rate basso peggiora sensibilmente: con 100 epoche il modello non ha abbastanza passi per convergere a una soluzione decente. L'AUROC scende a 0.656, il più basso di tutti gli esperimenti HGNN.

## Esperimento 7 — Configurazione combinata con più epoche (`exp07_best`)

Si combinano dropout 0.6 (miglior dropout) e weight decay più alto con 200 epoche per dare al modello più tempo di convergere.

Risultato: **AUPR = 0.067**, AUROC = 0.661, best\_epoch = 200.

Il best\_epoch = 200 — l'ultima epoca disponibile — indica che il modello non ha ancora trovato un plateau nemmeno con il doppio delle epoche. Più tempo non aiuta: il limite non è computazionale ma strutturale.

## GCN Baseline (`gcn_baseline`)

In parallelo a tutti gli esperimenti HGNN, si allena la GCN con la configurazione base (dropout 0.5, lr 0.001, patience 10, 100 epoche) sulla stessa struttura gold standard.

Risultato: **AUPR = 0.151**, AUROC = 0.802, best\_epoch = 99.

La GCN supera significativamente tutti gli esperimenti HGNN sullo stesso gold standard. La spiegazione principale è strutturale: la GCN opera su una matrice di adiacenza A (1643×1643) che contiene circa 8.000 connessioni gene-gene, il doppio rispetto alle 3.209 di H. Più connessioni significa propagazione più ricca e embedding più informativi.

---

\newpage

# Risultati e Confronto

## Tabella riassuntiva

| Esperimento | Modello | Dropout | LR | Patience | Epoche | AUPR | AUROC |
|---|---|---|---|---|---|---|---|
| exp01\_baseline | HGNN | 0.5 | 0.001 | 10 | 100 | 0.068 | 0.664 |
| exp02\_dropout06 | HGNN | 0.6 | 0.001 | 10 | 100 | 0.069 | 0.665 |
| exp03\_dropout07 | HGNN | 0.7 | 0.001 | 10 | 100 | 0.067 | 0.663 |
| exp04\_wd1e3 | HGNN | 0.5 | 0.001 | 10 | 100 | 0.060 | 0.661 |
| **exp05\_patience05** | **HGNN** | **0.5** | **0.001** | **5** | **100** | **0.074** | **0.660** |
| exp06\_lr0001 | HGNN | 0.5 | 0.0001 | 10 | 100 | 0.048 | 0.656 |
| exp07\_best | HGNN | 0.6 | 0.001 | 15 | 200 | 0.067 | 0.661 |
| **gcn\_baseline** | **GCN** | **0.5** | **0.001** | **10** | **100** | **0.151** | **0.802** |
| *AUPR casuale* | — | — | — | — | — | *0.014* | — |
| *Edge HGNN stratified* | — | — | — | — | — | *0.247* | — |

## Tre osservazioni principali

**Prima osservazione — Il tuning ha effetto marginale sulla HGNN.** Il range di AUPR tra tutti gli esperimenti HGNN è 0.048–0.074, una finestra molto stretta. Cambiare dropout, weight decay, learning rate o patience non produce variazioni significative. Il tetto delle performance è determinato dalla struttura di H, non dai parametri del modello. Questo è il segnale più chiaro che il problema è architetturale, non di ottimizzazione.

**Seconda osservazione — La GCN supera la HGNN sul gold standard.** Con AUPR = 0.151 contro 0.074, la GCN è più del doppio più performante della miglior HGNN in questa configurazione. La ragione sta nella densità: la matrice di adiacenza della GCN contiene il doppio delle connessioni di H, permettendo una propagazione dell'informazione più ricca. L'ipergrafo, paradossalmente, è penalizzato dalla propria struttura a iperarchi: con solo 3.209 connessioni distribuite su 195 colonne, molti geni risultano isolati o connessi a un solo TF.

**Terza osservazione — L'approccio gold standard è il limite inferiore.** Confrontando con l'edge prediction HGNN stratified (AUPR = 0.247), è evidente che ridurre H all'80% degli archi biologici veri è costoso. La scarsità di H si trasmette direttamente alla qualità di Theta e quindi alla qualità degli embedding. Il passaggio all'edge prediction — dove H viene costruita con il 100% delle interazioni e il test avviene tramite masking — porta un miglioramento di oltre tre volte sull'HGNN (+233%).

## Perché i risultati sono più bassi di quanto atteso

In una fase intermedia del progetto, la versione del codice produceva valori di AUPR molto più alti per gli stessi esperimenti (HGNN fino a 0.323, GCN fino a 0.469). Quei risultati erano reali ma corrispondevano a una architettura diversa: la struttura del grafo veniva costruita in modo più denso, o la separazione training/test non era implementata con la stessa rigore. Nella versione attuale del codice (src\_XIV), la costruzione di H è più fedele al principio del gold standard puro: solo l'80% degli archi veri entra in H, e il resto è riservato al test. Questo produce risultati più bassi ma metodologicamente più corretti.

---

\newpage

# Conclusioni e Lezione Appresa

L'approccio gold standard ha svolto un ruolo fondamentale nel progetto: non come configurazione vincente, ma come **esperimento di comprensione**. I suoi risultati contenuti hanno rivelato il meccanismo che governa le performance del modello.

Il problema non è il modello in sé — la HGNN è architetturalmente solida. Il problema è la qualità della struttura che le viene fornita. Quando H contiene solo l'80% delle interazioni biologiche vere, Theta ha rank ridotto, molti geni restano con embedding poco informativi, e il decoder non riesce a discriminare con sufficiente precisione. È una questione di informazione disponibile, non di capacità di apprendimento.

Questa conclusione ha guidato direttamente la scelta successiva: l'edge prediction, che usa il gold standard completo per costruire H e gestisce la separazione training/test tramite masking invece di ridurre la struttura. Il salto di performance da 0.074 a 0.247 conferma che quella era la direzione giusta.

---

\newpage

# Riferimenti

- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR*.
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.
- Latchman, D.S. (1997). **Transcription Factors: An Overview**. *Int. J. Biochem. Cell Biol.*, 29(12), 1305–1312.
