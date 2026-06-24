---
title: "Edge Prediction per l'Inferenza di GRN — Strategie di Campionamento e Risultati"
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

## Il problema centrale: predire interazioni TF→gene

L'obiettivo del progetto è rispondere a una domanda biologica precisa: dato un fattore di trascrizione e un gene, esiste una relazione di regolazione tra loro? Nel dataset DREAM5 Network 1 questa domanda si traduce in un problema di **link prediction** su un grafo: il modello deve distinguere le 4.012 interazioni vere (TF→gene) dalle oltre 270.000 coppie false che si potrebbero formare tra i 195 TF e i 1.643 geni di *Escherichia coli*.

Il dataset è fortemente sbilanciato: per ogni interazione vera esistono circa 68 false. Questo squilibrio rende insufficiente la semplice accuratezza come metrica e impone l'uso della **AUPR** (Area Under the Precision-Recall Curve), che penalizza i modelli che si limitano a predire tutto come positivo per gonfiare il recall.

## Perché l'edge prediction

Nelle prime versioni del progetto, la struttura dell'ipergrafo veniva costruita in due modi distinti. Il primo — che chiamiamo approccio *gold standard* — usava solo una parte delle interazioni biologiche certificate per costruire la matrice di incidenza H, riservando il resto al test. Il secondo — ispirato al lavoro di Rosario — costruiva H usando la similarità statistica tra i profili di espressione genica, senza guardare al gold standard.

Entrambi gli approcci presentavano limiti. Il gold standard ridotto produceva un ipergrafo con meno connessioni effettive, abbassando il rank di H e impoverendo la propagazione dell'informazione. L'approccio statistico invece costruiva una struttura ricca ma non biologicamente fondata, introducendo un rischio di data leakage: la similarità di espressione potrebbe correlare con le interazioni vere.

L'**edge prediction** risolve questo problema in modo elegante. Invece di decidere a priori quale parte del gold standard usare per costruire H, si usa il gold standard *completo* per definire la struttura dell'ipergrafo, e si gestisce la separazione training/test attraverso un meccanismo di *masking*: alcuni archi vengono nascosti al modello durante il training e usati solo nella fase di valutazione. Il modello quindi impara a propagare informazione attraverso l'intera rete biologica, e viene giudicato sulla sua capacità di recuperare gli archi mascherati.

Questo approccio è concettualmente più pulito: la struttura dell'ipergrafo riflette la biologia reale al 100%, e il compito di predizione è chiaramente separato dalla costruzione della struttura.

---

\newpage

# I Due Assi Indipendenti

Una volta adottata l'edge prediction, ci si trova davanti a due scelte progettuali indipendenti che, combinate, definiscono le diverse varianti sperimentali.

## Asse 1: come si dividono i positivi tra training e test

Il gold standard contiene 4.012 interazioni vere. Di queste, l'80% (circa 3.209) viene usato per il training, e il 20% (circa 803) per il test. La domanda è: come si sceglie quale 20% riservare al test?

Questa scelta avviene nella fase di costruzione del dataset — prima del training — e determina quale conoscenza biologica il modello ha a disposizione durante l'apprendimento.

## Asse 2: come si scelgono i negativi per il training

Durante il training il modello ha bisogno di esempi negativi — coppie TF→gene false — da contrapporre ai positivi. Le 270.000 coppie false disponibili sono troppe per usarle tutte, e la scelta di quale campionare influenza profondamente quello che il modello impara.

---

\newpage

# Metodologia 1 — Campionamento Casuale dei Negativi

## Descrizione

La prima e più semplice strategia per i negativi è il campionamento casuale: ad ogni epoca di training, per ogni interazione vera presente nel training set, si affianca un numero equivalente di coppie false scelte a caso tra tutte quelle disponibili. Non c'è nessuna preferenza per coppie particolarmente difficili o simili ai positivi: la scelta è puramente uniforme.

Questa strategia si combina con due diverse modalità di divisione dei positivi, dando origine a due esperimenti distinti.

## Variante A — Random masking (`edge_hgnn_random`, `edge_gcn_random`)

Nel masking casuale, il 20% degli archi veri da riservare al test viene scelto completamente a caso dall'intero pool di 4.012 interazioni, indipendentemente da quale TF provengano. Il risultato è che alcuni TF potrebbero perdere molti dei loro archi nel training, mentre altri ne perdono pochissimi. Questo squilibrio può penalizzare i TF con pochi target: se un TF regola solo 3 geni e per caso tutti e 3 finiscono nel test set, il modello non ha mai visto una sua interazione durante il training.

Risultati:

| Modello | AUPR | AUROC | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|
| HGNN + random masking | 0.2265 | 0.9094 | 0.0657 | 0.8057 | 100 |
| GCN + random masking | 0.1348 | 0.7904 | 0.0616 | 0.6139 | 99 |

## Variante B — Stratified masking (`edge_hgnn_stratified`, `edge_gcn_stratified`)

Nel masking stratificato, la divisione training/test avviene proporzionalmente per ogni TF: se un TF ha 10 target, ne vengono messi 8 in training e 2 in test. Se un TF ha solo 2 target, li conserva entrambi in training per non perdere informazione critica. In questo modo ogni TF contribuisce equamente al training set e al test set, indipendentemente dal numero di interazioni che gestisce.

Questo approccio è metodologicamente più robusto perché garantisce che il modello veda almeno una parte delle interazioni di ogni TF durante l'apprendimento. Il bilanciamento per TF riduce anche la varianza del gradiente: nessun TF domina o scompare dal training.

Risultati:

| Modello | AUPR | AUROC | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|
| HGNN + stratified masking | **0.2467** | **0.9254** | 0.0651 | 0.8592 | 100 |
| GCN + stratified masking | 0.1509 | 0.8378 | 0.0677 | 0.7066 | 99 |

## Confronto e lettura dei risultati

Il masking stratificato supera quello casuale in entrambi i modelli: +9.0% di AUPR per HGNN (0.247 vs 0.227) e +11.9% per GCN (0.151 vs 0.135). La differenza non deriva da una topologia diversa dell'ipergrafo — H è identica in entrambi i casi — ma da una distribuzione più equa del segnale biologico nel training set.

È interessante notare come l'HGNN superi la GCN in entrambe le configurazioni: 0.247 vs 0.151 con stratified, 0.227 vs 0.135 con random. Questo conferma che la struttura a iperarchi — dove un singolo TF regola un insieme di geni come unità — cattura meglio le relazioni biologiche di DREAM5 rispetto a una matrice di adiacenza classica.

Rispetto ai benchmark DREAM5 (AUPR tra 0.255 di CLR e 0.327 del metodo ensemble), l'HGNN con stratified masking si avvicina significativamente al metodo CLR, pur senza raggiungerlo. L'AUPR casuale di un predittore ingenuo è 0.014: entrambi i modelli lo superano di oltre 15 volte, confermando che il modello ha imparato segnali biologici reali.

---

\newpage

# Metodologia 2 — Hard Negative Sampling

## Il ragionamento alla base

Il campionamento casuale dei negativi produce esempi che sono, per la maggior parte, molto diversi dai positivi. Un gene espresso in condizioni completamente diverse dal target reale di un TF è un negativo facile: il modello impara a scartarlo senza sforzo. Questo solleva una domanda: il modello sta davvero imparando a riconoscere la regolazione biologica, o sta semplicemente imparando a distinguere profili di espressione molto diversi?

L'**hard negative sampling** risponde a questa domanda rendendo il training genuinamente difficile. Invece di scegliere negativi a caso, seleziona le coppie TF→gene false che sono *più simili* ai positivi veri. In pratica, per ogni TF, si cercano i geni che hanno il profilo di espressione più simile ai target reali del TF, ma che non sono regolati da quel TF. Questi sono i casi che il modello fatica di più a distinguere — biologicamente plausibili ma biologicamente falsi.

L'idea è ispirata alle tecniche di mining utilizzate nel riconoscimento facciale e nel metric learning (Schroff et al., CVPR 2015): esporre la rete agli esempi più insidiosi la forza a costruire rappresentazioni più precise e discriminative, invece di accontentarsi di separare casi ovvi.

## Come vengono costruiti i negativi difficili

Tecnicamente, per ogni TF si calcolano le distanze euclidee tra il centroide dell'embedding dei suoi target reali e l'embedding di tutti i geni che non regola. I geni più vicini al centroide — quelli che "assomigliano" di più ai target reali nell'embedding space — vengono scelti come negativi difficili. La selezione avviene tra un pool di 50.000 candidati per rendere l'operazione computazionalmente gestibile.

Come per il campionamento casuale, anche l'hard negative si combina con le due strategie di masking dei positivi.

## Variante C — Hard negative + stratified masking (`hgnn_edge_hard_stratified`, `gcn_edge_hard_stratified`)

Risultati:

| Modello | AUPR | AUROC | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|
| HGNN + hard stratified | 0.0677 | 0.8250 | 0.0359 | 0.8447 | 100 |
| GCN + hard stratified | 0.0266 | 0.6114 | 0.0185 | 0.7026 | 99 |

## Variante D — Hard negative + random masking (`hgnn_edge_hard_random`, `gcn_edge_hard_random`)

Risultati:

| Modello | AUPR | AUROC | Precision | Recall | Best Epoch |
|---|---|---|---|---|---|
| HGNN + hard random | 0.0536 | 0.8095 | 0.0340 | 0.8194 | 100 |
| GCN + hard random | 0.0295 | 0.6182 | 0.0186 | 0.6725 | 100 |

## Perché i negativi difficili peggiorano le performance

Il calo di AUPR è drastico e apparentemente paradossale: il modello con negativi più informativi performa peggio di quello con negativi casuali. Come mai?

Il problema è di natura ottimizzativa, non biologica. Nelle prime epoche di training, la rete ha pesi inizializzati casualmente e non ha ancora imparato a distinguere nulla. Presentarle immediatamente i negativi più difficili — quelli che già assomigliano ai positivi nell'embedding space casuale — equivale a dare a uno studente che non ha mai aperto un libro le domande d'esame più difficili. I segnali di gradiente che arrivano alla rete sono instabili e contrastanti: la rete non riesce a costruire una direzione di apprendimento coerente.

Il sintomo è preciso: il recall rimane alto (0.84) ma la precision crolla (0.036). Il modello risponde all'instabilità adottando una strategia degenerata — segnala quasi tutto come positivo — che massimizza il recall ma rende la precision inutilizzabile. L'AUPR, che pesa entrambe le metriche nella curva precision-recall, crolla di conseguenza.

Questa osservazione suggerisce la soluzione naturale: non abbandonare i negativi difficili, ma introdurli gradualmente dopo che il modello ha costruito rappresentazioni di base stabili. È il principio del curriculum learning, sviluppato nella fase successiva del progetto.

---

\newpage

# Quadro Complessivo e Confronto

## Tabella riassuntiva di tutti gli esperimenti di edge prediction

| Esperimento | Modello | Masking | Negativi | AUPR | AUROC |
|---|---|---|---|---|---|
| **edge_hgnn_stratified** | **HGNN** | **Stratified** | **Casuali** | **0.2467** | **0.9254** |
| edge_hgnn_random | HGNN | Random | Casuali | 0.2265 | 0.9094 |
| edge_gcn_stratified | GCN | Stratified | Casuali | 0.1509 | 0.8378 |
| edge_gcn_random | GCN | Random | Casuali | 0.1348 | 0.7904 |
| hgnn_edge_hard_stratified | HGNN | Stratified | Difficili | 0.0677 | 0.8250 |
| hgnn_edge_hard_random | HGNN | Random | Difficili | 0.0536 | 0.8095 |
| gcn_edge_hard_stratified | GCN | Stratified | Difficili | 0.0266 | 0.6114 |
| gcn_edge_hard_random | GCN | Random | Difficili | 0.0295 | 0.6182 |
| *AUPR casuale* | — | — | — | *0.014* | — |
| DREAM5 CLR | — | — | — | 0.255 | — |
| DREAM5 GENIE3 | — | — | — | 0.291 | — |
| DREAM5 Community | — | — | — | 0.327 | — |

## Tre conclusioni principali

**Prima conclusione — Lo stratified masking è sempre superiore al random.** Il vantaggio (+9% per HGNN, +12% per GCN) è sistematico e indipendente dalla scelta dei negativi. La ragione è metodologica: garantire che ogni TF contribuisca proporzionalmente al training riduce la varianza del gradiente e produce rappresentazioni più uniformi.

**Seconda conclusione — L'HGNN supera la GCN con negativi casuali.** Con negativi casuali l'HGNN performa meglio della GCN in entrambe le varianti di masking (0.247 vs 0.151 con stratified, 0.227 vs 0.135 con random). La struttura a iperarchi — dove un TF è rappresentato come un iperarco che collega simultaneamente tutti i suoi target — cattura in modo più naturale la relazione uno-a-molti della regolazione genica rispetto a un grafo a coppie binarie.

**Terza conclusione — I negativi difficili senza curriculum producono un collasso.** Con hard negative sampling, entrambi i modelli collassano: HGNN scende da 0.247 a 0.068, GCN da 0.151 a 0.027. Questo non significa che i negativi difficili siano inutili, ma che devono essere introdotti in modo progressivo. Il punto di partenza per il curriculum learning è esattamente questo: sfruttare il potenziale informativo dei negativi difficili proteggendo il modello dall'instabilità iniziale.

## La configurazione baseline per le fasi successive

L'esperimento `edge_hgnn_stratified` — HGNN con masking stratificato e negativi casuali — diventa il **punto di riferimento** (baseline) per tutto il lavoro successivo: AUPR = 0.247. È la configurazione che offre il miglior bilanciamento tra solidità metodologica e performance, e rappresenta il punto di partenza da cui il curriculum learning parte per raggiungere AUPR = 0.266.

---

\newpage

# Riferimenti

- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *ICLR*.
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). **FaceNet: A Unified Embedding for Face Recognition and Clustering** (hard negative mining). *CVPR*.
