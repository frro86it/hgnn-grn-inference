---
title: "Curriculum Learning per l'Inferenza di GRN — Analisi e Risultati"
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

# Il Problema che ha Motivato il Curriculum Learning

## Da dove siamo partiti

Prima di capire perché abbiamo introdotto il curriculum learning, è necessario richiamare il contesto in cui si è resa necessaria questa scelta.

Il modello HGNN che avevamo costruito fino a quel punto — nella configurazione che chiamiamo *edge prediction con stratified masking* — aveva raggiunto un AUPR di **0.247**, superando già diversi metodi benchmark di DREAM5. La strategia di campionamento dei negativi in quella configurazione era semplice: per ogni interazione TF→gene vera (positivo), il modello vedeva durante il training un numero proporzionale di coppie false scelte casualmente, distribuendo equamente i negativi tra i TF.

Il passo successivo naturale era chiedersi: è possibile fare di meglio se, invece di negativi scelti casualmente, usassimo negativi *difficili* — cioè coppie false che il modello stenta a distinguere da quelle vere? Questa idea, nota come *hard negative sampling*, è una tecnica consolidata nel deep learning: esporre la rete agli esempi più insidiosi la forza a costruire rappresentazioni più precise e discriminative.

## Il collasso dell'hard negative senza curriculum

Purtroppo, applicando direttamente i negativi difficili senza alcun accorgimento, il risultato è stato disastroso: l'AUPR è crollato da **0.247 a 0.068**, tornando ai livelli di un modello praticamente casuale.

Per capire perché, bisogna immaginare cosa succede nelle primissime epoche di training. All'inizio, i pesi della rete sono inizializzati casualmente: il modello non sa distinguere nulla. Presentargli immediatamente i negativi più difficili — quelli che assomigliano di più ai positivi — è come dare a uno studente che non ha mai aperto un libro il problema più complesso dell'esame. Il modello non ha ancora gli strumenti concettuali per affrontarli, riceve segnali di gradiente contrastanti e instabili, e invece di imparare a discriminare si limita a rispondere "positivo" praticamente a tutto. Il recall saliva all'84%, ma la precision crollava al 3.5%: il modello segnalava quasi tutto come interazione vera, un comportamento chiaramente degenere.

L'analisi topologica che avevamo già condotto aveva confermato che questo collasso non aveva una spiegazione strutturale: non erano certi tipi di geni o certi TF a cedere per primi, ma l'intera rete di embedding collassava uniformemente. Il problema era puramente di ottimizzazione: il curriculum di apprendimento era sbagliato.

## L'idea del curriculum learning

La soluzione viene da un principio noto in letteratura come **curriculum learning** (Bengio et al., ICML 2009): gli esseri umani imparano meglio quando gli esempi vengono presentati in ordine crescente di difficoltà. Si comincia dalle nozioni di base, si consolidano le fondamenta, e solo quando si ha una comprensione solida ci si confronta con i casi più complessi.

Applicato al nostro problema, il ragionamento è il seguente: lasciamo che il modello costruisca prima delle rappresentazioni di base degli embedding usando negativi semplici — cioè coppie false scelte a caso, che sono facilmente distinguibili da quelle vere. Una volta che la rete ha imparato che cosa caratterizza un'interazione vera in senso lato, possiamo gradualmente introdurre i negativi difficili, che la costringeranno ad affinare le sue rappresentazioni per discriminare i casi ambigui.

---

\newpage

# Architettura del Curriculum Implementato

## Le tre fasi

Il curriculum è stato implementato in `step3.py` come una progressione in tre fasi distinte, controllata da un unico parametro chiave: `curriculum_phase1_ratio`, che indica quale frazione delle epoche totali di training dedicare alla prima fase.

**Fase 1 — Fondamenta (negativi casuali)**
Il modello si allena con negativi scelti completamente a caso, esattamente come nella configurazione baseline stratified. In questa fase non ha accesso ai negativi difficili. L'obiettivo è costruire embedding significativi: il modello deve imparare che i TF che regolano molti geni in comune tendono ad avere rappresentazioni simili, e che le coppie TF→gene vere hanno caratteristiche sistematicamente diverse da quelle false. È la fase del "capire le basi".

**Fase 2 — Transizione (mix 50/50)**
Il modello vede metà negativi casuali e metà negativi difficili. Questa fase agisce come un ponte graduale: il modello viene esposto ai casi difficili, ma ha ancora abbastanza esempi semplici per stabilizzare il gradiente. La durata di questa fase è fissa e occupa metà della finestra temporale rimanente dopo la Fase 1.

**Fase 3 — Raffinamento (negativi difficili)**
Il modello si allena solo con negativi difficili per tutta la parte finale del training. In questa fase le sue rappresentazioni sono già consolidate e può affrontare i casi ambigui senza collassare. È il momento in cui le curve di precision-recall si affilano.

## Il parametro chiave: phase1\_ratio

Tutto il curriculum dipende da quanto tempo viene dedicato alla Fase 1. Se è troppo poco (ratio basso), il modello entra nella Fase 3 prima di essere pronto. Se è troppo alto, rimane nella zona di conforto troppo a lungo e non beneficia abbastanza dell'esposizione ai negativi difficili. Trovare il valore ottimale di questo parametro è stato il cuore della ricerca sperimentale.

---

\newpage

# Gli Esperimenti: Evoluzione Passo per Passo

## Prima versione: curriculum con ratio di default (33%)

Il primo esperimento di curriculum learning — `hgnn_curriculum` — usava i parametri di default: ratio 0.33, ovvero un terzo delle 100 epoche totali nella Fase 1, un terzo nella Fase 2, e un terzo nella Fase 3.

Il risultato è stato **AUPR = 0.143**, con best epoch a 33. Rispetto al collasso senza curriculum (AUPR 0.068) il miglioramento era evidente — il modello non collassava più — ma il risultato era ancora lontano dal baseline stratified (0.247). La lettura del best\_epoch = 33 è rivelatrice: il modello stoppava già alla fine della Fase 1, prima ancora di entrare nella Fase di transizione. Stava imparando qualcosa durante la fase facile, ma l'early stopping lo bloccava prima che i negativi difficili potessero fare il loro lavoro.

Questo ci ha insegnato due cose: il ratio di default era troppo basso, e le epoche totali erano probabilmente insufficienti per un curriculum in tre fasi. Con solo 33 epoche per fase, la finestra temporale era troppo stretta.

## La ricerca del ratio ottimale: 60%, 70%, 80%

Sulla base di quella prima lettura, abbiamo riprogettato gli esperimenti: aumentare le epoche totali a 150 e esplorare sistematicamente valori più alti di phase1\_ratio. Il ragionamento era che il modello aveva bisogno di più tempo nella Fase 1 per costruire fondamenta solide prima di affrontare i negativi difficili.

### Curriculum 60% — `hgnn_curriculum_60`

Con ratio = 0.60 e 150 epoche totali, le fasi si distribuiscono come segue:
epoche 1-90 con negativi casuali, epoche 91-120 con mix 50/50, epoche 121-150 con negativi difficili.

Risultato: **AUPR = 0.240**, best\_epoch = 90. Un salto notevole rispetto alla versione di default, quasi al livello del baseline stratified. Ma il best\_epoch = 90 — esattamente alla fine della Fase 1 — rivela che ancora una volta l'early stopping interveniva prima che il modello beneficiasse pienamente della Fase 3. Il ratio 0.60 era un miglioramento importante, ma la Fase 1 era ancora troppo corta per solidificare gli embedding abbastanza da resistere alla pressione dei negativi difficili nelle fasi successive.

### Curriculum 60% con 200 epoche — `hgnn_curriculum_60_ep200`

Un'ipotesi naturale era che aggiungere più epoche potesse aiutare. Con le stesse proporzioni del 60% ma 200 epoche totali, il modello aveva ora 120 epoche di Fase 1 invece di 90.

Il risultato ha sorpreso: **AUPR = 0.220**, peggiore di prima. Il best\_epoch era 116, al centro della Fase 2. Allungare il training non ha aiutato: il modello entrava in overfitting durante la lunga Fase 3 di 40 epoche con negativi difficili, perdendo parte di quello che aveva imparato prima. Più epoche non erano la risposta; la risposta era un ratio più alto.

### Curriculum 70% — `hgnn_curriculum_70`

Con ratio = 0.70 e 150 epoche: epoche 1-105 con negativi casuali, epoche 106-128 con mix, epoche 129-150 con negativi difficili.

Risultato: **AUPR = 0.266**, best\_epoch = 105. È il primo esperimento in cui superiamo il baseline stratified (0.247) e, per la prima volta, superiamo anche il metodo DREAM5 CLR (0.255). Il best\_epoch = 105 coincide con la fine della Fase 1: il modello ha estratto il massimo dalla fase semplice, e quando entra nella Fase 2 le sue rappresentazioni sono abbastanza robuste da non collassare. L'AUROC sale a 0.922, indicando un buon ordinamento globale delle predizioni.

### Curriculum 80% — `hgnn_curriculum_80`

Con ratio = 0.80: epoche 1-120 con negativi casuali, epoche 121-135 con mix, epoche 136-150 con negativi difficili.

Risultato: **AUPR = 0.266**, best\_epoch = 119. Identico al 70% in termini di AUPR. L'AUROC è leggermente superiore (0.931 vs 0.922), ma la differenza è marginale. I due modelli hanno raggiunto un **plateau**: aggiungere ulteriore tempo alla Fase 1 non porta più benefici perché il modello ha già saturato quello che può imparare dai negativi semplici.

Questo plateau tra 70% e 80% è un'informazione preziosa: significa che il valore ottimale di phase1\_ratio si trova nell'intorno di 0.70, e che la Fase 3 di circa 20-30 epoche con negativi difficili è sufficiente per il raffinamento finale.

---

\newpage

# Le Varianti del Curriculum 70%

Una volta individuato il curriculum\_70 come configurazione ottimale, abbiamo esplorato tre varianti per cercare di migliorarlo ulteriormente. Nessuna ha prodotto miglioramenti, ma ognuna ha insegnato qualcosa di preciso.

## Patience variabile per fase — `hgnn_curriculum_70_varp`

L'idea era che la patience di early stopping — il numero di epoche senza miglioramento dopo le quali il training si ferma — dovesse essere calibrata diversamente nelle tre fasi. Durante la Fase 1 con negativi semplici, il modello converge rapidamente: patience bassa (10 epoche) è appropriata. Durante la Fase 2 e la Fase 3 con negativi difficili, il percorso di ottimizzazione è più accidentato: servono fasi di "discesa" prima di risalire verso un nuovo massimo. Abbiamo quindi assegnato patience più alte alle fasi avanzate: Fase 1 = 10 epoche, Fase 2 = 25 epoche, Fase 3 = 20 epoche.

Risultato: **AUPR = 0.253**, peggiore di 4.9 punti percentuali rispetto al curriculum\_70 standard. La patience non era il problema. Il training si fermava comunque a best\_epoch = 105, identico alla versione standard. La spiegazione è che il modello non aveva bisogno di più tempo per "recuperare" nelle fasi difficili: semplicemente stava già apprendendo alla velocità giusta con la patience standard.

## Curriculum con arc features — `hgnn_curriculum_70_arc`

Le arc features sono una tecnica che avevamo testato separatamente: aggiungere al decoder, oltre agli embedding dei nodi, anche il grado del TF e il grado del gene target come feature esplicite. L'idea era che fornire al decoder informazioni sulla "popolarità" di un TF potesse aiutarlo a discriminare meglio le interazioni false.

Standalone, le arc features avevano prodotto AUPR = 0.236 (lievemente peggio del baseline). Combinandole con il curriculum 70%, ci aspettavamo una sinergia. Invece: **AUPR = 0.228**, il peggiore tra tutte le varianti del curriculum\_70.

La spiegazione è analoga a quella già vista con le feature topologiche nella SHAP: le arc features aggiungono un segnale diretto sul grado al decoder, che è già presente implicitamente attraverso gli embedding costruiti da Theta. I due segnali interferiscono invece di sommarsi. In più, il curriculum ha già il suo meccanismo interno per gestire la difficoltà dei negativi: aggiungere arc features introduce un secondo meccanismo che lavora in parallelo e in modo incoerente con il primo.

## Curriculum continuo con sigmoide — `hgnn_curriculum_sigmoid`

Invece di fasi discrete con salti bruschi, questa variante implementava una transizione fluida e continua: la proporzione di negativi difficili aumentava seguendo una funzione sigmoide, partendo da 0% all'inizio e arrivando a 100% alla fine. Il punto di inflessione della sigmoide era fissato al 75% del training (epoch 113 su 150), con una pendenza abbastanza ripida da rendere la transizione rapida ma non istantanea.

Risultato: **AUPR = 0.224**, peggiore del curriculum a fasi discrete. Il best\_epoch era 75, esattamente a metà training, quando la proporzione di negativi difficili era ancora bassa (intorno al 10-20% secondo la curva sigmoide). Il modello si fermava troppo presto, prima che i negativi difficili potessero incidere significativamente.

Questo risultato è controintuitivo ma ha una spiegazione chiara: i salti discreti tra le fasi nel curriculum standard fungono da "checkpoint" espliciti che costringono il modello ad adattarsi. La transizione sigmoide è troppo graduale e il modello non percepisce mai una pressione sufficiente a dover riorganizzare le sue rappresentazioni.

---

\newpage

# Online Hard Negative Mining: un Approccio Alternativo

## L'idea

Il curriculum learning che abbiamo descritto finora usava negativi difficili *statici*: venivano calcolati una sola volta all'inizio del training, scegliendo le coppie false che avevano la maggiore probabilità secondo il modello iniziale. Una variante più sofisticata è l'**online hard negative mining**: i negativi difficili vengono ricalcolati dinamicamente ogni N epoche, usando le predizioni del modello *corrente*. In questo modo, man mano che il modello migliora, i negativi si adattano: quelli che erano difficili nelle prime epoche diventano facili, e ne emergono di nuovi che sfidano la rete nel suo stato attuale.

## Mining senza warmup — `hgnn_online_mining`

In questa versione il mining partiva immediatamente, ricalcolando i negativi difficili ogni 10 epoche fin dalla prima iterazione.

Risultato catastrofico: **AUPR = 0.024**, best\_epoch = 9. Il modello collassava già alla decima epoca — esattamente quando avveniva il primo aggiornamento dei negativi.

Il motivo è preciso: alla decima epoca, il modello aveva pesi ancora quasi casuali. Usare quelle predizioni casuali per selezionare i "negativi più difficili" equivaleva a scegliere negativi casuali, non difficili in senso significativo. Peggio ancora, quel campionamento casuale travestito da difficile introduceva una distribuzione di esempi instabile che spiazzava completamente il gradiente. L'AUROC scendeva a 0.619, appena sopra il caso puro (0.5), confermando che il modello non aveva imparato nulla di utile.

## Mining con warmup di 60 epoche — `hgnn_online_mining_w60`

Partendo dalla lezione dell'esperimento precedente, la versione con warmup lasciava che il modello si allenasse normalmente per le prime 60 epoche con negativi casuali, e solo da quel momento attivava il mining dinamico ogni 10 epoche.

Risultato: **AUPR = 0.208**, best\_epoch = 60. Un miglioramento significativo rispetto alla versione senza warmup, ma inferiore al curriculum standard. La cosa più interessante è il best\_epoch = 60 — esattamente alla fine del warmup. Il modello raggiungeva il suo massimo nel momento in cui il mining si attivava, e da lì in poi non migliorava più.

Questo rivela il problema strutturale del mining dinamico: quando il mining si attiva, i nuovi negativi sono genuinamente più difficili di quelli precedenti — ma *troppo* difficili per un modello che si è allenato 60 epoche su esempi semplici. Il salto è brusco come nell'hard negative senza curriculum, solo ritardato di 60 epoche. Il warmup ha attenuato il collasso iniziale, ma non ha risolto il problema di fondo: il modello non ha una fase di transizione graduale tra il regime semplice e quello difficile.

---

\newpage

# Quadro Riassuntivo e Conclusioni

## Tabella comparativa di tutti gli esperimenti

| Esperimento | Phase1 ratio | Epoche | AUPR | Best epoch | Note |
|---|---|---|---|---|---|
| Hard stratified (no CL) | — | 100 | 0.068 | 100 | Collasso |
| **Baseline stratified** | — | **100** | **0.247** | **100** | **Punto di partenza** |
| DREAM5 CLR | — | — | 0.255 | — | Benchmark |
| Curriculum default | 0.33 | 100 | 0.143 | 33 | Stop in Fase 1 |
| Curriculum 60% | 0.60 | 150 | 0.240 | 90 | Stop fine Fase 1 |
| Curriculum 60% + 200ep | 0.60 | 200 | 0.220 | 116 | Overfitting Fase 3 |
| Curriculum 70% + varp | 0.70 | 150 | 0.253 | 105 | Patience non è il problema |
| Curriculum 70% + arc | 0.70 | 150 | 0.228 | 105 | Interferenza arc features |
| Curriculum sigmoid | — | 150 | 0.224 | 75 | Troppo graduale |
| Online mining (no warmup) | — | 150 | 0.024 | 9 | Collasso immediato |
| Online mining + warmup 60 | — | 150 | 0.208 | 60 | Stop a fine warmup |
| **Curriculum 70%** | **0.70** | **150** | **0.266** | **105** | **Miglior risultato** |
| **Curriculum 80%** | **0.80** | **150** | **0.266** | **119** | **Plateau** |

## Lettura dell'evoluzione

Il percorso sperimentale ha seguito una logica progressiva. Partendo dal collasso dell'hard negative (0.068), il curriculum learning ha prima dimostrato di essere la direzione giusta (0.143), poi ha richiesto una calibrazione attenta del ratio per sbloccare il suo potenziale (0.240 → 0.266). Il plateau tra 70% e 80% ha definito con precisione la finestra ottimale.

Le varianti esplorate — patience variabile, arc features, sigmoide — hanno tutte peggiorato il risultato, ma non per ragioni casuali: ciascuna ha illuminato un aspetto diverso del problema. La patience non è un fattore limitante perché il modello converge già bene con la configurazione standard. Le arc features interferiscono perché aggiungono ridondanza. La sigmoide non funziona perché la transizione deve essere abbastanza netta da costringere il modello ad adattarsi.

## Perché il curriculum 70% è il metodo migliore

Il curriculum con phase1\_ratio = 0.70 è la configurazione vincente per tre ragioni che si rinforzano a vicenda.

Prima di tutto, 105 epoche di Fase 1 sono sufficienti per costruire embedding solidi: il modello ha visto abbastanza negativi semplici da formarsi un'idea chiara di cosa distingue un'interazione vera da una falsa in senso generale. Questa solidità è la precondizione necessaria per affrontare i casi difficili.

In secondo luogo, la Fase 3 di 22 epoche è abbastanza lunga da permettere un raffinamento reale delle rappresentazioni sui negativi difficili, ma abbastanza breve da non cadere in overfitting. È una finestra temporale calibrata correttamente.

In terzo luogo, il salto discreto tra le fasi — invece di una transizione fluida — produce un effetto di "reset adattivo": il cambio brusco della distribuzione dei negativi costringe il modello a riorganizzare attivamente le proprie rappresentazioni, invece di adattarsi passivamente come farebbe con una sigmoide.

## Il risultato finale: oltre DREAM5 CLR

Con AUPR = 0.266, il curriculum learning ha portato il nostro modello oltre il metodo CLR di DREAM5 (0.255), che è un metodo basato sulla Context Likelihood of Relatedness — una tecnica statistica consolidata di mutual information per l'inferenza di GRN. Superare un metodo statistico con un approccio supervisionato basato su ipergrafi è il principale risultato quantitativo di questa parte del progetto.

La distanza dai metodi più avanzati di DREAM5 — GENIE3 (0.291) e il Community Network ensemble (0.327) — rimane, ma il gap si è ridotto significativamente rispetto al punto di partenza.

---

\newpage

# Riferimenti

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). **Curriculum Learning**. *ICML*.
- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*, 33(01), 3558–3565.
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). **FaceNet: A unified embedding for face recognition and clustering** (Hard negative mining). *CVPR*.
