# GRN Inference con Hypergraph Neural Networks — DREAM5 Network 1

**Progetto:** GRN-Hypergraph-Classifier  
**Autori:** Francesco Rollin, Fabio  
**Tutor:** Gerardo (Dino) — Università di Salerno, Bioinformatica  
**Versione:** src_XIV (22 giugno 2026)  
**Dataset:** DREAM5 Network 1 (*Escherichia coli*, 1643 geni, 195 TF, 4012 interazioni)

---

## Panoramica

Questo progetto implementa l'**inferenza di Reti di Regolazione Genica (GRN)** usando **Hypergraph Neural Networks (HGNN)** e **Graph Convolutional Networks (GCN)** sul dataset DREAM5 Network 1.

L'obiettivo è predire interazioni TF→gene (link prediction) e confrontare diversi approcci di costruzione del grafo/ipergrafo, strategie di campionamento dei negativi, e tecniche di training come il **curriculum learning**. Il modello viene valutato con la metrica **AUPR** (Area Under the Precision-Recall Curve), adatta al forte sbilanciamento del dataset (68 negativi per ogni positivo).

---

## Prerequisiti

- **Python** 3.10 o superiore
- **Dataset DREAM5** scaricato manualmente (vedi sotto)
- Circa **2 GB** di spazio su disco

---

## Download del Dataset

1. Vai su [https://zenodo.org/records/17854236](https://zenodo.org/records/17854236)
2. Scarica il file `1_Challenge_Data_Supplement.zip`
3. Estrai lo zip — trovi la cartella `Network1/` con questa struttura:

```
Network1/
├── input data/
│   ├── net1_expression_data.tsv      # Matrice espressione: 805 esperimenti × 1643 geni
│   └── net1_transcription_factors.tsv # Lista dei 195 TF
└── gold standard/
    └── DREAM5_NetworkInference_GoldStandard_Network1.tsv  # 4012 interazioni vere
```

> **Nota:** Il percorso alla cartella `Network1/` viene passato a tutti gli script con il flag `--network_path`.

---

## Installazione

**1. Crea il virtual environment:**
```bash
python3 -m venv grn_hgnn_env
```

**2. Attivalo:**
```bash
# Mac/Linux
source grn_hgnn_env/bin/activate

# Windows
grn_hgnn_env\Scripts\activate
```

**3. Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

**4. (Opzionale) Installa SHAP per la feature importance:**
```bash
pip install shap
```

---

## Dipendenze principali

| Libreria | Versione | Uso |
|---|---|---|
| `torch` | ≥2.0 | Framework per HGNN e GCN |
| `torch-geometric` | ≥2.3 | Utilities per grafi |
| `scikit-learn` | ≥1.2 | Metriche (AUPR), normalizzazione |
| `numpy` | ≥1.24 | Matrici H, Θ |
| `pandas` | ≥2.0 | Lettura file TSV/CSV |
| `scipy` | ≥1.10 | Calcoli scientifici |
| `matplotlib` | ≥3.7 | Grafici delle performance |
| `shap` | ≥0.49 | Feature importance (opzionale) |

> **Nota:** DHG non è installato perché richiede `torch<2.0`, incompatibile con `torch-geometric≥2.3`.
> La HGNN è implementata direttamente in PyTorch puro, basandosi su [Feng et al. AAAI 2019](https://arxiv.org/abs/1901.08150).

---

## Struttura del Progetto

```
src_XIV/
├── main.py                     # Orchestratore unificato di tutti gli esperimenti
├── step1.py                    # Caricamento dati DREAM5 (espressione, TF, gold standard)
├── step2.py                    # Costruzione ipergrafo H e matrice di propagazione Θ
├── step3.py                    # Training HGNN e GCN (con curriculum, arc features, ecc.)
├── step4.py                    # Valutazione: AUPR, AUROC, precision, recall, F1
├── utils.py                    # Funzioni condivise: checkpoint, JSON, logging
│
├── analyze_topology.py         # Analisi topologica: grado, entropia, spettro di Θ
├── analyze_sparsity.py         # Confronto sparsità H: gold standard vs similarità
├── analyze_shap.py             # Feature importance con Integrated Gradients (SHAP)
├── compare_experiments.py      # Genera comparison.png + experiments_summary.csv
│
├── run_all_experiments.sh      # Script bash: riproduce tutti gli esperimenti in sequenza
├── requirements.txt            # Dipendenze Python (versioni esatte)
│
├── 2_DREAM5_method_scores_Supplement.xls  # Benchmark ufficiale DREAM5
│
└── output/                     # Risultati degli esperimenti (generato automaticamente)
    ├── comparison.png           # Grafico AUPR a barre, tutti i modelli
    ├── experiments_summary.csv  # Tabella riassuntiva con tutte le metriche
    ├── topology_analysis/       # Grafici topologici (grado, entropia, embedding)
    ├── sparsity_analysis/       # Heatmap e confronto sparsità H
    ├── shap_analysis/           # Feature importance + ig_importance.csv
    └── <exp_name>/              # Una sottocartella per ogni esperimento
        ├── config.json          #   Parametri usati
        ├── metrics.json         #   Risultati (AUPR, AUROC, F1, ...)
        └── results.png          #   Curve loss e precision-recall
```

### Descrizione dei file principali

**`step1.py`** — Carica i tre file DREAM5 e costruisce il dizionario `data` con:
- `X`: matrice di espressione genica (1643 × 805), normalizzata
- `tf_idx`: indici dei 195 TF nella matrice X
- `edges_pos`: lista delle 4012 coppie (TF, gene) vere
- `edges_neg`: lista dei negativi campionati

**`step2.py`** — Costruisce le strutture per il modello scelto:
- **HGNN**: matrice di incidenza `H` (1643 × 195) e matrice di propagazione `Θ`
- **GCN**: matrice di adiacenza sparsa `A`
- Gestisce le strategie di masking: `random`, `stratified`, `hard_stratified`

**`step3.py`** — Training del modello con:
- Architettura HGNN (2 layer) o GCN (2 layer), decoder MLP per link prediction
- Early stopping su AUPR di validazione
- Flag opzionali: `--curriculum`, `--use_arc_features`, `--weighted_tf`, `--online_mining`

**`step4.py`** — Calcola e salva AUPR, AUROC, precision, recall, F1 sul test set.

**`main.py`** — Entry point unificato che chiama step1→2→3→4 in sequenza con i parametri scelti.

---

## Come Eseguire

### Opzione 1 — Singolo esperimento

```bash
python main.py \
  --network_path /percorso/a/Network1 \
  --analysis hgnn_edge_stratified \
  --exp_name mio_esperimento \
  --epochs 100 \
  --dropout 0.5
```

### Opzione 2 — Tutti gli esperimenti in sequenza

```bash
# Prima, modifica la variabile NET nel file:
# NET="/percorso/a/Network1"

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

> Gli esperimenti già completati vengono saltati automaticamente grazie al flag `--skip_existing`.

### Opzione 3 — Analisi separate

```bash
# Analisi topologica dell'ipergrafo
python analyze_topology.py --network_path /percorso/a/Network1 \
                           --output_dir ./output/topology_analysis

# Analisi sparsità delle matrici H
python analyze_sparsity.py --network_path /percorso/a/Network1 \
                           --k 10 \
                           --output_dir ./output/sparsity_analysis

# Feature importance (richiede prima: --save_checkpoints su edge_hgnn_stratified_topo)
python analyze_shap.py --network_path /percorso/a/Network1 \
                       --exp_dir ./output/edge_hgnn_stratified_topo \
                       --output_dir ./output/shap_analysis

# Grafico comparativo di tutti gli esperimenti già completati
python compare_experiments.py --output_base ./output
```

---

## Analisi Disponibili (`--analysis`)

| Flag | Struttura grafo | Modello | Descrizione |
|---|---|---|---|
| `hgnn_gold` | Gold standard (80/20) | HGNN | Baseline biologica |
| `gcn_gold` | Gold standard (80/20) | GCN | Baseline GCN |
| `hgnn_statistical` | Similarità KNN | HGNN | Approccio Rosario-like |
| `gcn_statistical` | Similarità KNN | GCN | Approccio Rosario-like |
| `hgnn_edge_stratified` | Edge prediction | HGNN | **Miglior baseline** |
| `hgnn_edge_random` | Edge prediction | HGNN | Masking casuale |
| `gcn_edge_stratified` | Edge prediction | GCN | — |
| `gcn_edge_random` | Edge prediction | GCN | — |
| `hgnn_edge_hard_stratified` | Edge prediction | HGNN | Hard negative senza curriculum |
| `hgnn_edge_hard_random` | Edge prediction | HGNN | Hard negative random |
| `gcn_edge_hard_stratified` | Edge prediction | GCN | — |
| `gcn_edge_hard_random` | Edge prediction | GCN | — |
| `all` | — | — | Tutte le 12 analisi di base |

### Flag aggiuntivi (`--analysis hgnn_edge_*`)

| Flag | Descrizione |
|---|---|
| `--use_topo_features` | Aggiunge grado ed entropia come feature in X (805→807) |
| `--weighted_tf` | Pesa gli iperarchi in Θ con W=1/√(grado_TF) |
| `--use_arc_features` | Aggiunge deg_TF e deg_gene al decoder |
| `--curriculum` | Curriculum learning: random→mixed→hard (3 fasi) |
| `--curriculum_phase1_ratio` | Frazione di epoche dedicate alla fase 1 (default: 0.33) |
| `--curriculum_patience` | Patience per fase, es. `10,25,20` |
| `--curriculum_sigmoid` | Curriculum continuo a sigmoide invece che a fasi |
| `--online_mining` | Hard negative mining dinamico ogni N epoche |
| `--online_mining_warmup` | Epoche di warmup prima di attivare il mining |
| `--save_checkpoints` | Salva step3.pkl con il modello (necessario per SHAP) |

---

## Risultati

Classifica finale per AUPR (↑ = migliore). Baseline random: AUPR = 0.014.

| Modello | AUPR | Note |
|---|---|---|
| DREAM5 Community (ensemble) | **0.327** | Miglior metodo DREAM5 |
| DREAM5 Best Regression | 0.313 | — |
| DREAM5 TIGRESS | 0.301 | — |
| DREAM5 GENIE3 | 0.291 | — |
| **HGNN Curriculum 70% / 80%** | **0.266** | ← Nostro miglior risultato |
| DREAM5 CLR | 0.255 | **Superato** dal curriculum ✅ |
| HGNN Edge Stratified (baseline) | 0.247 | Punto di partenza |
| GCN Statistical (Rosario-like) | 0.203 | Struttura da similarità |
| HGNN Statistical (Rosario-like) | 0.184 | — |
| GCN Gold Standard | 0.151 | — |
| HGNN Hard Stratified (no curriculum) | 0.068 | Hard negatives senza warm-up |
| HGNN Gold Standard (baseline) | 0.068 | Struttura biologica ridotta |

> Il **curriculum learning con phase1_ratio=0.70** è il miglior modello: AUPR = 0.266, che supera il metodo CLR di DREAM5 (0.255) e si avvicina a GENIE3 (0.291).

---

## Riferimenti

- Feng, Y. et al. (2019). **Hypergraph Neural Networks**. *AAAI*. [arXiv:1901.08150](https://arxiv.org/abs/1901.08150)
- Kipf, T.N. & Welling, M. (2017). **Semi-Supervised Classification with GCN**. *ICLR*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)
- Marbach, D. et al. (2012). **Wisdom of crowds for robust gene network inference**. *Nature Methods*, 9(8), 796–804. (DREAM5)
- Latchman, D.S. (1997). **Transcription Factors: An Overview**. *Int. J. Biochem. Cell Biol.*, 29(12), 1305–1312.
- Bengio, Y. et al. (2009). **Curriculum Learning**. *ICML*.
