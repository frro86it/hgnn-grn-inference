# GRN Inference con Hypergraph Neural Networks

Progetto per l'inferenza di Gene Regulatory Networks (GRN) usando Hypergraph Neural Networks (HGNN) sul dataset **DREAM5 Network 1**.

---

## Struttura del progetto

```
src_5/
├── step1.py                  # Preprocessing del dataset DREAM5
├── step2.py                  # Costruzione dell'ipergrafo
├── step3.py                  # Training della HGNN
├── step4.py                  # Valutazione e metriche
├── main.py                   # Entry point principale
├── utils.py                  # Funzioni di supporto
├── compare_experiments.py    # Confronto tra esperimenti
├── run_all.sh                # Script per eseguire tutto in sequenza
├── requirements.txt          # Dipendenze Python
├── 2_DREAM5_method_scores_Supplement.xls  # Dataset di riferimento
└── output/                   # Risultati degli esperimenti
```

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

---

## Dipendenze principali

| Libreria | Versione | Uso |
|---|---|---|
| `pandas` | >=2.0.0 | Lettura file TSV/CSV (expression data, gold standard) |
| `numpy` | >=1.24.0 | Operazioni su matrici (matrice di incidenza H) |
| `scipy` | >=1.10.0 | Calcoli scientifici aggiuntivi |
| `scikit-learn` | >=1.2.0 | Metriche (AUPR), train/test split, normalizzazione |
| `matplotlib` | >=3.7.0 | Grafici delle performance (loss, AUPR) |
| `torch` | >=2.0.0 | Framework principale per la rete neurale HGNN |
| `torch_geometric` | >=2.3.0 | Utilities per grafi e reti neurali su grafi |

> **Nota:** DHG non è installato perché richiede `torch<2.0`, incompatibile con `torch_geometric>=2.3.0`.
> La HGNN è implementata direttamente in PyTorch puro, basandosi su [Feng et al. AAAI 2019](https://arxiv.org/abs/1901.08150).

---

## Come eseguire

```bash
# Esegui tutto il pipeline in sequenza
bash run_all.sh

# Oppure step by step
python step1.py
python step2.py
python step3.py
python step4.py
```

---

## Risultati

I risultati degli esperimenti sono salvati nella cartella `output/`, organizzati per configurazione (baseline, dropout, learning rate, edge split strategy, ecc.). Ogni sottocartella contiene:
- `config.json` — parametri dell'esperimento
- `metrics.json` — metriche (AUPR, loss)
- `results.png` — grafici delle performance
