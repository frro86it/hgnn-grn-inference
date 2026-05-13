# ─────────────────────────────────────────────────────────────
# GRN Hypergraph Classifier — Requirements
# Progetto: GRN inference con Hypergraph Neural Networks
# Dataset:  DREAM5 Network 1
#
# Come installare:
#   1. Crea il venv:   python3 -m venv grn_hgnn_env
#   2. Attivalo:       source grn_hgnn_env/bin/activate   (Mac/Linux)
#                      grn_hgnn_env\Scripts\activate       (Windows)
#   3. Installa:       pip install -r requirements.txt
# ─────────────────────────────────────────────────────────────

# ── Dati e calcolo numerico ──────────────────────────────────
pandas>=2.0.0          # lettura file TSV/CSV (expression data, gold standard)
numpy>=1.24.0          # operazioni su matrici (matrice di incidenza H)
scipy>=1.10.0          # calcoli scientifici aggiuntivi

# ── Machine Learning classico ────────────────────────────────
scikit-learn>=1.2.0    # metriche (AUPR), train/test split, normalizzazione

# ── Visualizzazione ──────────────────────────────────────────
matplotlib>=3.7.0      # grafici delle performance (loss, AUPR)

# ── Deep Learning ────────────────────────────────────────────
torch>=2.0.0           # framework principale per la rete neurale HGNN
torch_geometric>=2.3.0 # utilities per grafi e reti neurali su grafi

# ─────────────────────────────────────────────────────────────
# NOTA: NON installiamo DHG perché richiede torch<2.0
#       che è incompatibile con torch_geometric>=2.3.0
#       Implementeremo la HGNN direttamente in PyTorch puro
#       basandoci sul paper: Feng et al. AAAI 2019
# ─────────────────────────────────────────────────────────────