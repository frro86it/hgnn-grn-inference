"""
step1.py — Caricamento dati DREAM5 Network 1
=============================================
Può essere usato in due modi:

1. Chiamato dal main.py (modalità pipeline):
       data = run(network_path)

2. Eseguito autonomamente (modalità standalone):
       python3 step1.py --network_path /percorso/Network1
       python3 step1.py --network_path /percorso/Network1 --save --output_dir ./output
"""

import os
import argparse
import pandas as pd
import numpy as np
from utils import save_checkpoint


# ══════════════════════════════════════════════════════════════════
# FUNZIONI INTERNE
# ══════════════════════════════════════════════════════════════════

def build_paths(network_path):
    """
    Costruisce automaticamente i percorsi dei 3 file
    a partire dalla cartella Network1.
    """
    expr_file = os.path.join(network_path, "input data",
                             "net1_expression_data.tsv")
    tf_file   = os.path.join(network_path, "input data",
                             "net1_transcription_factors.tsv")
    gold_file = os.path.join(network_path, "gold standard",
                             "DREAM5_NetworkInference_GoldStandard_Network1.tsv")

    for f in [expr_file, tf_file, gold_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"\n❌ File non trovato: {f}"
                f"\n   Controlla che --network_path sia corretto."
            )

    print("  ✅ Tutti i file trovati!")
    return expr_file, tf_file, gold_file


def load_expression(expr_file):
    """
    Carica la matrice di espressione genica.
    Righe = esperimenti, Colonne = geni (G1...G1643)
    """
    print("  Carico expression data...")
    df = pd.read_csv(expr_file, sep='\t', index_col=None)
    print(f"     → {df.shape[1]} geni, {df.shape[0]} esperimenti")
    return df


def load_tf_list(tf_file):
    """
    Carica la lista dei Transcription Factors.
    """
    print("  Carico transcription factors...")
    tf_list = pd.read_csv(tf_file, sep='\t', header=None)[0].tolist()
    print(f"     → {len(tf_list)} TF trovati")
    return tf_list


def load_gold_standard(gold_file):
    """
    Carica il gold standard.
    Colonne: TF | Gene | Label (1=vera, 0=falsa)
    """
    print("  Carico gold standard...")
    gold_df = pd.read_csv(gold_file, sep='\t', header=None,
                          names=['TF', 'Gene', 'Label'])
    n_pos = (gold_df['Label'] == 1).sum()
    n_neg = (gold_df['Label'] == 0).sum()
    print(f"     → {n_pos} interazioni VERE, {n_neg} FALSE")
    print(f"     → Sbilanciamento: {n_neg // n_pos}x")
    return gold_df


def print_summary(expression_df, tf_list, gold_df):
    """
    Stampa un riepilogo finale dei dati caricati.
    """
    gene_list = expression_df.columns.tolist()
    tf_in_genes = [tf for tf in tf_list if tf in gene_list]

    print("\n" + "─" * 45)
    print("  RIEPILOGO STEP 1")
    print("─" * 45)
    print(f"  Geni totali           : {len(gene_list)}")
    print(f"  Transcription Factors : {len(tf_list)}")
    print(f"  TF in expression data : {len(tf_in_genes)}/{len(tf_list)}")
    print(f"  Esperimenti           : {expression_df.shape[0]}")
    print(f"  Interazioni totali    : {len(gold_df)}")
    print(f"    - Vere  (1)         : {(gold_df['Label']==1).sum()}")
    print(f"    - False (0)         : {(gold_df['Label']==0).sum()}")
    print("─" * 45)


# ══════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE — chiamata dal main.py
# ══════════════════════════════════════════════════════════════════

def run(network_path):
    """
    Esegue lo Step 1 completo.
    Questa è la funzione che chiama main.py.

    Ritorna un dizionario con:
        expression_df : DataFrame (805 esperimenti x 1643 geni)
        tf_list       : lista dei 195 TF
        gold_df       : DataFrame [TF, Gene, Label]
        gene_list     : lista di tutti i geni in ordine
    """
    print("\n" + "═" * 55)
    print("  STEP 1 — Caricamento dati DREAM5 Network 1")
    print("═" * 55)

    expr_file, tf_file, gold_file = build_paths(network_path)
    expression_df = load_expression(expr_file)
    tf_list       = load_tf_list(tf_file)
    gold_df       = load_gold_standard(gold_file)
    gene_list     = expression_df.columns.tolist()

    print_summary(expression_df, tf_list, gold_df)
    print("\n  ✅ Step 1 completato!")

    return {
        "expression_df": expression_df,
        "tf_list":       tf_list,
        "gold_df":       gold_df,
        "gene_list":     gene_list
    }


# ══════════════════════════════════════════════════════════════════
# ESECUZIONE STANDALONE — python3 step1.py --network_path ...
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 — Carica dati DREAM5")
    parser.add_argument("--network_path", type=str, required=True,
                        help="Percorso alla cartella Network1")
    parser.add_argument("--save", action="store_true",
                        help="Salva l'output in un file .pkl")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Cartella dove salvare il checkpoint (default: ./output)")
    args = parser.parse_args()

    # Esegui
    result = run(args.network_path)

    # Salva se richiesto
    if args.save:
        save_checkpoint(result,
                        os.path.join(args.output_dir, "step1.pkl"))
