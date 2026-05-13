"""
step4.py — Valutazione con AUPR sul test set
=============================================
Metriche calcolate:
    AUPR  — Area Under Precision-Recall curve (metrica principale)
    AUROC — Area Under ROC curve
    Precision, Recall, F1 a soglia 0.5

PROPORZIONE NEL TEST SET:
    Usiamo la stessa proporzione del gold standard DREAM5 (1:68)
    per rendere la baseline confrontabile con DREAM5.

    803 positivi × 68 = 54.604 negativi nel test
    baseline = 803 / 55.407 = 1.45% ≈ baseline DREAM5 (1.44%)

    Prima usavamo 10x (9.09%) — non confrontabile con DREAM5.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score
)
from utils import save_checkpoint, load_checkpoint

# Proporzione positivi/negativi nel gold standard DREAM5
# 4012 positivi / 274380 negativi ≈ 1:68
GOLD_STANDARD_RATIO = 68


# ══════════════════════════════════════════════════════════════════
# PREDIZIONE SUL TEST SET
# ══════════════════════════════════════════════════════════════════

def predict_test_set(model, embeddings, test_pos, test_neg,
                     gene_list, device):
    """
    Calcola le probabilità per tutte le coppie del test set.

    PROPORZIONE 1:68 — stessa del gold standard DREAM5
    Questo rende la baseline AUPR confrontabile con DREAM5
    (≈1.44% invece del 9.09% precedente).

    Ritorna:
        y_true : etichette reali (1=vera, 0=falsa)
        y_pred : probabilità predette
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    test_pairs  = []
    test_labels = []

    # Positivi del test
    for _, row in test_pos.iterrows():
        if row['TF'] in gene_to_idx and row['Gene'] in gene_to_idx:
            test_pairs.append((gene_to_idx[row['TF']],
                               gene_to_idx[row['Gene']]))
            test_labels.append(1.0)

    n_pos = len(test_labels)

    # Negativi del test — proporzione 1:68 come DREAM5
    # Usa tutti i test_neg disponibili fino al massimo 68x i positivi
    max_neg         = min(len(test_neg), n_pos * GOLD_STANDARD_RATIO)
    test_neg_sample = test_neg.sample(n=max_neg, random_state=42)

    for _, row in test_neg_sample.iterrows():
        if row['TF'] in gene_to_idx and row['Gene'] in gene_to_idx:
            test_pairs.append((gene_to_idx[row['TF']],
                               gene_to_idx[row['Gene']]))
            test_labels.append(0.0)

    n_neg = test_labels.count(0.0)
    print(f"     → Positivi nel test : {n_pos}")
    print(f"     → Negativi nel test : {n_neg}"
          f" (proporzione 1:{n_neg//n_pos if n_pos>0 else 0})")
    print(f"     → Totale coppie     : {len(test_pairs)}")

    # Calcola probabilità
    emb_tensor = torch.tensor(embeddings,
                               dtype=torch.float32).to(device)
    tf_idx   = torch.tensor([p[0] for p in test_pairs],
                              dtype=torch.long).to(device)
    gene_idx = torch.tensor([p[1] for p in test_pairs],
                              dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        preds = model.predict(emb_tensor, tf_idx, gene_idx)
        preds = preds.squeeze().cpu().numpy()

    return np.array(test_labels), preds


# ══════════════════════════════════════════════════════════════════
# CALCOLO METRICHE
# ══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Calcola tutte le metriche di valutazione.

    La baseline AUPR è calcolata come proporzione di positivi
    nel test set — con proporzione 1:68 vale circa 1.45%,
    confrontabile con la baseline DREAM5 (1.44%).
    """
    aupr  = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)

    y_binary  = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_binary, zero_division=0)
    recall    = recall_score(y_true,    y_binary, zero_division=0)
    f1        = f1_score(y_true,        y_binary, zero_division=0)

    # Baseline: proporzione positivi nel test set
    # Con ratio 1:68 → baseline ≈ 1.45% ≈ DREAM5
    baseline_aupr = float(y_true.mean())
    improvement   = aupr / baseline_aupr if baseline_aupr > 0 else 0

    return {
        "aupr":          aupr,
        "auroc":         auroc,
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "baseline_aupr": baseline_aupr,
        "improvement":   improvement,
    }


# ══════════════════════════════════════════════════════════════════
# GRAFICI
# ══════════════════════════════════════════════════════════════════

def plot_curves(y_true, y_pred, history, model_name='Model',
                output_dir=None):
    """
    Genera due grafici:
        1. Precision-Recall curve con AUPR e baseline
        2. Training history: loss e AUPR per epoca
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Grafico 1: Precision-Recall curve ─────────────────────────
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true, y_pred
    )
    aupr     = average_precision_score(y_true, y_pred)
    baseline = float(y_true.mean())
    ratio    = int(round((1 - baseline) / baseline))

    axes[0].plot(recall_vals, precision_vals,
                 color='steelblue', linewidth=2,
                 label=f"{model_name} (AUPR = {aupr:.4f})")
    axes[0].axhline(
        y=baseline, color='red', linestyle='--', alpha=0.7,
        label=f"Random baseline (AUPR = {baseline:.4f})\n"
              f"proporzione 1:{ratio} come DREAM5"
    )
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve — Test Set")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])

    # ── Grafico 2: Training history ────────────────────────────────
    if history:
        epochs      = [h['epoch'] for h in history]
        losses      = [h['loss']  for h in history]
        aupr_vals   = [h['aupr']  for h in history if h['aupr'] > 0]
        aupr_epochs = [h['epoch'] for h in history if h['aupr'] > 0]

        ax2 = axes[1]
        ax3 = ax2.twinx()

        ax2.plot(epochs, losses,
                 color='steelblue', linewidth=2, label="Loss")
        if aupr_vals:
            ax3.plot(aupr_epochs, aupr_vals,
                     color='orange', linewidth=2,
                     linestyle='--', label="AUPR (train)")

        ax2.set_xlabel("Epoca")
        ax2.set_ylabel("Loss",  color='steelblue')
        ax3.set_ylabel("AUPR",  color='orange')
        ax2.set_title("Training History")
        ax2.grid(True, alpha=0.3)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper right')

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "results.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  📊 Grafici salvati: {path}")
    else:
        plt.show()

    plt.close()


# ══════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE
# ══════════════════════════════════════════════════════════════════

def run(data, output_dir=None):
    """
    Esegue lo Step 4 completo.
    Il modello arriva già in memoria dallo Step 3.
    """
    print("\n" + "═" * 55)
    print("  STEP 4 — Valutazione AUPR sul test set")
    print(f"  (proporzione test 1:{GOLD_STANDARD_RATIO} come DREAM5)")
    print("═" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model      = data['model']
    embeddings = data['embeddings']
    test_pos   = data['test_pos']
    test_neg   = data['test_neg']
    gene_list  = data['gene_list']
    history    = data.get('history', [])
    model_name = data.get('model_type', 'Model').upper()

    # Predici
    print("\n  Calcolo predizioni sul test set...")
    y_true, y_pred = predict_test_set(
        model, embeddings, test_pos, test_neg, gene_list, device
    )

    # Metriche
    print("\n  Calcolo metriche...")
    metrics = compute_metrics(y_true, y_pred)

    # Stampa
    print("\n" + "─" * 50)
    print("  RISULTATI FINALI — TEST SET")
    print("─" * 50)
    print(f"  AUPR            : {metrics['aupr']:.4f}"
          f"  ← metrica principale")
    print(f"  AUROC           : {metrics['auroc']:.4f}")
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  {'─'*35}")
    print(f"  Baseline random : {metrics['baseline_aupr']:.4f}"
          f"  (≈ DREAM5 baseline)")
    print(f"  Miglioramento   : {metrics['improvement']:.1f}x"
          f" rispetto al random")
    print("─" * 50)

    # Grafici
    print("\n  Genero grafici...")
    plot_curves(y_true, y_pred, history,
                model_name=model_name,
                output_dir=output_dir)

    print("\n  ✅ Step 4 completato!")

    return {
        "aupr":          metrics['aupr'],
        "auroc":         metrics['auroc'],
        "precision":     metrics['precision'],
        "recall":        metrics['recall'],
        "f1":            metrics['f1'],
        "baseline_aupr": metrics['baseline_aupr'],
        "improvement":   metrics['improvement'],
        "y_true":        y_true,
        "y_pred":        y_pred,
    }


# ══════════════════════════════════════════════════════════════════
# ESECUZIONE STANDALONE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4 — Valuta il modello con AUPR"
    )
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--output_dir",  type=str, default="./output")
    args = parser.parse_args()

    data_step3 = load_checkpoint(args.checkpoint)

    from step3 import HGNN
    in_dim     = data_step3['X'].shape[1]
    hidden_dim = data_step3['embeddings'].shape[1]
    model      = HGNN(in_dim=in_dim, hidden_dim=hidden_dim,
                      out_dim=hidden_dim)
    model.load_state_dict(
        torch.load(args.model_path, map_location='cpu')
    )
    data_step3['model'] = model

    results = run(data_step3, output_dir=args.output_dir)

    if args.save:
        save_checkpoint(
            {k: v for k, v in results.items()
             if k not in ['y_true', 'y_pred']},
            os.path.join(args.output_dir, "step4_results.pkl")
        )
