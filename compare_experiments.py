"""
compare_experiments.py — Confronto di tutti gli esperimenti
=============================================================
Legge tutti i config.json e metrics.json nella cartella output
e genera una tabella comparativa e un grafico.

Uso:
    python compare_experiments.py --output_base ./output

Output:
    ./output/experiments_summary.csv  ← tabella con tutti i risultati
    ./output/comparison.png           ← grafico comparativo
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# LEGGI TUTTI GLI ESPERIMENTI
# ══════════════════════════════════════════════════════════════════

def load_all_experiments(output_base):
    """
    Scansiona la cartella output e legge
    config.json e metrics.json di ogni esperimento.
    """
    experiments = []

    for exp_name in sorted(os.listdir(output_base)):
        exp_dir = os.path.join(output_base, exp_name)

        # Salta se non è una cartella
        if not os.path.isdir(exp_dir):
            continue

        config_path  = os.path.join(exp_dir, "config.json")
        metrics_path = os.path.join(exp_dir, "metrics.json")

        # Salta se non ha i file necessari
        if not os.path.exists(config_path) or \
           not os.path.exists(metrics_path):
            continue

        with open(config_path)  as f: config  = json.load(f)
        with open(metrics_path) as f: metrics = json.load(f)

        experiments.append({**config, **metrics})
        print(f"  ✅ Caricato: {exp_name}")

    return experiments


# ══════════════════════════════════════════════════════════════════
# AGGIUNGI BASELINE DREAM5
# ══════════════════════════════════════════════════════════════════

def add_dream5_baselines(experiments):
    """
    Aggiunge le baseline dei metodi DREAM5
    lette dal file XLS nella sessione precedente.
    """
    dream5_baselines = [
        {
            "exp_name":    "DREAM5_Random",
            "model":       "Random",
            "aupr":        0.0144,   # baseline reale DREAM5: 4012/278392
            "auroc":       0.500,
            "improvement": 1.0,
        },
        {
            "exp_name":    "DREAM5_CLR",
            "model":       "CLR (Mutual Info)",
            "aupr":        0.255,
            "auroc":       0.773,
            "improvement": 2.80,
        },
        {
            "exp_name":    "DREAM5_GENIE3",
            "model":       "GENIE3 (Random Forest)",
            "aupr":        0.291,
            "auroc":       0.815,
            "improvement": 3.20,
        },
        {
            "exp_name":    "DREAM5_TIGRESS",
            "model":       "TIGRESS (Regression)",
            "aupr":        0.301,
            "auroc":       0.782,
            "improvement": 3.31,
        },
        {
            "exp_name":    "DREAM5_Best_Regression",
            "model":       "Best Regression DREAM5",
            "aupr":        0.313,
            "auroc":       0.764,
            "improvement": 3.44,
        },
        {
            "exp_name":    "DREAM5_Community",
            "model":       "Community (Ensemble)",
            "aupr":        0.327,
            "auroc":       0.809,
            "improvement": 3.60,
        },
    ]

    return experiments + dream5_baselines


# ══════════════════════════════════════════════════════════════════
# TABELLA COMPARATIVA
# ══════════════════════════════════════════════════════════════════

def print_comparison_table(df):
    """
    Stampa la tabella ordinata per AUPR decrescente.
    """
    cols = ['exp_name', 'model', 'aupr', 'auroc',
            'improvement', 'dropout', 'epochs', 'best_epoch']

    # Prendi solo le colonne disponibili
    available = [c for c in cols if c in df.columns]
    df_show   = df[available].sort_values('aupr', ascending=False)

    print("\n" + "═" * 80)
    print("  CONFRONTO ESPERIMENTI")
    print("═" * 80)
    print(df_show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("═" * 80)

    # Miglior esperimento
    best = df_show.iloc[0]
    print(f"\n  🏆 Miglior risultato: {best['exp_name']}")
    print(f"     AUPR = {best['aupr']:.4f}")
    print(f"     {best['improvement']:.1f}x rispetto al random")


# ══════════════════════════════════════════════════════════════════
# GRAFICO COMPARATIVO
# ══════════════════════════════════════════════════════════════════

def plot_comparison(df, output_base):
    """
    Genera un grafico a barre con AUPR di tutti gli esperimenti.
    Colori diversi per HGNN, GCN e baseline DREAM5.
    """
    df_sorted = df.sort_values('aupr', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.5)))

    colors = []
    for _, row in df_sorted.iterrows():
        model = str(row.get('model', ''))
        if 'HGNN_Edge_Hard_Stratified' in model:
            colors.append('darkviolet')
        elif 'HGNN_Edge_Hard_Random' in model:
            colors.append('orchid')
        elif 'GCN_Edge_Hard_Stratified' in model:
            colors.append('darkgreen')
        elif 'GCN_Edge_Hard_Random' in model:
            colors.append('limegreen')
        elif 'HGNN_Edge_Stratified' in model:
            colors.append('mediumpurple')
        elif 'HGNN_Edge_Random' in model:
            colors.append('plum')
        elif 'GCN_Edge_Stratified' in model:
            colors.append('mediumseagreen')
        elif 'GCN_Edge_Random' in model:
            colors.append('palegreen')
        elif 'HGNN_Statistical' in model or 'HGNN_Rosario' in model:
            colors.append('lightblue')
        elif 'GCN_Statistical' in model or 'GCN_Rosario' in model:
            colors.append('moccasin')
        elif 'HGNN' in model:
            colors.append('steelblue')
        elif 'GCN' in model:
            colors.append('darkorange')
        elif 'Community' in model:
            colors.append('darkgreen')
        elif 'Random' in model:
            colors.append('red')
        else:
            colors.append('gray')

    bars = ax.barh(df_sorted['exp_name'],
                   df_sorted['aupr'],
                   color=colors, alpha=0.85, edgecolor='white')

    # Aggiungi valori sulle barre
    for bar, val in zip(bars, df_sorted['aupr']):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    # Linea baseline random (1.44% = baseline DREAM5)
    random_aupr = df_sorted[df_sorted['exp_name'] == 'DREAM5_Random']['aupr']
    if len(random_aupr) > 0:
        ax.axvline(x=float(random_aupr.iloc[0]),
                   color='red', linestyle='--',
                   alpha=0.5,
                   label=f"Random baseline DREAM5 ({float(random_aupr.iloc[0]):.4f})")

    # Legenda colori
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue',      label='HGNN (gold standard)'),
        Patch(facecolor='darkorange',     label='GCN (gold standard)'),
        Patch(facecolor='lightblue',      label='HGNN (similarità - statistico)'),
        Patch(facecolor='moccasin',       label='GCN (similarità - statistico)'),
        Patch(facecolor='mediumpurple',   label='HGNN (edge pred - stratified)'),
        Patch(facecolor='plum',           label='HGNN (edge pred - random)'),
        Patch(facecolor='mediumseagreen', label='GCN (edge pred - stratified)'),
        Patch(facecolor='palegreen',      label='GCN (edge pred - random)'),
        Patch(facecolor='darkviolet',     label='HGNN (edge pred - hard stratified)'),
        Patch(facecolor='orchid',         label='HGNN (edge pred - hard random)'),
        Patch(facecolor='darkgreen',      label='GCN (edge pred - hard stratified)'),
        Patch(facecolor='limegreen',      label='GCN (edge pred - hard random)'),
        Patch(facecolor='gray',           label='DREAM5 methods'),
        Patch(facecolor='darkgreen',      label='DREAM5 Community'),
        Patch(facecolor='red',            label='Random'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlabel("AUPR")
    ax.set_title("Confronto AUPR — HGNN vs GCN vs DREAM5 Baselines")
    ax.set_xlim(0, max(df_sorted['aupr']) * 1.15)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_base, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Grafico salvato: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Confronta tutti gli esperimenti"
    )
    parser.add_argument("--output_base", type=str, default="./output",
                        help="Cartella base con tutti gli esperimenti")
    parser.add_argument("--no_dream5", action="store_true",
                        help="Non aggiungere le baseline DREAM5")
    args = parser.parse_args()

    print(f"\n  Scansiono cartella: {args.output_base}")
    experiments = load_all_experiments(args.output_base)

    if not experiments:
        print("\n  ❌ Nessun esperimento trovato!")
        print("     Assicurati di aver lanciato main.py con --exp_name")
        return

    # Aggiungi baseline DREAM5
    if not args.no_dream5:
        experiments = add_dream5_baselines(experiments)

    # Crea DataFrame
    df = pd.DataFrame(experiments)

    # Stampa tabella
    print_comparison_table(df)

    # Salva CSV
    csv_path = os.path.join(args.output_base, "experiments_summary.csv")
    df.sort_values('aupr', ascending=False).to_csv(csv_path, index=False)
    print(f"\n  💾 CSV salvato: {csv_path}")

    # Genera grafico
    plot_comparison(df, args.output_base)

    print("\n  ✅ Confronto completato!")


if __name__ == "__main__":
    main()
