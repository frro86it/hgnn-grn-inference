"""
analyze_sparsity.py — Analisi e visualizzazione della sparsità
===============================================================
Confronta la densità delle 4 strutture H:
    1. HGNN dal gold standard  (1643 × 195)
    2. GCN  dal gold standard  (1643 × 1643)
    3. GCN  dalla similarità   (1643 × 1643)
    4. HGNN dalla similarità   (1643 × 1643)

Genera:
    1. sparsity_comparison.png  → grafici a barre
    2. matrix_heatmaps.png      → matrici ridimensionate a 200×200

Uso:
    python analyze_sparsity.py \
        --network_path /percorso/Network1 \
        --k 10 \
        --output_dir ./output/sparsity_analysis
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import step1
import step2 as s2


# ══════════════════════════════════════════════════════════════════
# STATISTICHE DI SPARSITÀ
# ══════════════════════════════════════════════════════════════════

def compute_sparsity_stats(matrix, name):
    """Calcola le statistiche di sparsità di una matrice."""
    n_rows, n_cols   = matrix.shape
    total_cells      = n_rows * n_cols
    n_edges          = int((matrix > 0).sum())
    density          = n_edges / total_cells * 100
    edges_per_node   = n_edges / n_rows
    degree           = (matrix > 0).sum(axis=1)

    return {
        "nome":          name,
        "forma":         f"{n_rows} × {n_cols}",
        "celle_totali":  total_cells,
        "connessioni":   n_edges,
        "densita_pct":   round(density, 3),
        "conn_per_nodo": round(edges_per_node, 1),
        "grado_medio":   round(float(degree.mean()), 1),
        "grado_max":     int(degree.max()),
    }


# ══════════════════════════════════════════════════════════════════
# RIDIMENSIONAMENTO MATRICE
# ══════════════════════════════════════════════════════════════════

def resize_matrix(matrix, target_size=200):
    """
    Ridimensiona la matrice a target_size × target_size
    usando media a blocchi — ogni pixel = media di un blocco.
    Mostra TUTTA la struttura reale, non un campione casuale.
    """
    n_rows, n_cols = matrix.shape
    out = np.zeros((target_size, target_size), dtype=np.float32)

    for i in range(target_size):
        for j in range(target_size):
            r_start = int(i * n_rows / target_size)
            r_end   = int((i + 1) * n_rows / target_size)
            c_start = int(j * n_cols / target_size)
            c_end   = int((j + 1) * n_cols / target_size)
            block   = matrix[r_start:r_end, c_start:c_end]
            out[i][j] = block.mean() if block.size > 0 else 0

    return out


# ══════════════════════════════════════════════════════════════════
# GRAFICO A BARRE — SPARSITÀ
# ══════════════════════════════════════════════════════════════════

def plot_sparsity_comparison(stats_list, output_dir):
    """Genera 3 grafici a barre + tabella riepilogativa."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Analisi della Sparsità — Confronto strutture",
                 fontsize=14, fontweight='bold', y=0.98)

    nomi        = [s['nome'] for s in stats_list]
    connessioni = [s['connessioni'] for s in stats_list]
    densita     = [s['densita_pct'] for s in stats_list]
    conn_nodo   = [s['conn_per_nodo'] for s in stats_list]

    colori_map = {
        'HGNN\n(gold standard)': 'steelblue',
        'GCN\n(gold standard)':  'darkorange',
        'GCN\n(similarità)':     'moccasin',
        'HGNN\n(similarità)':    'lightblue',
    }
    colors = [colori_map.get(n, 'gray') for n in nomi]

    # ── Grafico 1: Connessioni totali ─────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    bars = ax1.bar(nomi, connessioni, color=colors,
                   edgecolor='gray', linewidth=0.5)
    for bar, val in zip(bars, connessioni):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 200,
                 f'{val:,}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax1.set_title("Numero di connessioni totali in H", fontweight='bold')
    ax1.set_ylabel("Connessioni")
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(connessioni) * 1.2)

    # ── Grafico 2: Densità % ───────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    bars2 = ax2.bar(nomi, densita, color=colors,
                    edgecolor='gray', linewidth=0.5)
    for bar, val in zip(bars2, densita):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.3f}%', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax2.set_title("Densità della matrice H (%)", fontweight='bold')
    ax2.set_ylabel("Densità (%)")
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max(densita) * 1.3)

    # ── Grafico 3: Connessioni per nodo ───────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    bars3 = ax3.bar(nomi, conn_nodo, color=colors,
                    edgecolor='gray', linewidth=0.5)
    for bar, val in zip(bars3, conn_nodo):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax3.set_title("Connessioni medie per nodo (gene)", fontweight='bold')
    ax3.set_ylabel("Connessioni / nodo")
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(conn_nodo) * 1.2)

    # ── Grafico 4: Tabella riepilogativa ──────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    table_data = []
    for s in stats_list:
        table_data.append([
            s['nome'].replace('\n', ' '),
            s['forma'],
            f"{s['connessioni']:,}",
            f"{s['densita_pct']:.3f}%",
            f"{s['conn_per_nodo']:.1f}",
        ])
    col_labels = ['Approccio', 'Forma H', 'Connessioni', 'Densità', 'Conn/nodo']
    table = ax4.table(cellText=table_data, colLabels=col_labels,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    ax4.set_title("Tabella riepilogativa", fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(output_dir, "sparsity_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 Grafico sparsità salvato: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# HEATMAP MATRICI RIDIMENSIONATE
# ══════════════════════════════════════════════════════════════════

def plot_matrix_heatmaps(matrices_dict, output_dir, target_size=200):
    """
    Mostra le 4 matrici H ridimensionate a target_size × target_size.
    Ogni pixel = media di un blocco di celle originali.
    Chiaro = sparso, Scuro = denso.
    """
    n    = len(matrices_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(
        f"Struttura completa delle matrici H\n"
        f"(ridimensionate a {target_size}×{target_size} pixel — "
        f"ogni pixel = media di un blocco)\n"
        f"Chiaro = sparso, Scuro = denso",
        fontsize=11, fontweight='bold'
    )

    for ax, (name, matrix) in zip(axes, matrices_dict.items()):
        resized  = resize_matrix(matrix > 0, target_size=target_size)
        n_conn   = int((matrix > 0).sum())
        density  = n_conn / (matrix.shape[0] * matrix.shape[1]) * 100

        im = ax.imshow(resized, cmap='Blues', aspect='auto',
                       interpolation='nearest',
                       vmin=0, vmax=resized.max())
        ax.set_title(
            f"{name}\nconnessioni: {n_conn:,}\ndensità: {density:.3f}%",
            fontsize=10
        )
        ax.set_xlabel(f"Colonne → {matrix.shape[1]}")
        ax.set_ylabel(f"Righe → {matrix.shape[0]}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Densità locale')

    plt.tight_layout()
    path = os.path.join(output_dir, "matrix_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 Heatmap matrici salvata: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# STAMPA SPIEGAZIONE
# ══════════════════════════════════════════════════════════════════

def print_explanation(stats_list, n_genes):
    print("\n" + "═" * 62)
    print("  SPIEGAZIONE DEI NUMERI DI SPARSITÀ")
    print("═" * 62)
    print(f"\n  Numero totale di geni : {n_genes}")
    print(f"  Celle totali (quadrata): "
          f"{n_genes} × {n_genes} = {n_genes*n_genes:,}")
    print("\n" + "─" * 62)
    for s in stats_list:
        nome = s['nome'].replace('\n', ' ')
        print(f"\n  [{nome}]")
        print(f"  Forma matrice  : {s['forma']}")
        print(f"  Connessioni    : {s['connessioni']:,}")
        print(f"  Densità        : {s['connessioni']:,} / "
              f"{s['celle_totali']:,} × 100 = {s['densita_pct']:.3f}%")
        print(f"  Conn per nodo  : "
              f"{s['connessioni']:,} / {n_genes} = {s['conn_per_nodo']:.1f}")
    print("\n" + "─" * 62)
    print("\n  INTERPRETAZIONE:")
    print("  Più connessioni → ogni gene riceve informazione")
    print("  da più vicini durante la propagazione HGNN/GCN")
    print("  → embedding più ricchi → AUPR più alta sul test set")
    print("═" * 62)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analisi sparsità strutture H"
    )
    parser.add_argument("--network_path", type=str, required=True)
    parser.add_argument("--k",            type=int, default=10)
    parser.add_argument("--output_dir",   type=str,
                        default="./output/sparsity_analysis")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("\n" + "█" * 55)
    print("  ANALISI SPARSITÀ STRUTTURE H")
    print("█" * 55)

    # ── Carica dati ───────────────────────────────────────────────
    print("\n  Carico dati DREAM5...")
    data_step1 = step1.run(args.network_path)
    n_genes    = len(data_step1['gene_list'])

    # ── Costruisci le 4 strutture con nuovo step2 unificato ───────
    print("\n  Costruisco le 4 strutture...")

    print("\n  [1/4] HGNN dal gold standard...")
    d1 = s2.run(data_step1, structure='gold_standard',
                model_type='hgnn', seed=args.seed)
    H_hgnn_gold = d1['H']

    print("\n  [2/4] GCN dal gold standard...")
    d2 = s2.run(data_step1, structure='gold_standard',
                model_type='gcn', seed=args.seed)
    A_gcn_gold = d2['A']

    print("\n  [3/4] GCN dalla similarità genetica...")
    d3 = s2.run(data_step1, structure='statistical',
                model_type='gcn', k=args.k, seed=args.seed)
    A_gcn_stat = d3['A']

    print("\n  [4/4] HGNN dalla similarità genetica...")
    d4 = s2.run(data_step1, structure='statistical',
                model_type='hgnn', k=args.k, seed=args.seed)
    H_hgnn_stat = d4['H']

    # ── Calcola statistiche ───────────────────────────────────────
    print("\n  Calcolo statistiche di sparsità...")
    stats = [
        compute_sparsity_stats(H_hgnn_gold, 'HGNN\n(gold standard)'),
        compute_sparsity_stats(A_gcn_gold,  'GCN\n(gold standard)'),
        compute_sparsity_stats(A_gcn_stat,  'GCN\n(similarità)'),
        compute_sparsity_stats(H_hgnn_stat, 'HGNN\n(similarità)'),
    ]

    # ── Stampa tabella ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  TABELLA SPARSITÀ")
    print("═" * 70)
    print(f"  {'Approccio':<25} {'Forma':>12} "
          f"{'Connessioni':>12} {'Densità':>10} {'Conn/nodo':>10}")
    print("  " + "─" * 68)
    for s in stats:
        nome = s['nome'].replace('\n', ' ')
        print(f"  {nome:<25} {s['forma']:>12} "
              f"{s['connessioni']:>12,} "
              f"{s['densita_pct']:>9.3f}% "
              f"{s['conn_per_nodo']:>10.1f}")
    print("═" * 70)

    # ── Spiegazione ───────────────────────────────────────────────
    print_explanation(stats, n_genes)

    # ── Grafici ───────────────────────────────────────────────────
    print("\n  Genero grafici...")
    plot_sparsity_comparison(stats, args.output_dir)

    matrices = {
        'HGNN\n(gold)':       H_hgnn_gold,
        'GCN\n(gold)':        A_gcn_gold,
        'GCN\n(similarità)':  A_gcn_stat,
        'HGNN\n(similarità)': H_hgnn_stat,
    }
    plot_matrix_heatmaps(matrices, args.output_dir, target_size=200)

    print("\n  ✅ Analisi sparsità completata!")
    print(f"     Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
