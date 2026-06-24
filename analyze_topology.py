"""
analyze_topology.py — Analisi topologica della rete HGNN
=========================================================
Implementa le 3 analisi richieste dal tutor:

1. ANALISI TOPOLOGICA PER NODO
   - Grado d(v): quanti TF regolano ogni gene
   - Entropia H(v): quanto è distribuita la regolazione
   - Identifica nodi critici (hub, geni altamente regolati)

2. ANALISI SPETTRALE DI Θ
   - Autovalori della matrice di propagazione
   - Spectral gap: misura la qualità della propagazione
   - Confronto tra H gold standard vs H edge prediction

3. CORRELAZIONE TOPOLOGIA / PERFORMANCE
   - Distribuzione del grado nei nodi mascherati (training vs test)
   - Differenza tra stratified e random masking
   - Quali nodi sono più difficili da classificare?

PARTE AVANZATA (richiede --save_checkpoints):
   - Confronto embeddings tra esperimenti diversi
   - Per attivare: rigira gli esperimenti con --save_checkpoints

Uso:
    python analyze_topology.py \\
        --network_path /percorso/Network1 \\
        --output_dir ./output/topology_analysis

    # Con confronto embeddings (richiede step3.pkl):
    python analyze_topology.py \\
        --network_path /percorso/Network1 \\
        --output_dir ./output/topology_analysis \\
        --exp_stratified ./output/edge_hgnn_stratified \\
        --exp_hard ./output/hgnn_edge_hard_stratified
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg

import step1
import step2 as s2


# ══════════════════════════════════════════════════════════════════
# 1. METRICHE TOPOLOGICHE PER NODO
# ══════════════════════════════════════════════════════════════════

def compute_node_degree(H):
    """
    Calcola il grado di ogni nodo (gene) in H.

    d(v_i) = numero di TF che regolano il gene i
           = somma della riga i di H

    Interpretazione biologica:
        d=0 → gene non regolato da nessun TF nel gold standard
        d=1 → gene regolato da 1 solo TF (specifico)
        d>5 → gene hub, regolato da molti TF (più ambiguo)
    """
    return H.sum(axis=1)  # (n_genes,)


def compute_node_entropy(H):
    """
    Calcola l'entropia topologica di ogni nodo (gene) in H.

    Per il gene i, se è regolato da k TF con pesi w_j:
        p_j = H[i][j] / sum(H[i,:])
        H(v_i) = -sum(p_j * log2(p_j))

    Interpretazione:
        H=0   → regolato da 1 solo TF (certezza massima)
        H=1   → regolato da 2 TF con peso uguale
        H alta → regolato da molti TF diversi (ambiguo)

    Connessione con l'hard negative:
        Geni con H alta sono più difficili da classificare
        perché la loro regolazione è distribuita tra molti TF.
        L'hard negative campiona geni simili → ancora più ambiguità.
    """
    degree = H.sum(axis=1, keepdims=True)
    # Evita divisione per zero per geni con grado 0
    degree_safe = np.where(degree > 0, degree, 1.0)
    P = H / degree_safe  # matrice delle probabilità (n_genes × n_tfs)

    # Entropia di Shannon per ogni gene
    # Maschera valori zero per evitare log(0)
    P_safe    = np.where(P > 0, P, 1.0)
    log_P     = np.log2(P_safe)
    entropy   = -np.sum(P * log_P * (P > 0), axis=1)

    # Geni con grado 0 → entropia = 0
    entropy[H.sum(axis=1) == 0] = 0.0
    return entropy  # (n_genes,)


def compute_tf_degree(H):
    """
    Calcola il grado di ogni TF (iperarco) in H.

    δ(e_j) = numero di geni regolati dal TF j
           = somma della colonna j di H
    """
    return H.sum(axis=0)  # (n_tfs,)


# ══════════════════════════════════════════════════════════════════
# 2. ANALISI SPETTRALE DI Θ
# ══════════════════════════════════════════════════════════════════

def compute_theta(H):
    """
    Calcola Θ = D_v^{-1/2} · H · W · D_e^{-1} · H^T · D_v^{-1/2}
    dalla matrice di incidenza H.
    """
    Dv         = H.sum(axis=1)
    Dv_invsqrt = np.where(Dv > 0, 1.0 / np.sqrt(Dv), 0.0)
    De         = H.sum(axis=0)
    De_inv     = np.where(De > 0, 1.0 / De, 0.0)

    # W = identità (pesi uniformi)
    # Θ = diag(Dv^{-1/2}) · H · diag(De^{-1}) · H^T · diag(Dv^{-1/2})
    DH   = H * Dv_invsqrt[:, None]         # (n×m)
    DHDe = DH * De_inv[None, :]             # (n×m)
    theta = DHDe @ H.T * Dv_invsqrt[:, None]  # (n×n)
    # Applica Dv^{-1/2} da sinistra
    theta = Dv_invsqrt[:, None] * theta
    return theta


def compute_spectral_properties(theta, n_eigenvalues=20):
    """
    Calcola le proprietà spettrali di Θ.

    Spectral gap = λ_1 - λ_2
    (differenza tra il primo e secondo autovalore più grande)

    Interpretazione:
        Gap grande → informazione si propaga velocemente
                     → embeddings più ricchi → AUPR migliore

        Gap piccolo → propagazione lenta
                      → embeddings meno informativi
    """
    print(f"     → Calcolo autovalori di Θ ({theta.shape})...")
    # Calcola solo i primi n_eigenvalues autovalori (più efficienti)
    # Usa linalg.eigvalsh per matrici simmetriche reali
    eigenvalues = linalg.eigvalsh(theta)
    eigenvalues = np.sort(eigenvalues)[::-1]  # ordine decrescente

    spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0

    print(f"     → λ_1 (max): {eigenvalues[0]:.4f}")
    print(f"     → λ_2      : {eigenvalues[1]:.4f}")
    print(f"     → Spectral gap: {spectral_gap:.4f}")
    print(f"     → λ_min    : {eigenvalues[-1]:.4f}")

    return {
        'eigenvalues':  eigenvalues,
        'spectral_gap': spectral_gap,
        'lambda_max':   eigenvalues[0],
        'lambda_2':     eigenvalues[1],
        'lambda_min':   eigenvalues[-1],
        'effective_rank': int(np.sum(eigenvalues > 0.01)),
    }


# ══════════════════════════════════════════════════════════════════
# 3. ANALISI MASKING — stratified vs random
# ══════════════════════════════════════════════════════════════════

def analyze_masking_topology(H, gene_list, tf_list,
                              gold_df, degree, entropy, seed=42):
    """
    Confronta le proprietà topologiche dei nodi mascherati
    nei due approcci (random vs stratified).

    Risponde alla domanda:
        "Stratified maschera nodi con topologia diversa
         rispetto a random?"

    Questo spiega perché stratified performa meglio:
        se maschera nodi più facili (basso grado/entropia)
        → test più semplice → AUPR più alta
    """
    pos = gold_df[gold_df['Label'] == 1].copy()

    # Random masking — test set
    pos_shuffled  = pos.sample(frac=1, random_state=seed)
    split         = int(len(pos_shuffled) * 0.8)
    random_train  = pos_shuffled.iloc[:split]
    random_test   = pos_shuffled.iloc[split:]

    # Stratified masking — test set
    rng = np.random.RandomState(seed)
    strat_train_list, strat_test_list = [], []
    for tf, group in pos.groupby('TF'):
        group  = group.sample(frac=1, random_state=rng.randint(0, 10000))
        n      = len(group)
        if n <= 2:
            strat_train_list.append(group)
            continue
        n_test = max(1, int(n * 0.2))
        strat_test_list.append(group.iloc[:n_test])
        strat_train_list.append(group.iloc[n_test:])

    strat_test = pd.concat(strat_test_list).reset_index(drop=True) \
                 if strat_test_list else pd.DataFrame()

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    def get_gene_metrics(df):
        """Estrai grado ed entropia dei geni nel dataset."""
        degrees   = []
        entropies = []
        for _, row in df.iterrows():
            gene = row['Gene']
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                degrees.append(degree[idx])
                entropies.append(entropy[idx])
        return np.array(degrees), np.array(entropies)

    rd, re = get_gene_metrics(random_test)
    sd, se = get_gene_metrics(strat_test)
    td, te = get_gene_metrics(random_train)

    return {
        'random_test_degree':     rd,
        'random_test_entropy':    re,
        'stratified_test_degree': sd,
        'stratified_test_entropy':se,
        'train_degree':           td,
        'train_entropy':          te,
        'n_random_test':          len(rd),
        'n_stratified_test':      len(sd),
        'n_train':                len(td),
    }


# ══════════════════════════════════════════════════════════════════
# 4. CONFRONTO EMBEDDINGS (opzionale — richiede step3.pkl)
# ══════════════════════════════════════════════════════════════════

def compare_embeddings(exp_dir_1, exp_dir_2, label_1, label_2):
    """
    Confronta gli embeddings di due esperimenti.
    Richiede che gli esperimenti siano stati eseguiti con
    --save_checkpoints.

    Ritorna:
        diff     : norma della differenza per ogni gene
        cosine   : similarità coseno per ogni gene
        sorted   : geni ordinati per differenza (più diversi prima)
    """
    from utils import load_checkpoint

    pkl1 = os.path.join(exp_dir_1, "step3.pkl")
    pkl2 = os.path.join(exp_dir_2, "step3.pkl")

    if not os.path.exists(pkl1) or not os.path.exists(pkl2):
        print(f"\n  ⚠️  Checkpoint non trovati per il confronto embeddings.")
        print(f"     Rigira gli esperimenti con --save_checkpoints")
        return None

    print(f"\n  Carico embeddings da {label_1}...")
    data1 = load_checkpoint(pkl1)
    emb1  = data1['embeddings']  # (n_genes × hidden_dim)

    print(f"  Carico embeddings da {label_2}...")
    data2 = load_checkpoint(pkl2)
    emb2  = data2['embeddings']

    # Normalizza per confronto equo
    norm1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    norm2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

    # Differenza euclidea per ogni gene
    diff   = np.linalg.norm(emb1 - emb2, axis=1)

    # Similarità coseno per ogni gene
    cosine = np.sum(norm1 * norm2, axis=1)

    print(f"\n  Differenza media embeddings: {diff.mean():.4f}")
    print(f"  Similarità coseno media:     {cosine.mean():.4f}")
    print(f"  Geni più diversi (top 10):")
    top10 = np.argsort(-diff)[:10]
    for idx in top10:
        gene = data1['gene_list'][idx] if 'gene_list' in data1 else f"G{idx}"
        print(f"    {gene}: diff={diff[idx]:.4f}, cos={cosine[idx]:.4f}")

    return {
        'diff':       diff,
        'cosine':     cosine,
        'embeddings1': emb1,
        'embeddings2': emb2,
        'gene_list':  data1.get('gene_list', []),
    }


# ══════════════════════════════════════════════════════════════════
# 5. VISUALIZZAZIONI
# ══════════════════════════════════════════════════════════════════

def plot_topology_analysis(degree, entropy, tf_degree,
                            spectral_gold, spectral_edge,
                            masking_stats, gene_list, tf_list,
                            output_dir):
    """
    Genera 6 grafici per l'analisi topologica.
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Analisi Topologica — Ipergrafo HGNN su DREAM5 Network 1",
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Distribuzione grado dei geni ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    d_nonzero = degree[degree > 0]
    ax1.hist(d_nonzero, bins=30, color='steelblue', alpha=0.8,
             edgecolor='white')
    ax1.axvline(d_nonzero.mean(), color='red', linestyle='--',
                label=f'Media = {d_nonzero.mean():.1f}')
    ax1.set_xlabel("Grado d(v) — n° TF regolatori")
    ax1.set_ylabel("N° geni")
    ax1.set_title("Distribuzione del grado\nper gene (nodi con d>0)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Distribuzione entropia topologica ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    e_nonzero = entropy[entropy > 0]
    ax2.hist(e_nonzero, bins=30, color='darkorange', alpha=0.8,
             edgecolor='white')
    ax2.axvline(e_nonzero.mean(), color='red', linestyle='--',
                label=f'Media = {e_nonzero.mean():.2f}')
    ax2.set_xlabel("Entropia H(v) [bit]")
    ax2.set_ylabel("N° geni")
    ax2.set_title("Distribuzione entropia\ntopologica per gene")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── 3. Grado vs Entropia (scatter) ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    mask = degree > 0
    sc = ax3.scatter(degree[mask], entropy[mask],
                     alpha=0.3, s=10, c=degree[mask],
                     cmap='viridis')
    plt.colorbar(sc, ax=ax3, label='Grado')
    ax3.set_xlabel("Grado d(v)")
    ax3.set_ylabel("Entropia H(v) [bit]")
    ax3.set_title("Grado vs Entropia\nper gene")
    ax3.grid(True, alpha=0.3)

    # ── 4. Distribuzione grado TF ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(tf_degree, bins=20, color='mediumpurple', alpha=0.8,
             edgecolor='white')
    ax4.axvline(tf_degree.mean(), color='red', linestyle='--',
                label=f'Media = {tf_degree.mean():.1f}')
    ax4.set_xlabel("Grado δ(e) — n° geni target")
    ax4.set_ylabel("N° TF")
    ax4.set_title("Distribuzione del grado\nper TF (iperarchi)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. Autovalori di Θ — confronto Gold vs Edge ───────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    n_show = min(50, len(spectral_gold['eigenvalues']))
    x      = np.arange(1, n_show + 1)
    ax5.plot(x, spectral_gold['eigenvalues'][:n_show],
             'steelblue', linewidth=2, marker='o', markersize=3,
             label=f"H gold standard\n(gap={spectral_gold['spectral_gap']:.4f})")
    ax5.plot(x, spectral_edge['eigenvalues'][:n_show],
             'darkorange', linewidth=2, marker='s', markersize=3,
             label=f"H edge prediction (100%)\n(gap={spectral_edge['spectral_gap']:.4f})")
    ax5.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax5.set_xlabel("Indice autovalore (ordine decrescente)")
    ax5.set_ylabel("Valore autovalore λ")
    ax5.set_title("Spettro di Θ — Gold standard vs Edge prediction")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── 6. Grado nodi mascherati: Random vs Stratified ────────────
    ax6 = fig.add_subplot(gs[2, :2])
    bins = np.arange(0, max(
        masking_stats['random_test_degree'].max(),
        masking_stats['stratified_test_degree'].max()
    ) + 2) - 0.5

    ax6.hist(masking_stats['random_test_degree'], bins=bins,
             alpha=0.6, color='plum', edgecolor='white',
             label=f"Random test "
                   f"(media={masking_stats['random_test_degree'].mean():.1f})")
    ax6.hist(masking_stats['stratified_test_degree'], bins=bins,
             alpha=0.6, color='mediumpurple', edgecolor='white',
             label=f"Stratified test "
                   f"(media={masking_stats['stratified_test_degree'].mean():.1f})")
    ax6.set_xlabel("Grado d(v) del gene mascherato")
    ax6.set_ylabel("N° archi mascherati")
    ax6.set_title("Distribuzione grado geni mascherati\nRandom vs Stratified")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── 7. Entropia nodi mascherati: Random vs Stratified ─────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.hist(masking_stats['random_test_entropy'], bins=20,
             alpha=0.6, color='plum', edgecolor='white',
             label="Random")
    ax7.hist(masking_stats['stratified_test_entropy'], bins=20,
             alpha=0.6, color='mediumpurple', edgecolor='white',
             label="Stratified")
    ax7.set_xlabel("Entropia H(v) [bit]")
    ax7.set_ylabel("N° archi mascherati")
    ax7.set_title("Entropia geni mascherati\nRandom vs Stratified")
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "topology_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 Grafico salvato: {path}")
    plt.close()


def plot_embedding_comparison(emb_data, degree, entropy,
                               label_1, label_2, output_dir):
    """Grafici confronto embeddings (solo se disponibili)."""
    if emb_data is None:
        return

    diff   = emb_data['diff']
    cosine = emb_data['cosine']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Confronto Embeddings: {label_1} vs {label_2}",
                 fontsize=12, fontweight='bold')

    # Distribuzione differenze
    axes[0].hist(diff, bins=50, color='steelblue', alpha=0.8,
                 edgecolor='white')
    axes[0].set_xlabel("Differenza euclidea")
    axes[0].set_ylabel("N° geni")
    axes[0].set_title("Distribuzione differenze\nembeddings")
    axes[0].grid(True, alpha=0.3)

    # Differenza vs Grado
    mask = degree > 0
    axes[1].scatter(degree[mask], diff[mask],
                    alpha=0.3, s=10, c='darkorange')
    axes[1].set_xlabel("Grado d(v)")
    axes[1].set_ylabel("Differenza embedding")
    axes[1].set_title("Grado vs Differenza\nembedding")
    axes[1].grid(True, alpha=0.3)

    # Differenza vs Entropia
    axes[2].scatter(entropy[mask], diff[mask],
                    alpha=0.3, s=10, c='mediumpurple')
    axes[2].set_xlabel("Entropia H(v) [bit]")
    axes[2].set_ylabel("Differenza embedding")
    axes[2].set_title("Entropia vs Differenza\nembedding")
    axes[2].grid(True, alpha=0.3)

    path = os.path.join(output_dir, "embedding_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 Confronto embeddings salvato: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════
# 6. STAMPA RIEPILOGO
# ══════════════════════════════════════════════════════════════════

def print_summary(degree, entropy, tf_degree,
                  spectral_gold, spectral_edge,
                  masking_stats, gene_list):
    """Stampa un riepilogo testuale dei risultati."""

    print("\n" + "═" * 65)
    print("  RIEPILOGO ANALISI TOPOLOGICA")
    print("═" * 65)

    # Statistiche nodi
    d_nz = degree[degree > 0]
    e_nz = entropy[entropy > 0]
    print(f"\n  GENI (nodi):")
    print(f"    Totali:           {len(gene_list)}")
    print(f"    Con almeno 1 TF:  {len(d_nz)}")
    print(f"    Non regolati:     {(degree==0).sum()}")
    print(f"    Grado medio:      {d_nz.mean():.2f}")
    print(f"    Grado max:        {d_nz.max():.0f}")
    print(f"    Entropia media:   {e_nz.mean():.3f} bit")
    print(f"    Entropia max:     {e_nz.max():.3f} bit")

    # Top 10 geni più regolati
    top10_deg = np.argsort(-degree)[:10]
    print(f"\n  TOP 10 GENI PIÙ REGOLATI (grado più alto):")
    for idx in top10_deg:
        g = gene_list[idx]
        print(f"    {g:<10} d={degree[idx]:.0f}  H={entropy[idx]:.3f} bit")

    # Statistiche TF
    print(f"\n  TF (iperarchi):")
    print(f"    Totali:           {len(tf_degree)}")
    print(f"    Grado medio:      {tf_degree.mean():.1f} geni/TF")
    print(f"    Grado max:        {tf_degree.max():.0f} geni/TF")
    print(f"    Grado min (>0):   {tf_degree[tf_degree>0].min():.0f} geni/TF")

    # Analisi spettrale
    print(f"\n  ANALISI SPETTRALE DI Θ:")
    print(f"    Gold standard:")
    print(f"      λ_1 = {spectral_gold['lambda_max']:.4f}")
    print(f"      λ_2 = {spectral_gold['lambda_2']:.4f}")
    print(f"      Spectral gap = {spectral_gold['spectral_gap']:.4f}")
    print(f"      Rank effettivo = {spectral_gold['effective_rank']}")
    print(f"    Edge prediction (100%):")
    print(f"      λ_1 = {spectral_edge['lambda_max']:.4f}")
    print(f"      λ_2 = {spectral_edge['lambda_2']:.4f}")
    print(f"      Spectral gap = {spectral_edge['spectral_gap']:.4f}")
    print(f"      Rank effettivo = {spectral_edge['effective_rank']}")

    # Confronto masking
    rd = masking_stats['random_test_degree']
    sd = masking_stats['stratified_test_degree']
    re = masking_stats['random_test_entropy']
    se = masking_stats['stratified_test_entropy']
    print(f"\n  CONFRONTO MASKING:")
    print(f"    Random test:     {len(rd)} archi")
    print(f"      Grado medio:   {rd.mean():.2f}")
    print(f"      Entropia media:{re.mean():.3f} bit")
    print(f"    Stratified test: {len(sd)} archi")
    print(f"      Grado medio:   {sd.mean():.2f}")
    print(f"      Entropia media:{se.mean():.3f} bit")

    if rd.mean() != sd.mean():
        diff_pct = (sd.mean() - rd.mean()) / rd.mean() * 100
        print(f"    Differenza grado: {diff_pct:+.1f}%")
        if abs(diff_pct) > 5:
            print(f"    → Stratified maschera geni con grado diverso!")

    print("═" * 65)


# ══════════════════════════════════════════════════════════════════
# 7. SALVA CSV RISULTATI PER NODO
# ══════════════════════════════════════════════════════════════════

def save_node_metrics(gene_list, tf_list, degree, entropy,
                      tf_degree, output_dir):
    """Salva le metriche per nodo in un CSV."""
    df_genes = pd.DataFrame({
        'gene':    gene_list,
        'degree':  degree,
        'entropy': entropy,
    })
    df_genes = df_genes.sort_values('degree', ascending=False)

    df_tfs = pd.DataFrame({
        'tf':     tf_list,
        'degree': tf_degree,
    })
    df_tfs = df_tfs.sort_values('degree', ascending=False)

    path_genes = os.path.join(output_dir, "node_metrics.csv")
    path_tfs   = os.path.join(output_dir, "tf_metrics.csv")

    df_genes.to_csv(path_genes, index=False)
    df_tfs.to_csv(path_tfs, index=False)

    print(f"  💾 Metriche nodi salvate: {path_genes}")
    print(f"  💾 Metriche TF salvate:   {path_tfs}")
    return df_genes, df_tfs


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analisi topologica della rete HGNN"
    )
    parser.add_argument("--network_path",    type=str, required=True)
    parser.add_argument("--output_dir",      type=str,
                        default="./output/topology_analysis")
    parser.add_argument("--seed",            type=int, default=42)
    # Opzionale: confronto embeddings
    parser.add_argument("--exp_stratified",  type=str, default=None,
                        help="Cartella esperimento stratified (per confronto "
                             "embeddings, richiede step3.pkl)")
    parser.add_argument("--exp_hard",        type=str, default=None,
                        help="Cartella esperimento hard negative (per confronto "
                             "embeddings, richiede step3.pkl)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("\n" + "█" * 60)
    print("  ANALISI TOPOLOGICA — HGNN su DREAM5 Network 1")
    print("█" * 60)

    # ── Carica dati ───────────────────────────────────────────────
    print("\n  [1/7] Carico dati DREAM5...")
    data_step1 = step1.run(args.network_path)
    gene_list  = data_step1['gene_list']
    tf_list    = data_step1['tf_list']
    gold_df    = data_step1['gold_df']

    # ── Costruisci H gold standard (80%) ──────────────────────────
    print("\n  [2/7] Costruisco H gold standard (80%)...")
    data_gold = s2.run(data_step1,
                       structure='gold_standard',
                       model_type='hgnn',
                       seed=args.seed)
    H_gold = data_gold['H']

    # ── Costruisci H edge prediction (100%) ───────────────────────
    print("\n  [3/7] Costruisco H edge prediction (100%)...")
    data_edge = s2.run(data_step1,
                       structure='edge_prediction',
                       model_type='hgnn',
                       masking='stratified',
                       seed=args.seed)
    H_edge = data_edge['H']

    # ── Calcola metriche topologiche ──────────────────────────────
    print("\n  [4/7] Calcolo metriche topologiche per nodo...")
    degree    = compute_node_degree(H_edge)   # usa H completa
    entropy   = compute_node_entropy(H_edge)
    tf_degree = compute_tf_degree(H_edge)

    print(f"     → Grado medio per gene: {degree[degree>0].mean():.2f}")
    print(f"     → Entropia media:       {entropy[entropy>0].mean():.3f} bit")
    print(f"     → Grado medio per TF:   {tf_degree.mean():.1f}")

    # ── Analisi spettrale ─────────────────────────────────────────
    print("\n  [5/7] Analisi spettrale di Θ...")
    print("     → Gold standard H:")
    theta_gold   = compute_theta(H_gold)
    spectral_gold = compute_spectral_properties(theta_gold)

    print("     → Edge prediction H (100%):")
    theta_edge   = compute_theta(H_edge)
    spectral_edge = compute_spectral_properties(theta_edge)

    # ── Analisi masking ───────────────────────────────────────────
    print("\n  [6/7] Analisi topologica del masking...")
    masking_stats = analyze_masking_topology(
        H_edge, gene_list, tf_list, gold_df,
        degree, entropy, seed=args.seed
    )

    # ── Stampa riepilogo ──────────────────────────────────────────
    print_summary(degree, entropy, tf_degree,
                  spectral_gold, spectral_edge,
                  masking_stats, gene_list)

    # ── Genera grafici ────────────────────────────────────────────
    print("\n  [7/7] Genero grafici...")
    plot_topology_analysis(
        degree, entropy, tf_degree,
        spectral_gold, spectral_edge,
        masking_stats, gene_list, tf_list,
        args.output_dir
    )

    # ── Salva CSV ─────────────────────────────────────────────────
    save_node_metrics(gene_list, tf_list, degree, entropy,
                      tf_degree, args.output_dir)

    # ── Confronto embeddings (opzionale) ──────────────────────────
    if args.exp_stratified and args.exp_hard:
        print("\n  [EXTRA] Confronto embeddings stratified vs hard...")
        emb_data = compare_embeddings(
            args.exp_stratified,
            args.exp_hard,
            "stratified",
            "hard_stratified"
        )
        if emb_data:
            plot_embedding_comparison(
                emb_data, degree, entropy,
                "Stratified", "Hard Stratified",
                args.output_dir
            )
    else:
        print("\n  ℹ️  Confronto embeddings non disponibile.")
        print("     Per attivarlo, rigira gli esperimenti con:")
        print("     python main.py ... --analysis hgnn_edge_stratified "
              "--exp_name edge_hgnn_stratified --save_checkpoints")
        print("     python main.py ... --analysis hgnn_edge_hard_stratified "
              "--exp_name hgnn_edge_hard_stratified --save_checkpoints")
        print("     Poi rilancia con:")
        print("     --exp_stratified ./output/edge_hgnn_stratified "
              "--exp_hard ./output/hgnn_edge_hard_stratified")

    print(f"\n  ✅ Analisi topologica completata!")
    print(f"     Output in: {args.output_dir}")
    print(f"\n  File prodotti:")
    print(f"    topology_analysis.png  ← 7 grafici topologici")
    print(f"    node_metrics.csv       ← metriche per gene")
    print(f"    tf_metrics.csv         ← metriche per TF")


if __name__ == "__main__":
    main()
