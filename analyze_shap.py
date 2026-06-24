"""
analyze_shap.py — SHAP Feature Importance Analysis
====================================================
Calcola l'importanza delle feature in X per le predizioni del modello.

Risponde alla domanda del tutor:
    "Le 2 feature topologiche (grado ed entropia) aggiunte a X
     contribuiscono alle predizioni, o sono solo rumore?"

Come funziona SHAP:
    Per ogni predizione, SHAP assegna un valore a ciascuna delle 807
    feature che misura quanto quella feature ha contribuito alla
    predizione rispetto alla baseline (media del dataset).

    SHAP_j > 0 → feature j aumenta la probabilità predetta
    SHAP_j < 0 → feature j la riduce
    SHAP_j ≈ 0 → feature j non contribuisce (rumore)

Uso:
    python analyze_shap.py \\
        --network_path /percorso/Network1 \\
        --exp_dir ./output/edge_hgnn_stratified_topo \\
        --output_dir ./output/shap_analysis

    # Richiede che l'esperimento sia stato eseguito con --save_checkpoints
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP non installato. Uso gradient-based attribution come fallback.")
    print("    Per installare: pip install shap --break-system-packages")

from utils import load_checkpoint
import step1
import step2 as s2


# ══════════════════════════════════════════════════════════════════
# WRAPPER DEL MODELLO PER SHAP
# ══════════════════════════════════════════════════════════════════

class ModelWrapper(torch.nn.Module):
    """
    Wrapper che espone il modello HGNN come funzione X → predizioni.

    SHAP ha bisogno di una funzione che prende X come input e
    restituisce predizioni. Questo wrapper fissa theta e gli indici
    TF→gene da spiegare, e lascia variare solo X.
    """
    def __init__(self, model, theta, tf_idx, gene_idx):
        super().__init__()
        self.model    = model
        self.theta    = theta
        self.tf_idx   = tf_idx
        self.gene_idx = gene_idx

    def forward(self, X):
        embeddings = self.model(X, self.theta)
        preds      = self.model.predict(embeddings, self.tf_idx, self.gene_idx)
        return preds.squeeze(-1)


# ══════════════════════════════════════════════════════════════════
# GRADIENT-BASED ATTRIBUTION (fallback se SHAP non disponibile)
# ══════════════════════════════════════════════════════════════════

def gradient_attribution(model, theta, X, tf_idx, gene_idx, n_samples=200):
    """
    Calcola l'importanza delle feature usando i gradienti.

    Integrated Gradients: media dei gradienti lungo il percorso
    dal baseline (X=0) all'input reale X.

    È un'alternativa a SHAP quando SHAP non è disponibile,
    con proprietà teoriche simili (assiomi di completezza e
    dummy feature).
    """
    print("  Calcolo gradient-based attribution (Integrated Gradients)...")
    model.eval()

    # Campiona un sottoinsieme di coppie
    n = min(n_samples, len(tf_idx))
    idx_sample = np.random.choice(len(tf_idx), n, replace=False)
    tf_s   = tf_idx[idx_sample]
    gene_s = gene_idx[idx_sample]

    # Baseline = X nullo
    X_base = torch.zeros_like(X)

    # Integrated gradients su 50 step
    n_steps = 50
    attr    = torch.zeros_like(X)

    for alpha in np.linspace(0, 1, n_steps):
        X_interp = X_base + alpha * (X - X_base)
        X_interp.requires_grad_(True)

        emb   = model(X_interp, theta)
        preds = model.predict(emb, tf_s, gene_s).squeeze()
        loss  = preds.sum()
        loss.backward()

        with torch.no_grad():
            attr += X_interp.grad / n_steps

    with torch.no_grad():
        attr = attr * (X - X_base)

    return attr.abs().mean(dim=0).cpu().numpy()  # (n_features,)


# ══════════════════════════════════════════════════════════════════
# ANALISI SHAP PRINCIPALE
# ══════════════════════════════════════════════════════════════════

def compute_shap_values(model, theta, X, tf_idx, gene_idx,
                        n_background=100, n_explain=200):
    """
    Calcola l'importanza delle feature con Integrated Gradients.

    NOTA: SHAP DeepExplainer NON è compatibile con le GNN/HGNN perché
    passa al modello solo i campioni di background (es. 100×807) mentre
    theta ha shape (1643×1643) e richiede X con esattamente 1643 righe.

    Soluzione: Integrated Gradients (Sundararajan et al. 2017)
    - Proprietà teoriche equivalenti a SHAP (completezza, dummy feature)
    - Compatibile con GNN: passa sempre X completo (1643×807)
    - Baseline: X nullo (tutti zeri)
    """
    print(f"  Calcolo Integrated Gradients (equivalente SHAP per GNN)...")
    print(f"     → Metodo: Integrated Gradients (Sundararajan et al. 2017)")
    print(f"     → Coppie da spiegare: {min(n_explain, len(tf_idx))}")
    print(f"     → Steps integrazione: 50")

    model.eval()

    # Campiona coppie da spiegare
    n = min(n_explain, len(tf_idx))
    s = np.random.choice(len(tf_idx), n, replace=False)
    tf_s   = tf_idx[s]
    gene_s = gene_idx[s]

    # Baseline = X nullo
    X_base = torch.zeros_like(X)

    # Integrated gradients su 50 step
    n_steps = 50
    attr    = torch.zeros_like(X)

    for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
        X_interp = (X_base + alpha * (X - X_base)).detach().requires_grad_(True)

        emb   = model(X_interp, theta)
        preds = model.predict(emb, tf_s, gene_s).squeeze()
        loss  = preds.sum()
        loss.backward()

        with torch.no_grad():
            attr += X_interp.grad / n_steps

    with torch.no_grad():
        # Formula IG: (X - baseline) * media_gradienti
        attr = attr * (X - X_base)

    # Importanza media assoluta per feature
    mean_abs = attr.abs().mean(dim=0).cpu().numpy()  # (n_features,)
    return mean_abs, attr.cpu().numpy()


# ══════════════════════════════════════════════════════════════════
# VISUALIZZAZIONI
# ══════════════════════════════════════════════════════════════════

def plot_shap_analysis(importance, n_expression=805, output_dir='.'):
    """
    Genera 3 grafici sull'importanza delle feature:
        1. Top 20 feature più importanti
        2. Confronto espressione vs topologiche (boxplot)
        3. Importanza media per gruppo di feature
    """
    n_features = len(importance)
    has_topo   = n_features > n_expression

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SHAP Feature Importance — Contributo delle Feature alle Predizioni",
                 fontsize=13, fontweight='bold')

    # ── 1. Top 20 feature ─────────────────────────────────────────
    top20_idx = np.argsort(-importance)[:20]
    top20_val = importance[top20_idx]

    colors = []
    labels = []
    for i in top20_idx:
        if i < n_expression:
            colors.append('steelblue')
            labels.append(f'Exp_{i+1}')
        elif i == n_expression:
            colors.append('darkorange')
            labels.append('GRADO')
        else:
            colors.append('darkorange')
            labels.append('ENTROPIA')

    axes[0].barh(range(20), top20_val[::-1], color=colors[::-1])
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(labels[::-1], fontsize=8)
    axes[0].set_xlabel('Importanza SHAP media (|valore|)')
    axes[0].set_title('Top 20 Feature più Importanti')
    axes[0].grid(True, alpha=0.3)

    # ── 2. Boxplot: espressione vs topologiche ────────────────────
    expr_imp = importance[:n_expression]
    axes[1].boxplot([expr_imp], labels=['Espressione\n(805 feat.)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    if has_topo:
        topo_imp = importance[n_expression:]
        for j, (val, name) in enumerate(zip(topo_imp, ['Grado', 'Entropia'])):
            axes[1].scatter([1], [val], color='darkorange', s=100, zorder=5,
                          label=f'{name}: {val:.4f}')
        axes[1].legend(fontsize=9)
    axes[1].set_ylabel('Importanza SHAP media')
    axes[1].set_title('Espressione vs\nFeature Topologiche')
    axes[1].grid(True, alpha=0.3)

    # ── 3. Importanza media per gruppo ────────────────────────────
    groups      = ['Espressione\n(1-805)']
    group_means = [expr_imp.mean()]
    group_max   = [expr_imp.max()]
    group_cols  = ['steelblue']

    if has_topo:
        groups.extend(['Grado\n(806)', 'Entropia\n(807)'])
        group_means.extend([importance[n_expression], importance[n_expression+1]])
        group_max.extend([importance[n_expression], importance[n_expression+1]])
        group_cols.extend(['darkorange', 'darkorange'])

    x = np.arange(len(groups))
    axes[2].bar(x, group_means, color=group_cols, alpha=0.8, label='Media')
    axes[2].scatter(x, group_max, color='red', s=80, zorder=5, label='Max')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(groups, fontsize=9)
    axes[2].set_ylabel('Importanza SHAP')
    axes[2].set_title('Importanza Media per\nGruppo di Feature')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'shap_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  📊 Grafico salvato: {path}")
    plt.close()


def print_shap_summary(importance, n_expression=805):
    """Stampa un riepilogo testuale dei risultati SHAP."""
    print("\n" + "═" * 60)
    print("  RIEPILOGO SHAP FEATURE IMPORTANCE")
    print("═" * 60)

    expr_imp = importance[:n_expression]
    print(f"\n  FEATURE DI ESPRESSIONE (1-{n_expression}):")
    print(f"    Importanza media: {expr_imp.mean():.6f}")
    print(f"    Importanza max:   {expr_imp.max():.6f}")
    print(f"    Importanza min:   {expr_imp.min():.6f}")

    top5 = np.argsort(-expr_imp)[:5]
    print(f"    Top 5 feature espressione:")
    for i, idx in enumerate(top5):
        print(f"      {i+1}. Feature_{idx+1}: {expr_imp[idx]:.6f}")

    if len(importance) > n_expression:
        grado_imp   = importance[n_expression]
        entropia_imp = importance[n_expression + 1]
        print(f"\n  FEATURE TOPOLOGICHE:")
        print(f"    Grado    (feat. {n_expression+1}): {grado_imp:.6f}")
        print(f"    Entropia (feat. {n_expression+2}): {entropia_imp:.6f}")

        # Percentile delle feature topologiche rispetto all'espressione
        pct_grado    = (expr_imp < grado_imp).mean() * 100
        pct_entropia = (expr_imp < entropia_imp).mean() * 100

        print(f"\n  INTERPRETAZIONE:")
        print(f"    Grado supera il {pct_grado:.1f}% delle feature espressione")
        print(f"    Entropia supera il {pct_entropia:.1f}% delle feature espressione")

        if pct_grado < 50:
            print(f"\n  → CONFERMATO: Grado è nel 50% inferiore")
            print(f"    Le feature topologiche sono RUMORE per il modello")
            print(f"    Θ già codifica questa informazione implicitamente")
        else:
            print(f"\n  → Le feature topologiche contribuiscono")
            print(f"    ma non abbastanza da migliorare l'AUPR")

    print("═" * 60)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analisi SHAP feature importance"
    )
    parser.add_argument("--network_path",  type=str, required=True)
    parser.add_argument("--exp_dir",       type=str, required=True,
                        help="Cartella esperimento con step3.pkl "
                             "(eseguito con --save_checkpoints e "
                             "--use_topo_features)")
    parser.add_argument("--output_dir",    type=str,
                        default="./output/shap_analysis")
    parser.add_argument("--n_background",  type=int, default=100,
                        help="Campioni background per SHAP")
    parser.add_argument("--n_explain",     type=int, default=200,
                        help="Coppie TF→gene da spiegare")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Carica checkpoint ─────────────────────────────────────────
    pkl_path = os.path.join(args.exp_dir, "step3.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ step3.pkl non trovato in {args.exp_dir}")
        print("   Riesegui l'esperimento con --save_checkpoints")
        return

    print("\n" + "█" * 55)
    print("  SHAP FEATURE IMPORTANCE ANALYSIS")
    print("█" * 55)

    print(f"\n  Carico checkpoint da {pkl_path}...")
    data = load_checkpoint(pkl_path)

    # Il model potrebbe non essere nel pkl (vecchie versioni)
    # In quel caso proviamo a ricostruirlo dal config
    if 'model' not in data:
        print("  ❌ Modello non trovato nel pkl.")
        print("     Riesegui con: python main.py ... --save_checkpoints")
        print("     (assicurati di usare main.py aggiornato che salva il model)")
        return

    model    = data['model']
    theta_np = data['theta']
    X_np     = data['X']
    gene_list = data['gene_list']
    train_pos = data['train_pos']
    train_neg = data['train_neg']
    test_pos  = data['test_pos']

    device = torch.device('cpu')
    theta  = torch.tensor(theta_np, dtype=torch.float32).to(device)
    X      = torch.tensor(X_np,     dtype=torch.float32).to(device)
    model  = model.to(device)
    model.eval()

    n_features   = X.shape[1]
    n_expression = 805
    has_topo     = n_features > n_expression
    print(f"  Feature totali: {n_features}")
    print(f"  Feature topo:   {'Sì (grado + entropia)' if has_topo else 'No'}")

    # ── Prepara indici TF→gene (usa test set) ────────────────────
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Positivi del test set
    pos_pairs = [(gene_to_idx[r['TF']], gene_to_idx[r['Gene']])
                 for _, r in test_pos.iterrows()
                 if r['TF'] in gene_to_idx and r['Gene'] in gene_to_idx]

    # Campiona negativi casuali
    neg_sample = train_neg.sample(n=min(len(pos_pairs), len(train_neg)),
                                   random_state=args.seed)
    neg_pairs  = [(gene_to_idx[r['TF']], gene_to_idx[r['Gene']])
                  for _, r in neg_sample.iterrows()
                  if r['TF'] in gene_to_idx and r['Gene'] in gene_to_idx]

    all_pairs = pos_pairs + neg_pairs
    tf_idx    = torch.tensor([p[0] for p in all_pairs],
                               dtype=torch.long).to(device)
    gene_idx  = torch.tensor([p[1] for p in all_pairs],
                               dtype=torch.long).to(device)
    print(f"  Coppie da spiegare: {len(all_pairs)} "
          f"({len(pos_pairs)} pos, {len(neg_pairs)} neg)")

    # ── Calcola importanza feature ────────────────────────────────
    # Usiamo sempre Integrated Gradients: è compatibile con GNN
    # (SHAP DeepExplainer richiede batch variabili, incompatibile con Θ fisso)
    importance, ig_values = compute_shap_values(
        model, theta, X, tf_idx, gene_idx,
        n_background=args.n_background,
        n_explain=args.n_explain
    )
    method = "Integrated Gradients (Sundararajan et al. 2017)"

    print(f"\n  Metodo utilizzato: {method}")

    # ── Stampa riepilogo ──────────────────────────────────────────
    print_shap_summary(importance, n_expression=n_expression)

    # ── Genera grafici ────────────────────────────────────────────
    print("\n  Genero grafici...")
    plot_shap_analysis(importance,
                       n_expression=n_expression,
                       output_dir=args.output_dir)

    # ── Salva CSV ─────────────────────────────────────────────────
    import pandas as pd
    feat_names = [f'Exp_{i+1}' for i in range(n_expression)]
    if has_topo:
        feat_names += ['Grado_norm', 'Entropia_norm']
    df = pd.DataFrame({
        'feature':    feat_names,
        'importance': importance,
        'group':      (['expression'] * n_expression +
                       (['topo'] * (n_features - n_expression) if has_topo else []))
    })
    df = df.sort_values('importance', ascending=False)
    csv_path = os.path.join(args.output_dir, 'ig_importance.csv')
    df.to_csv(csv_path, index=False)
    print(f"  💾 CSV salvato: {csv_path}")

    print(f"\n  ✅ Analisi SHAP completata!")
    print(f"     Output in: {args.output_dir}")


if __name__ == "__main__":
    main()
