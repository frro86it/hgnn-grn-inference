"""
step2.py — Costruzione struttura (UNIFICATO)
============================================
Gestisce tutti i tipi di struttura con un unico file.

STRUTTURE DISPONIBILI:
    gold_standard   → H o A dall'80% degli archi del gold standard
    statistical     → H o A dalla similarità genetica (approccio ibrido)
    edge_prediction → H completa (100% archi) + masking

MASKING DISPONIBILI (solo per edge_prediction):
    random           → positivi a caso + negativi random
    stratified       → positivi per TF + negativi random
    hard_random      → positivi a caso + negativi HARD (simili ai pos)
    hard_stratified  → positivi per TF + negativi HARD (simili ai pos)

OPZIONE B — Separazione corretta dei negativi:
    train_neg (80%) → campionamento durante training (se non hard)
    test_neg  (20%) → usati SOLO per il test
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import save_checkpoint, load_checkpoint
import step1


# ══════════════════════════════════════════════════════════════════
# NORMALIZZAZIONE
# ══════════════════════════════════════════════════════════════════

def normalize_expression(expression_df):
    """Normalizza expression data con Z-score."""
    print("\n  Normalizzo expression data (Z-score)...")
    X    = expression_df.values.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0,  keepdims=True)
    std[std == 0] = 1.0
    X_norm = ((X - mean) / std).T
    print(f"     → Feature per nodo: {X_norm.shape[1]}")
    print(f"     → Nodi totali:      {X_norm.shape[0]}")
    return X_norm


# ══════════════════════════════════════════════════════════════════
# FEATURE TOPOLOGICHE (opzionale)
# ══════════════════════════════════════════════════════════════════

def add_topo_features(X_norm, H):
    """
    Aggiunge feature topologiche alle feature di espressione.

    Per ogni gene aggiunge 2 colonne a X_norm:
        colonna -2: grado normalizzato d(v) / max_degree
                    "quanti TF regolano questo gene?"
        colonna -1: entropia normalizzata H(v) / max_entropy
                    "quanto è distribuita la regolazione?"

    X_old: (n_geni × 805)
    X_new: (n_geni × 807)

    Perché normalizzate?
        Grado max = 12, entropia max = 3.585 bit
        Se non normalizziamo queste feature
        dominano i 805 valori di espressione
        che sono già in scala Z-score (media=0, std=1)

    Attivata solo con --use_topo_features
    → di default non viene chiamata
    → tutti i vecchi esperimenti rimangono invariati
    """
    # Grado per ogni gene
    degree  = H.sum(axis=1).astype(np.float32)  # (n_geni,)

    # Entropia per ogni gene
    deg_safe  = np.where(degree > 0, degree, 1.0)
    P         = H / deg_safe[:, None]
    P_safe    = np.where(P > 0, P, 1.0)
    entropy   = -np.sum(P * np.log2(P_safe) * (P > 0), axis=1)
    entropy   = entropy.astype(np.float32)
    entropy[degree == 0] = 0.0

    # Normalizza tra 0 e 1
    max_deg  = degree.max()  if degree.max()  > 0 else 1.0
    max_ent  = entropy.max() if entropy.max() > 0 else 1.0
    deg_norm = (degree  / max_deg)[:, None]   # (n_geni × 1)
    ent_norm = (entropy / max_ent)[:, None]   # (n_geni × 1)

    X_new = np.concatenate([X_norm, deg_norm, ent_norm], axis=1)

    print(f"     → Feature aggiunte: grado e entropia normalizzati")
    print(f"     → X shape: {X_norm.shape} → {X_new.shape}")
    return X_new


# ══════════════════════════════════════════════════════════════════
# OPZIONE B — separazione negativi
# ══════════════════════════════════════════════════════════════════

def split_negatives(gold_df, test_ratio=0.2, seed=42):
    """
    Divide i negativi in train_neg (80%) e test_neg (20%).
    Garantisce zero sovrapposizione tra training e test.
    """
    neg = gold_df[gold_df['Label'] == 0].copy()
    neg = neg.sample(frac=1, random_state=seed).reset_index(drop=True)

    split     = int(len(neg) * (1 - test_ratio))
    train_neg = neg.iloc[:split]
    test_neg  = neg.iloc[split:]

    print(f"     → Negativi train : {len(train_neg)}")
    print(f"     → Negativi test  : {len(test_neg)}")
    return train_neg, test_neg


# ══════════════════════════════════════════════════════════════════
# HARD NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════════

def build_hard_negatives(train_pos, gold_df, X_norm, gene_list):
    """
    Costruisce negativi difficili (hard negatives) per il training.

    Per ogni arco vero (TF_i → Gene_j) nel training set:
        1. Trova tutti i geni che TF_i NON regola nel gold standard
        2. Tra questi, trova il gene più simile a Gene_j
           usando la similarità coseno sull'expression data
        3. Crea la coppia falsa: (TF_i → Gene_più_simile_ma_non_regolato)

    Perché è difficile per la rete:
        Il negativo hard assomiglia molto al positivo corrispondente
        → la rete deve imparare distinzioni biologiche sottili
        → diventa più robusta rispetto ai negativi random

    Ritorna:
        hard_neg_df : DataFrame con le coppie hard negative
    """
    print("\n  Costruisco hard negatives...")
    print("     → Calcolo matrice di similarità tra geni...")

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes     = len(gene_list)

    # Calcola similarità coseno tra tutti i geni
    # X_norm shape: (n_geni × n_esperimenti)
    sim_matrix = cosine_similarity(X_norm)  # (n_geni × n_geni)
    print(f"     → Similarità calcolata: {sim_matrix.shape}")

    # Per ogni TF, costruisci il set di geni che regola (gold standard)
    tf_regulated = {}
    for _, row in gold_df[gold_df['Label'] == 1].iterrows():
        tf   = row['TF']
        gene = row['Gene']
        if tf not in tf_regulated:
            tf_regulated[tf] = set()
        tf_regulated[tf].add(gene)

    # Costruisci hard negatives
    hard_neg_pairs = []
    skipped        = 0

    for _, row in train_pos.iterrows():
        tf   = row['TF']
        gene = row['Gene']

        if gene not in gene_to_idx:
            skipped += 1
            continue

        gene_idx = gene_to_idx[gene]

        # Geni che TF_i NON regola
        regulated     = tf_regulated.get(tf, set())
        not_regulated = [g for g in gene_list
                         if g not in regulated
                         and g != gene
                         and g in gene_to_idx]

        if not not_regulated:
            skipped += 1
            continue

        # Indici dei geni non regolati
        not_reg_idx = np.array([gene_to_idx[g] for g in not_regulated])

        # Similarità di gene_j con tutti i geni non regolati
        similarities = sim_matrix[gene_idx, not_reg_idx]

        # Prendi il più simile
        best_pos  = np.argmax(similarities)
        best_gene = not_regulated[best_pos]

        hard_neg_pairs.append({
            'TF':    tf,
            'Gene':  best_gene,
            'Label': 0
        })

    hard_neg_df = pd.DataFrame(hard_neg_pairs)

    print(f"     → Hard negatives costruiti: {len(hard_neg_df)}")
    print(f"     → Skippati: {skipped}")

    if len(hard_neg_df) > 0:
        # Statistiche sulla similarità media
        n_sample = min(100, len(hard_neg_df))
        sample   = hard_neg_df.sample(n=n_sample, random_state=42)
        sims = []
        for _, row in sample.iterrows():
            if row['Gene'] in gene_to_idx and row['TF'] in gene_to_idx:
                i = gene_to_idx[row['Gene']]
                j = gene_to_idx[row['TF']]
                sims.append(sim_matrix[i, j])
        if sims:
            print(f"     → Similarità media hard neg: {np.mean(sims):.3f}")

    return hard_neg_df


# ══════════════════════════════════════════════════════════════════
# GOLD STANDARD — costruisce H/A dall'80% degli archi
# ══════════════════════════════════════════════════════════════════

def build_from_gold_standard(gold_df, gene_list, tf_list,
                              model_type, test_ratio=0.2, seed=42):
    """Costruisce H o A dall'80% degli archi del gold standard."""
    print("\n  Divido gold standard in train/test...")

    pos = gold_df[gold_df['Label'] == 1].copy()
    pos = pos.sample(frac=1, random_state=seed).reset_index(drop=True)

    split     = int(len(pos) * (1 - test_ratio))
    train_pos = pos.iloc[:split]
    test_pos  = pos.iloc[split:]

    print(f"     → Train: {len(train_pos)} archi veri")
    print(f"     → Test:  {len(test_pos)} archi veri")

    train_neg, test_neg = split_negatives(gold_df, test_ratio, seed)

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    tf_to_idx   = {t: j for j, t in enumerate(tf_list)}

    if model_type == 'hgnn':
        print("\n  Costruisco H (ipergrafo) dal gold standard...")
        n_genes = len(gene_list)
        n_tfs   = len(tf_list)
        H       = np.zeros((n_genes, n_tfs), dtype=np.float32)
        skipped = 0
        for _, row in train_pos.iterrows():
            tf, gene = row['TF'], row['Gene']
            if tf in tf_to_idx and gene in gene_to_idx:
                H[gene_to_idx[gene]][tf_to_idx[tf]] = 1.0
            else:
                skipped += 1
        print(f"     → H shape: {H.shape}")
        print(f"     → Connessioni: {int(H.sum())} | Saltate: {skipped}")
        return H, train_pos, test_pos, train_neg, test_neg

    else:  # gcn
        print("\n  Costruisco A (grafo) dal gold standard...")
        n = len(gene_list)
        A = np.eye(n, dtype=np.float32)
        skipped = 0
        for _, row in train_pos.iterrows():
            tf, gene = row['TF'], row['Gene']
            if tf in gene_to_idx and gene in gene_to_idx:
                i, j    = gene_to_idx[tf], gene_to_idx[gene]
                A[i][j] = 1.0
                A[j][i] = 1.0
            else:
                skipped += 1
        print(f"     → A shape: {A.shape}")
        print(f"     → Archi: {int(A.sum())-n} | Saltate: {skipped}")
        return A, train_pos, test_pos, train_neg, test_neg


# ══════════════════════════════════════════════════════════════════
# STATISTICAL — costruisce H/A dalla similarità genetica
# ══════════════════════════════════════════════════════════════════

def build_from_similarity(gold_df, X_norm, model_type,
                           k=10, test_ratio=0.2, seed=42):
    """Costruisce H o A dalla co-espressione genetica."""
    print(f"\n  Calcolo similarità tra geni (k={k})...")
    pearson = np.corrcoef(X_norm)
    cosine  = cosine_similarity(X_norm)

    def normalize(m):
        mn, mx = m.min(), m.max()
        return (m - mn) / (mx - mn + 1e-8)

    combined = 0.6 * normalize(pearson) + 0.4 * normalize(cosine)
    print(f"     → Matrice similarità: {combined.shape}")
    n = combined.shape[0]

    if model_type == 'hgnn':
        print("\n  Costruisco ipergrafo KNN...")
        H = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sorted_idx = np.argsort(-combined[i])
            neighbors  = [j for j in sorted_idx if j != i][:k]
            H[i][i]    = 1.0
            for j in neighbors:
                H[j][i] = 1.0
        print(f"     → H shape: {H.shape} | Connessioni: {int(H.sum())}")
        structure, key = H, 'H'
    else:
        print("\n  Costruisco grafo KNN...")
        A = np.eye(n, dtype=np.float32)
        for i in range(n):
            sorted_idx = np.argsort(-combined[i])
            neighbors  = [j for j in sorted_idx if j != i][:k]
            for j in neighbors:
                A[i][j] = 1.0
                A[j][i] = 1.0
        print(f"     → A shape: {A.shape} | Archi: {int(A.sum())-n}")
        structure, key = A, 'A'

    print("\n  Divido gold standard in train/test...")
    pos  = gold_df[gold_df['Label'] == 1].copy()
    pos  = pos.sample(frac=1, random_state=seed).reset_index(drop=True)
    spl  = int(len(pos) * (1 - test_ratio))
    train_pos = pos.iloc[:spl]
    test_pos  = pos.iloc[spl:]
    print(f"     → Train: {len(train_pos)} | Test: {len(test_pos)}")

    train_neg, test_neg = split_negatives(gold_df, test_ratio, seed)
    return structure, key, train_pos, test_pos, train_neg, test_neg


# ══════════════════════════════════════════════════════════════════
# EDGE PREDICTION — H completa + masking
# ══════════════════════════════════════════════════════════════════

def build_full_H(gold_df, gene_list, tf_list):
    """Costruisce H con il 100% degli archi veri."""
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    tf_to_idx   = {t: j for j, t in enumerate(tf_list)}
    H           = np.zeros((len(gene_list), len(tf_list)),
                            dtype=np.float32)
    skipped = 0
    for _, row in gold_df[gold_df['Label'] == 1].iterrows():
        tf, gene = row['TF'], row['Gene']
        if tf in tf_to_idx and gene in gene_to_idx:
            H[gene_to_idx[gene]][tf_to_idx[tf]] = 1.0
        else:
            skipped += 1
    print(f"     → H shape: {H.shape}")
    print(f"     → Connessioni: {int(H.sum())} (100%) | Saltate: {skipped}")
    return H


def random_masking(pos_df, test_ratio=0.2, seed=42):
    """Maschera il test_ratio% degli archi a caso."""
    pos = pos_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    spl = int(len(pos) * (1 - test_ratio))
    train_pos = pos.iloc[:spl]
    test_pos  = pos.iloc[spl:]
    print(f"     → [Random] Train: {len(train_pos)} | Test: {len(test_pos)}")
    return train_pos, test_pos


def stratified_masking(pos_df, test_ratio=0.2, seed=42):
    """Maschera il test_ratio% degli archi per ogni TF."""
    rng        = np.random.RandomState(seed)
    train_list = []
    test_list  = []
    for tf, group in pos_df.groupby('TF'):
        group  = group.sample(frac=1,
                              random_state=rng.randint(0, 10000))
        n      = len(group)
        if n <= 2:
            train_list.append(group)
            continue
        n_test = max(1, int(n * test_ratio))
        test_list.append(group.iloc[:n_test])
        train_list.append(group.iloc[n_test:])

    train_pos = pd.concat(train_list).reset_index(drop=True)
    test_pos  = (pd.concat(test_list).reset_index(drop=True)
                 if test_list else pd.DataFrame())
    print(f"     → [Stratified] Train: {len(train_pos)}"
          f" | Test: {len(test_pos)}")
    print(f"     → TF nel test: {test_pos['TF'].nunique()}")
    return train_pos, test_pos


def build_from_edge_prediction(gold_df, gene_list, tf_list,
                                X_norm, masking='random',
                                test_ratio=0.2, seed=42):
    """
    Edge prediction: H completa (100%) + masking del 20%.

    Masking disponibili:
        random           → positivi a caso + negativi random
        stratified       → positivi per TF + negativi random
        hard_random      → positivi a caso + negativi HARD
        hard_stratified  → positivi per TF + negativi HARD
    """
    print("\n  Costruisco H con il 100% degli archi...")
    H = build_full_H(gold_df, gene_list, tf_list)

    # Determina metodo positivi e negativi
    use_hard = masking.startswith('hard_')
    pos_method = masking.replace('hard_', '') if use_hard else masking

    print(f"\n  Applico {pos_method} masking dei positivi"
          f" ({test_ratio*100:.0f}%)...")
    pos = gold_df[gold_df['Label'] == 1].copy()

    if pos_method == 'random':
        train_pos, test_pos = random_masking(pos, test_ratio, seed)
    elif pos_method == 'stratified':
        train_pos, test_pos = stratified_masking(pos, test_ratio, seed)
    else:
        raise ValueError(f"Masking '{masking}' non valido. "
                         f"Usa: random, stratified, "
                         f"hard_random, hard_stratified")

    # Negativi
    train_neg, test_neg = split_negatives(gold_df, test_ratio, seed)

    # Hard negatives (sostituiscono train_neg nel training)
    hard_neg = None
    if use_hard:
        hard_neg = build_hard_negatives(
            train_pos, gold_df, X_norm, gene_list
        )
        print(f"     → Hard negatives: {len(hard_neg)} coppie")

    return H, train_pos, test_pos, train_neg, test_neg, hard_neg


# ══════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE
# ══════════════════════════════════════════════════════════════════

def run(data, structure='gold_standard', model_type='hgnn',
        masking='random', k=10, test_ratio=0.2, seed=42,
        use_topo_features=False):
    """
    Esegue lo Step 2 unificato.

    Parametri:
        structure         : 'gold_standard' | 'statistical' | 'edge_prediction'
        model_type        : 'hgnn' | 'gcn'
        masking           : 'random' | 'stratified' |
                            'hard_random' | 'hard_stratified'
                            (solo per edge_prediction)
        k                 : K nearest neighbors (solo per statistical)
        test_ratio        : frazione di test (default 0.2)
        seed              : riproducibilità
        use_topo_features : se True aggiunge grado ed entropia a X
                            (default False → comportamento identico a prima)
    """
    print("\n" + "═" * 55)
    print(f"  STEP 2 — {structure.upper()} | {model_type.upper()}")
    if structure == 'edge_prediction':
        print(f"  Masking: {masking}")
    if structure == 'statistical':
        print(f"  K nearest neighbors: {k}")
    if use_topo_features:
        print(f"  Feature topologiche: ATTIVE (X: 805 → 807)")
    print("═" * 55)

    expression_df = data['expression_df']
    gold_df       = data['gold_df']
    gene_list     = data['gene_list']
    tf_list       = data['tf_list']

    X_norm = normalize_expression(expression_df)

    hard_neg = None

    if structure == 'gold_standard':
        struct, train_pos, test_pos, train_neg, test_neg = \
            build_from_gold_standard(
                gold_df, gene_list, tf_list, model_type,
                test_ratio, seed
            )
        struct_key = 'H' if model_type == 'hgnn' else 'A'

    elif structure == 'statistical':
        struct, struct_key, train_pos, test_pos, train_neg, test_neg = \
            build_from_similarity(
                gold_df, X_norm, model_type, k, test_ratio, seed
            )

    elif structure == 'edge_prediction':
        struct, train_pos, test_pos, train_neg, test_neg, hard_neg = \
            build_from_edge_prediction(
                gold_df, gene_list, tf_list, X_norm,
                masking, test_ratio, seed
            )
        struct_key = 'H'

    else:
        raise ValueError(f"Structure '{structure}' non valida.")

    # Aggiunge feature topologiche se richiesto
    if use_topo_features:
        print("\n  Aggiungo feature topologiche a X...")
        # Usa la struttura già costruita per calcolare grado ed entropia
        H_for_topo = struct if struct_key == 'H' else data.get('H', struct)
        X_norm = add_topo_features(X_norm, H_for_topo)

    # Riepilogo
    print("\n" + "─" * 50)
    print("  RIEPILOGO STEP 2")
    print("─" * 50)
    print(f"  Structure    : {structure}")
    print(f"  Model type   : {model_type}")
    if structure == 'edge_prediction':
        print(f"  Masking      : {masking}")
    if use_topo_features:
        print(f"  Topo features: grado + entropia aggiunti a X")
    print(f"  {struct_key} shape     : {struct.shape}")
    print(f"  X_norm       : {X_norm.shape}")
    print(f"  Train pos    : {len(train_pos)}")
    print(f"  Test  pos    : {len(test_pos)}")
    print(f"  Train neg    : {len(train_neg)}")
    print(f"  Test  neg    : {len(test_neg)}")
    if hard_neg is not None:
        print(f"  Hard neg     : {len(hard_neg)} ← usati nel training")
    print("─" * 50)
    print("\n  ✅ Step 2 completato!")

    result = {
        **data,
        struct_key:   struct,
        "X_norm":     X_norm,
        "train_pos":  train_pos,
        "test_pos":   test_pos,
        "train_neg":  train_neg,
        "test_neg":   test_neg,
        "structure":  structure,
        "model_type": model_type,
    }

    # Aggiunge hard_neg solo se presente
    if hard_neg is not None:
        result["hard_neg"] = hard_neg

    return result


# ══════════════════════════════════════════════════════════════════
# ESECUZIONE STANDALONE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2 — Costruzione struttura (unificato)"
    )
    parser.add_argument("--network_path", type=str, required=True)
    parser.add_argument("--structure",    type=str,
                        default='gold_standard',
                        choices=['gold_standard', 'statistical',
                                 'edge_prediction'])
    parser.add_argument("--model_type",   type=str, default='hgnn',
                        choices=['hgnn', 'gcn'])
    parser.add_argument("--masking",      type=str, default='random',
                        choices=['random', 'stratified',
                                 'hard_random', 'hard_stratified'])
    parser.add_argument("--k",            type=int,   default=10)
    parser.add_argument("--test_ratio",   type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--output_dir",   type=str,   default="./output")
    args = parser.parse_args()

    data_step1 = step1.run(args.network_path)
    result     = run(data_step1,
                     structure=args.structure,
                     model_type=args.model_type,
                     masking=args.masking,
                     k=args.k,
                     test_ratio=args.test_ratio,
                     seed=args.seed)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        save_checkpoint(result,
                        os.path.join(args.output_dir, "step2.pkl"))
