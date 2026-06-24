"""
step3.py — Training HGNN e GCN (UNIFICATO)
===========================================
Gestisce sia HGNN che GCN con un unico file.
Il tipo di modello viene scelto dal parametro model_type.

Implementazioni:
    HGNN : Feng et al. "Hypergraph Neural Networks" AAAI 2019
    GCN  : Kipf & Welling "Semi-Supervised Classification" ICLR 2017

OPZIONE B — usa train_neg per il campionamento durante il training,
            lasciando test_neg intatto per la valutazione.

Uso standalone:
    python step3.py --checkpoint ./output/exp/step2.pkl \
                    --model_type hgnn --epochs 100

    python step3.py --checkpoint ./output/exp/step2.pkl \
                    --model_type gcn --epochs 100
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from utils import save_checkpoint, load_checkpoint


# ══════════════════════════════════════════════════════════════════
# ARCHITETTURA HGNN
# ══════════════════════════════════════════════════════════════════

class HGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.theta   = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(out_dim)

    def forward(self, X, theta):
        X = self.dropout(X)
        X = self.theta(X)
        X = torch.mm(theta, X)
        X = self.bn(X)
        return F.relu(X)


class HGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5,
                 use_arc_features=False):
        super().__init__()
        self.use_arc_features = use_arc_features
        arc_dim = 2 if use_arc_features else 0
        self.layer1  = HGNNLayer(in_dim,     hidden_dim, dropout)
        self.layer2  = HGNNLayer(hidden_dim, out_dim,    dropout)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2 + arc_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, X, theta):
        return self.layer2(self.layer1(X, theta), theta)

    def predict(self, embeddings, tf_idx, gene_idx, arc_features=None):
        pair = torch.cat([embeddings[tf_idx], embeddings[gene_idx]], dim=-1)
        if self.use_arc_features and arc_features is not None:
            pair = torch.cat([pair, arc_features], dim=-1)
        return torch.sigmoid(self.decoder(pair))


# ══════════════════════════════════════════════════════════════════
# ARCHITETTURA GCN
# ══════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.theta   = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(out_dim)

    def forward(self, X, theta):
        X = self.dropout(X)
        X = self.theta(X)
        X = torch.mm(theta, X)
        X = self.bn(X)
        return F.relu(X)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5,
                 use_arc_features=False):
        super().__init__()
        self.use_arc_features = use_arc_features
        arc_dim = 2 if use_arc_features else 0
        self.layer1  = GCNLayer(in_dim,     hidden_dim, dropout)
        self.layer2  = GCNLayer(hidden_dim, out_dim,    dropout)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim * 2 + arc_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, X, theta):
        return self.layer2(self.layer1(X, theta), theta)

    def predict(self, embeddings, tf_idx, gene_idx, arc_features=None):
        pair = torch.cat([embeddings[tf_idx], embeddings[gene_idx]], dim=-1)
        if self.use_arc_features and arc_features is not None:
            pair = torch.cat([pair, arc_features], dim=-1)
        return torch.sigmoid(self.decoder(pair))


# ══════════════════════════════════════════════════════════════════
# MATRICI DI PROPAGAZIONE
# ══════════════════════════════════════════════════════════════════

def build_theta_hgnn(H, weighted_tf=False):
    H_t = torch.tensor(H, dtype=torch.float32)
    if weighted_tf:
        De = H_t.sum(dim=0)
        W  = 1.0 / torch.sqrt(De.clamp(min=1.0))
        W  = W / W.max()
        print(f"     → W min={W.min():.4f} max={W.max():.4f}")
    else:
        W  = torch.ones(H_t.shape[1])

    Dv         = H_t.sum(dim=1)
    Dv_invsqrt = torch.where(Dv > 0,
                              1.0 / torch.sqrt(Dv),
                              torch.zeros_like(Dv))
    De     = H_t.sum(dim=0)
    De_inv = torch.where(De > 0,
                          1.0 / De,
                          torch.zeros_like(De))

    Dv_mat = torch.diag(Dv_invsqrt)
    W_mat  = torch.diag(W)
    De_mat = torch.diag(De_inv)

    return Dv_mat @ H_t @ W_mat @ De_mat @ H_t.T @ Dv_mat


def build_theta_gcn(A):
    """
    Calcola Θ = D^{-1/2} A D^{-1/2}
    dalla matrice di adiacenza A (Kipf & Welling 2017).
    """
    A_t = torch.tensor(A, dtype=torch.float32)
    D   = A_t.sum(dim=1)
    D_invsqrt = torch.where(D > 0,
                              1.0 / torch.sqrt(D),
                              torch.zeros_like(D))
    D_mat = torch.diag(D_invsqrt)
    return D_mat @ A_t @ D_mat


def build_adjacency_from_gold(train_pos, gene_list):
    """Costruisce A dal gold standard (solo per GCN senza A pre-calcolata)."""
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n       = len(gene_list)
    A       = np.eye(n, dtype=np.float32)
    skipped = 0
    for _, row in train_pos.iterrows():
        tf, gene = row['TF'], row['Gene']
        if tf in gene_to_idx and gene in gene_to_idx:
            i, j = gene_to_idx[tf], gene_to_idx[gene]
            A[i][j] = 1.0
            A[j][i] = 1.0
        else:
            skipped += 1
    n_edges = int(A.sum()) - n
    print(f"     → A shape: {A.shape}")
    print(f"     → Archi attivi: {n_edges} | Saltate: {skipped}")
    return A


# ══════════════════════════════════════════════════════════════════
# PREPARAZIONE DATI — usa train_neg (Opzione B)
# ══════════════════════════════════════════════════════════════════

def prepare_training_data(train_pos, train_neg, gene_list, device,
                          hard_neg=None):
    """
    Prepara coppie TF-gene bilanciate per il training.

    OPZIONE B: campiona i negativi SOLO da train_neg
    (mai usati nel test set) → zero data leakage.

    Se hard_neg è presente, usa quelli invece di campionare
    da train_neg. Gli hard negatives sono costruiti in step2
    e sono biologicamente più difficili da classificare.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    pos_pairs = []
    for _, row in train_pos.iterrows():
        if row['TF'] in gene_to_idx and row['Gene'] in gene_to_idx:
            pos_pairs.append((gene_to_idx[row['TF']],
                              gene_to_idx[row['Gene']], 1.0))

    # Scegli i negativi
    if hard_neg is not None and len(hard_neg) > 0:
        # Hard negatives: geni simili ai positivi ma non regolati
        neg_source = hard_neg
        neg_label  = "hard neg"
    else:
        # Random negatives: campiona da train_neg
        n_neg      = min(len(pos_pairs), len(train_neg))
        neg_source = train_neg.sample(n=n_neg, random_state=42)
        neg_label  = "random neg"

    neg_pairs = []
    for _, row in neg_source.iterrows():
        if row['TF'] in gene_to_idx and row['Gene'] in gene_to_idx:
            neg_pairs.append((gene_to_idx[row['TF']],
                              gene_to_idx[row['Gene']], 0.0))

    all_pairs = pos_pairs + neg_pairs
    np.random.shuffle(all_pairs)

    tf_idx   = torch.tensor([p[0] for p in all_pairs],
                              dtype=torch.long).to(device)
    gene_idx = torch.tensor([p[1] for p in all_pairs],
                              dtype=torch.long).to(device)
    labels   = torch.tensor([p[2] for p in all_pairs],
                              dtype=torch.float32).to(device)

    print(f"     → Coppie training: {len(all_pairs)}")
    print(f"       ({len(pos_pairs)} pos, {len(neg_pairs)} {neg_label})")
    return tf_idx, gene_idx, labels


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP — comune a HGNN e GCN
# ══════════════════════════════════════════════════════════════════

def build_arc_features(H, tf_idx, gene_idx, device):
    """Feature a livello di arco: grado normalizzato TF e gene."""
    H_t        = torch.tensor(H, dtype=torch.float32)
    tf_degree  = H_t.sum(dim=0)
    gene_degree = H_t.sum(dim=1)
    tf_dn      = (tf_degree[tf_idx]    / tf_degree.max().clamp(min=1.0)).unsqueeze(-1).to(device)
    gene_dn    = (gene_degree[gene_idx] / gene_degree.max().clamp(min=1.0)).unsqueeze(-1).to(device)
    return torch.cat([tf_dn, gene_dn], dim=-1)


def score_negatives(model, X, theta, neg_df, gene_list, device, sample_size=50000):
    """
    Calcola la predizione del modello su un campione di negativi.
    Ritorna gli indici ordinati per predizione decrescente
    (= i più difficili per il modello in questo momento).
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    sample      = neg_df.sample(n=min(sample_size, len(neg_df)), random_state=42)
    valid       = sample[sample['TF'].isin(gene_to_idx) &
                         sample['Gene'].isin(gene_to_idx)]
    if len(valid) == 0:
        return neg_df.sample(n=min(sample_size, len(neg_df)))

    tf_idx_s   = torch.tensor([gene_to_idx[r] for r in valid['TF']],
                               dtype=torch.long).to(device)
    gene_idx_s = torch.tensor([gene_to_idx[r] for r in valid['Gene']],
                               dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        emb   = model(X, theta)
        preds = model.predict(emb, tf_idx_s, gene_idx_s).squeeze().cpu().numpy()

    valid = valid.copy()
    valid['_score'] = preds
    valid = valid.sort_values('_score', ascending=False)
    return valid.drop(columns=['_score'])


def train_model(model, X, theta, tf_idx, gene_idx, labels,
                epochs=100, lr=0.001, weight_decay=5e-4,
                patience=10, curriculum_data=None,
                arc_features=None, phase_patience=None,
                curriculum_sigmoid=False,
                online_mining_data=None):
    """
    Allena il modello con early stopping.

    Tutti i parametri nuovi hanno default backward-compatible:
      curriculum_data:    None  → comportamento originale
      arc_features:       None  → decoder standard
      phase_patience:     None  → patience globale per tutte le fasi
      curriculum_sigmoid: False → fasi discrete (comportamento originale)
      online_mining_data: None  → nessun mining (comportamento originale)
                          dict  → ogni 'freq' epoche ricalcola i negativi hard
                                  usando le predizioni del modello corrente.
                                  'warmup' epoche iniziali su random puro
                                  prima del primo aggiornamento.
    """
    import pandas as pd
    import numpy as np

    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    loss_fn    = nn.BCELoss()
    history    = []
    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    best_epoch = 0

    # ── Decide il percorso di training ───────────────────────────
    # I 4 percorsi sono MUTUAMENTE ESCLUSIVI e tutti backward-compat
    use_online   = online_mining_data is not None       # NUOVO
    use_sigmoid  = (not use_online) and curriculum_sigmoid                    and curriculum_data is not None
    use_discrete = (not use_online) and (not use_sigmoid)                    and curriculum_data is not None
    # use_online=False, use_sigmoid=False, use_discrete=False
    # → percorso originale (tf_idx/gene_idx/labels fissi)

    # ── Setup fasi discrete ───────────────────────────────────────
    phase1_end = phase2_end = 0
    if use_discrete:
        r1         = curriculum_data.get('phase1_ratio', 0.33)
        r2         = r1 + (1.0 - r1) / 2
        phase1_end = int(epochs * r1)
        phase2_end = int(epochs * r2)
        print(f"\n  Curriculum FASI DISCRETE: F1 ep1-{phase1_end} | "
              f"F2 ep{phase1_end+1}-{phase2_end} | F3 ep{phase2_end+1}-{epochs}")

    # ── Setup sigmoid ─────────────────────────────────────────────
    sig_mid = sig_temp = 0
    if use_sigmoid:
        sig_mid  = curriculum_data.get('sigmoid_midpoint', 0.75) * epochs
        sig_temp = curriculum_data.get('sigmoid_temp', 10.0)
        print(f"\n  Curriculum SIGMOID: midpoint ep={sig_mid:.0f} temp={sig_temp}")

    # ── Setup online mining ───────────────────────────────────────
    if use_online:
        om        = online_mining_data
        om_freq   = om.get('freq',    10)
        om_k      = om.get('k',       len(om['train_pos']))
        om_samp   = om.get('sample',  50000)
        om_warmup = om.get('warmup',  0)
        print(f"\n  Online Hard Negative Mining ATTIVO:")
        print(f"     Warmup: ep 1-{om_warmup} su random puro "
              f"{'(nessun warmup)' if om_warmup == 0 else ''}")
        print(f"     Mining: ogni {om_freq} epoche da ep {om_warmup+1}")
        print(f"     Top-{om_k} negativi su campione di {om_samp}")
        print(f"     Ciclo: predict → rank → seleziona top-K → training")

    # ── Patience per fase ─────────────────────────────────────────
    p_fase = list(phase_patience)              if phase_patience is not None and len(phase_patience) == 3              else [patience, patience, patience]
    cur_patience = p_fase[0]

    print(f"\n  Inizio training ({epochs} epoche, patience={patience})...")
    print(f"  {'Epoca':>6} | {'Loss':>10} | {'AUPR':>8} | Fase")
    print(f"  {'─'*6}-+-{'─'*10}-+-{'─'*8}-+-────")

    cur_tf_idx   = tf_idx
    cur_gene_idx = gene_idx
    cur_labels   = labels
    cur_arc      = arc_features
    cur_phase    = "random"

    for epoch in range(1, epochs + 1):

        # ══ PERCORSO A: ONLINE MINING ═════════════════════════════
        if use_online:
            om = online_mining_data

            if epoch <= om_warmup:
                # ── Fase warmup: random puro, nessun mining ───────
                if epoch == 1:
                    cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                        om['train_pos'], om['train_neg'],
                        om['gene_list'], om['device'], hard_neg=None)
                    cur_phase = "warmup"
                    print(f"  [ep   1] Warmup su random puro "
                          f"({om_warmup} epoche)")

            elif epoch == om_warmup + 1 or                  (epoch > om_warmup and (epoch - om_warmup) % om_freq == 0):
                # ── Fase mining: aggiorna negativi ogni freq epoche ─
                # Il modello ha già om_warmup epoche di training
                # → embeddings significativi → mining affidabile
                hard_now = score_negatives(
                    model, X, theta,
                    om['train_neg'], om['gene_list'],
                    om['device'], sample_size=om_samp
                ).head(om_k)

                cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                    om['train_pos'], om['train_neg'],
                    om['gene_list'], om['device'],
                    hard_neg=hard_now)
                n_update = (epoch - om_warmup) // om_freq
                cur_phase = f"mine#{n_update}"
                print(f"  [ep {epoch:>3}] Mining #{n_update}: "
                      f"{om_k} hard neg aggiornati")

        # ══ PERCORSO B: SIGMOID ════════════════════════════════════
        elif use_sigmoid:
            p_hard  = 1.0 / (1.0 + np.exp(-(epoch - sig_mid) / sig_temp))
            n       = len(curriculum_data['train_pos'])
            n_hard  = min(int(n * p_hard),
                          len(curriculum_data['hard_neg']))
            n_rand  = n - n_hard
            hard_s  = curriculum_data['hard_neg'].sample(
                          n=n_hard, random_state=epoch) if n_hard > 0                       else pd.DataFrame()
            rand_s  = curriculum_data['train_neg'].sample(
                          n=n_rand, random_state=epoch)
            mixed   = pd.concat([hard_s, rand_s]) if n_hard > 0 else rand_s
            cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                curriculum_data['train_pos'], mixed,
                curriculum_data['gene_list'],
                curriculum_data['device'], hard_neg=None)
            cur_phase = f"s{int(p_hard*100)}%"

        # ══ PERCORSO C: FASI DISCRETE ══════════════════════════════
        elif use_discrete:
            cd = curriculum_data
            if epoch == 1:
                cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                    cd['train_pos'], cd['train_neg'],
                    cd['gene_list'], cd['device'], hard_neg=None)
                cur_phase    = "random"
                cur_patience = p_fase[0]
                no_improve   = 0
            elif epoch == phase1_end + 1:
                if cd.get('hard_neg') is not None:
                    n           = len(cd['train_pos'])
                    half_hard   = cd['hard_neg'].sample(n=n//2, random_state=epoch)
                    half_random = cd['train_neg'].sample(n=n//2, random_state=epoch)
                    mixed_neg   = pd.concat([half_hard, half_random])
                    cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                        cd['train_pos'], mixed_neg,
                        cd['gene_list'], cd['device'], hard_neg=None)
                cur_phase    = "mixed"
                cur_patience = p_fase[1]
                no_improve   = 0
            elif epoch == phase2_end + 1:
                if cd.get('hard_neg') is not None:
                    cur_tf_idx, cur_gene_idx, cur_labels = prepare_training_data(
                        cd['train_pos'], cd['train_neg'],
                        cd['gene_list'], cd['device'],
                        hard_neg=cd['hard_neg'])
                cur_phase    = "hard"
                cur_patience = p_fase[2]
                no_improve   = 0

        # ══ Training step (uguale per tutti i percorsi) ════════════
        model.train()
        optimizer.zero_grad()
        embeddings = model(X, theta)
        preds      = model.predict(
            embeddings, cur_tf_idx, cur_gene_idx,
            arc_features=cur_arc).squeeze()
        loss = loss_fn(preds, cur_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = loss.item()
        aupr = 0.0

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                emb_e  = model(X, theta)
                pred_e = model.predict(
                    emb_e, cur_tf_idx, cur_gene_idx,
                    arc_features=cur_arc).squeeze().cpu().numpy()
            aupr = average_precision_score(cur_labels.cpu().numpy(), pred_e)
            phase_str = cur_phase if use_online else cur_phase[:8]
            print(f"  {epoch:>6} | {current_loss:>10.4f} | "
                  f"{aupr:>8.4f} | {phase_str:<8}")

        history.append({'epoch': epoch, 'loss': current_loss, 'aupr': aupr})

        if current_loss < best_loss:
            best_loss  = current_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cur_patience:
                print(f"\n  ⏹  Early stopping a epoca {epoch} "
                      f"(patience={cur_patience})")
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  Miglior loss: {best_loss:.4f} (epoca {best_epoch})")
    return model, history, best_epoch


# ══════════════════════════════════════════════════════════════════
# FUNZIONE PRINCIPALE
# ══════════════════════════════════════════════════════════════════

def run(data, model_type='hgnn', epochs=100, lr=0.001,
        hidden_dim=128, dropout=0.5, weight_decay=5e-4, patience=10,
        weighted_tf=False, use_arc_features=False,
        curriculum=False, curriculum_phase1_ratio=0.33,
        phase_patience=None,
        curriculum_sigmoid=False,
        curriculum_sigmoid_midpoint=0.75,
        curriculum_sigmoid_temp=10.0,
        online_mining=False,
        online_mining_freq=10,
        online_mining_k=0,
        online_mining_sample=50000,
        online_mining_warmup=0):
    """
    Esegue il training del modello scelto.
    Tutti i nuovi flag hanno default backward-compatible (False/0/None).

    Parametri:
        model_type : 'hgnn' o 'gcn'

    Seleziona automaticamente:
        HGNN → usa H e build_theta_hgnn
        GCN  → usa A (se disponibile in data) o la costruisce
               da train_pos con build_adjacency_from_gold
    """
    model_name = model_type.upper()
    print("\n" + "═" * 55)
    print(f"  STEP 3 — Training {model_name}")
    print("═" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo  : {device}")
    print(f"  Model type   : {model_name}")
    print(f"  Dropout      : {dropout}")
    print(f"  Weight decay : {weight_decay}")
    print(f"  Patience     : {patience}")

    X_norm    = data['X_norm']
    train_pos = data['train_pos']
    train_neg = data['train_neg']   # Opzione B
    gene_list = data['gene_list']

    # ── Costruisci matrice di propagazione Θ ──────────────────────
    if model_type == 'hgnn':
        print("\n  Calcolo Θ (ipergrafo)...")
        H     = data['H']
        theta = build_theta_hgnn(H, weighted_tf=weighted_tf).to(device)
    else:  # gcn
        if 'A' in data:
            print("\n  Uso A pre-calcolata (approccio statistico)...")
            A = data['A']
        else:
            print("\n  Costruisco A dal gold standard...")
            A = build_adjacency_from_gold(train_pos, gene_list)
        theta = build_theta_gcn(A).to(device)
    print(f"     → Θ shape: {theta.shape}")

    # ── Feature nodi ──────────────────────────────────────────────
    X = torch.tensor(X_norm, dtype=torch.float32).to(device)

    # ── Dati training (Opzione B — campiona da train_neg) ─────────
    hard_neg = data.get('hard_neg', None)
    if hard_neg is not None:
        print("\n  Preparo coppie di training (HARD NEGATIVES)...")
    else:
        print("\n  Preparo coppie di training...")
    tf_idx, gene_idx, labels = prepare_training_data(
        train_pos, train_neg, gene_list, device,
        hard_neg=hard_neg
    )

    # ── Modello ───────────────────────────────────────────────────
    in_dim = X.shape[1]
    print(f"\n  Architettura {model_name}:")
    print(f"     {in_dim} → {hidden_dim} → {hidden_dim}")
    print(f"     Dropout: {dropout}")

    ModelClass = HGNN if model_type == 'hgnn' else GCN
    model      = ModelClass(in_dim=in_dim, hidden_dim=hidden_dim,
                             out_dim=hidden_dim, dropout=dropout,
                             use_arc_features=use_arc_features).to(device)

    # ── Arc features ─────────────────────────────────────────────
    arc_feats = None
    if use_arc_features and model_type == 'hgnn' and 'H' in data:
        arc_feats = build_arc_features(data['H'], tf_idx, gene_idx, device)

    # ── Curriculum data (fasi discrete e sigmoid) ─────────────────
    curriculum_data = None
    use_any_curr = curriculum or curriculum_sigmoid
    if use_any_curr and data.get('hard_neg') is not None:
        curriculum_data = {
            'train_pos':        train_pos,
            'train_neg':        train_neg,
            'hard_neg':         data['hard_neg'],
            'gene_list':        gene_list,
            'device':           device,
            'phase1_ratio':     curriculum_phase1_ratio,
            'sigmoid_midpoint': curriculum_sigmoid_midpoint,
            'sigmoid_temp':     curriculum_sigmoid_temp,
        }
    elif use_any_curr:
        print("  ⚠️  Curriculum richiede hard_neg")

    # ── Online Mining data (percorso separato) ────────────────────
    online_mining_data = None
    if online_mining and 'train_neg' in data:
        n_pos  = len(train_pos)
        om_k   = online_mining_k if online_mining_k > 0 else n_pos
        online_mining_data = {
            'train_pos':  train_pos,
            'train_neg':  train_neg,
            'gene_list':  gene_list,
            'device':     device,
            'freq':       online_mining_freq,
            'k':          om_k,
            'sample':     online_mining_sample,
            'warmup':     online_mining_warmup,
        }
    elif online_mining:
        print("  ⚠️  Online Mining richiede train_neg nei dati")

    # ── Training ──────────────────────────────────────────────────
    model, history, best_epoch = train_model(
        model, X, theta, tf_idx, gene_idx, labels,
        epochs=epochs, lr=lr,
        weight_decay=weight_decay, patience=patience,
        curriculum_data=curriculum_data,
        arc_features=arc_feats,
        phase_patience=phase_patience,
        curriculum_sigmoid=curriculum_sigmoid,
        online_mining_data=online_mining_data
    )

    # ── Embedding finali ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        embeddings = model(X, theta)

    print("\n" + "─" * 45)
    print(f"  RIEPILOGO STEP 3")
    print("─" * 45)
    print(f"  Epoche eseguite : {len(history)}")
    print(f"  Miglior epoca   : {best_epoch}")
    print(f"  Loss finale     : {history[-1]['loss']:.4f}")
    print(f"  Embedding shape : {embeddings.shape}")
    print("─" * 45)
    print(f"\n  ✅ Step 3 ({model_name}) completato!")

    return {
        **data,
        "model":      model,
        "embeddings": embeddings.cpu().detach().numpy(),
        "history":    history,
        "best_epoch": best_epoch,
        "theta":      theta.cpu().detach().numpy(),
        "X":          X.cpu().detach().numpy(),
    }


# ══════════════════════════════════════════════════════════════════
# ESECUZIONE STANDALONE
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 — Training HGNN o GCN (unificato)"
    )
    parser.add_argument("--checkpoint",   type=str,   required=True)
    parser.add_argument("--model_type",   type=str,   default='hgnn',
                        choices=['hgnn', 'gcn'])
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--output_dir",   type=str,   default="./output")
    args = parser.parse_args()

    data_step2 = load_checkpoint(args.checkpoint)
    result     = run(data_step2,
                     model_type=args.model_type,
                     epochs=args.epochs,
                     lr=args.lr,
                     hidden_dim=args.hidden_dim,
                     dropout=args.dropout,
                     weight_decay=args.weight_decay,
                     patience=args.patience)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(result['model'].state_dict(),
                   os.path.join(args.output_dir,
                                f"model_{args.model_type}.pt"))
        result_no_model = {k: v for k, v in result.items()
                           if k != 'model'}
        save_checkpoint(result_no_model,
                        os.path.join(args.output_dir, "step3.pkl"))
