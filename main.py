"""
main.py — Orchestratore UNIFICATO
===================================
Gestisce tutti gli esperimenti con un unico file.

ANALISI DISPONIBILI:
    hgnn_gold           → HGNN con H dal gold standard (80/20)
    gcn_gold            → GCN  con A dal gold standard (80/20)
    hgnn_statistical    → HGNN con H dalla similarità genetica
    gcn_statistical     → GCN  con A dalla similarità genetica
    hgnn_edge_random    → HGNN con edge prediction + random masking
    hgnn_edge_stratified     → HGNN con edge prediction + stratified masking
    gcn_edge_random          → GCN  con edge prediction + random masking
    gcn_edge_stratified      → GCN  con edge prediction + stratified masking
    hgnn_edge_hard_random    → HGNN con edge prediction + hard random
    hgnn_edge_hard_stratified→ HGNN con edge prediction + hard stratified
    gcn_edge_hard_random     → GCN  con edge prediction + hard random
    gcn_edge_hard_stratified → GCN  con edge prediction + hard stratified
    all                      → esegue tutte le 12 analisi

USO:
    # Singola analisi con nome custom
    python main.py --network_path /percorso/Network1 \
                   --analysis hgnn_gold \
                   --exp_name exp01_baseline \
                   --epochs 100 --dropout 0.5

    # Ablation study HGNN
    python main.py --network_path /percorso/Network1 \
                   --analysis hgnn_gold \
                   --exp_name exp02_dropout06 \
                   --dropout 0.6

    # Tutte le analisi (con nomi default)
    python main.py --network_path /percorso/Network1 \
                   --analysis all

    # Tutte le analisi, salta quelle già fatte
    python main.py --network_path /percorso/Network1 \
                   --analysis all --skip_existing
"""

import argparse
import os
import time
import json
import numpy as np

import step1
import step2 as s2
import step3 as s3
import step4
from utils import save_checkpoint, load_checkpoint, save_json


# ══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE ANALISI
# ══════════════════════════════════════════════════════════════════

ANALYSES = {
    'hgnn_gold': {
        'structure':    'gold_standard',
        'model_type':   'hgnn',
        'masking':      None,
        'k':            None,
        'model_name':   'HGNN',
        'default_name': 'hgnn_gold',
        'description':  'HGNN + gold standard (80/20)',
    },
    'gcn_gold': {
        'structure':    'gold_standard',
        'model_type':   'gcn',
        'masking':      None,
        'k':            None,
        'model_name':   'GCN',
        'default_name': 'gcn_gold',
        'description':  'GCN + gold standard (80/20)',
    },
    'hgnn_statistical': {
        'structure':    'statistical',
        'model_type':   'hgnn',
        'masking':      None,
        'k':            10,
        'model_name':   'HGNN_Statistical',
        'default_name': 'hgnn_statistical_k10',
        'description':  'HGNN + similarità genetica (Rosario-like)',
    },
    'gcn_statistical': {
        'structure':    'statistical',
        'model_type':   'gcn',
        'masking':      None,
        'k':            10,
        'model_name':   'GCN_Statistical',
        'default_name': 'gcn_statistical_k10',
        'description':  'GCN + similarità genetica (Rosario-like)',
    },
    'hgnn_edge_random': {
        'structure':    'edge_prediction',
        'model_type':   'hgnn',
        'masking':      'random',
        'k':            None,
        'model_name':   'HGNN_Edge_Random',
        'default_name': 'hgnn_edge_random',
        'description':  'HGNN + edge prediction + random masking',
    },
    'hgnn_edge_stratified': {
        'structure':    'edge_prediction',
        'model_type':   'hgnn',
        'masking':      'stratified',
        'k':            None,
        'model_name':   'HGNN_Edge_Stratified',
        'default_name': 'hgnn_edge_stratified',
        'description':  'HGNN + edge prediction + stratified masking',
    },
    'gcn_edge_random': {
        'structure':    'edge_prediction',
        'model_type':   'gcn',
        'masking':      'random',
        'k':            None,
        'model_name':   'GCN_Edge_Random',
        'default_name': 'gcn_edge_random',
        'description':  'GCN + edge prediction + random masking',
    },
    'gcn_edge_stratified': {
        'structure':    'edge_prediction',
        'model_type':   'gcn',
        'masking':      'stratified',
        'k':            None,
        'model_name':   'GCN_Edge_Stratified',
        'default_name': 'gcn_edge_stratified',
        'description':  'GCN + edge prediction + stratified masking',
    },
    # ── HARD NEGATIVE SAMPLING ────────────────────────────────────
    'hgnn_edge_hard_random': {
        'structure':    'edge_prediction',
        'model_type':   'hgnn',
        'masking':      'hard_random',
        'k':            None,
        'model_name':   'HGNN_Edge_Hard_Random',
        'default_name': 'hgnn_edge_hard_random',
        'description':  'HGNN + edge prediction + hard random masking',
    },
    'hgnn_edge_hard_stratified': {
        'structure':    'edge_prediction',
        'model_type':   'hgnn',
        'masking':      'hard_stratified',
        'k':            None,
        'model_name':   'HGNN_Edge_Hard_Stratified',
        'default_name': 'hgnn_edge_hard_stratified',
        'description':  'HGNN + edge prediction + hard stratified masking',
    },
    'gcn_edge_hard_random': {
        'structure':    'edge_prediction',
        'model_type':   'gcn',
        'masking':      'hard_random',
        'k':            None,
        'model_name':   'GCN_Edge_Hard_Random',
        'default_name': 'gcn_edge_hard_random',
        'description':  'GCN + edge prediction + hard random masking',
    },
    'gcn_edge_hard_stratified': {
        'structure':    'edge_prediction',
        'model_type':   'gcn',
        'masking':      'hard_stratified',
        'k':            None,
        'model_name':   'GCN_Edge_Hard_Stratified',
        'default_name': 'gcn_edge_hard_stratified',
        'description':  'GCN + edge prediction + hard stratified masking',
    },
}


# ══════════════════════════════════════════════════════════════════
# ARGOMENTI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="GRN-HGNN-DREAM5 — Pipeline unificata"
    )

    # Input
    parser.add_argument("--network_path", type=str, required=True,
                        help="Percorso alla cartella Network1")

    # Analisi
    valid_analyses = list(ANALYSES.keys()) + ['all']
    parser.add_argument("--analysis", type=str, required=True,
                        choices=valid_analyses,
                        help="Tipo di analisi da eseguire")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Nome esperimento custom (solo per analisi singola)")

    # Output
    parser.add_argument("--output_base",      type=str, default="./output")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Salva checkpoint .pkl di ogni step")
    parser.add_argument("--skip_existing",    action="store_true",
                        help="Salta esperimenti già completati (metrics.json esiste)")

    # Parametri modello
    parser.add_argument("--test_ratio",   type=float, default=0.2)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--k",            type=int,   default=10,
                        help="K nearest neighbors (solo per statistical)")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════
# ESECUZIONE DI UN SINGOLO ESPERIMENTO
# ══════════════════════════════════════════════════════════════════

def run_experiment(analysis_key, exp_name, args, data_step1):
    """
    Esegue un singolo esperimento completo.
    """
    cfg     = ANALYSES[analysis_key]
    exp_dir = os.path.join(args.output_base, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Salta se già completato
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if args.skip_existing and os.path.exists(metrics_path):
        print(f"\n  ⏭  Esperimento '{exp_name}' già completato — skippato")
        return None

    start_time = time.time()
    k          = args.k if cfg['k'] is None else cfg['k']

    print("\n" + "█" * 55)
    print(f"  {cfg['description'].upper()}")
    print(f"  Esperimento: {exp_name}")
    print("█" * 55)

    # Salva config
    config = {
        "exp_name":     exp_name,
        "analysis":     analysis_key,
        "model":        cfg['model_name'],
        "structure":    cfg['structure'],
        "model_type":   cfg['model_type'],
        "masking":      cfg['masking'],
        "k":            k,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "hidden_dim":   args.hidden_dim,
        "dropout":      args.dropout,
        "weight_decay": args.weight_decay,
        "patience":     args.patience,
        "test_ratio":   args.test_ratio,
        "seed":         args.seed,
        "neg_split":    "opzione_B",   # documenta la correzione
    }
    save_json(config, os.path.join(exp_dir, "config.json"))

    # ── STEP 2 ────────────────────────────────────────────────────
    run_kwargs = dict(
        structure  = cfg['structure'],
        model_type = cfg['model_type'],
        masking    = cfg['masking'] or 'random',
        k          = k,
        test_ratio = args.test_ratio,
        seed       = args.seed,
    )
    data_step2 = s2.run(data_step1, **run_kwargs)

    if args.save_checkpoints:
        save_checkpoint(data_step2,
                        os.path.join(exp_dir, "step2.pkl"))

    # ── STEP 3 ────────────────────────────────────────────────────
    data_step3 = s3.run(data_step2,
                        model_type   = cfg['model_type'],
                        epochs       = args.epochs,
                        lr           = args.lr,
                        hidden_dim   = args.hidden_dim,
                        dropout      = args.dropout,
                        weight_decay = args.weight_decay,
                        patience     = args.patience)

    if args.save_checkpoints:
        result_no_model = {k: v for k, v in data_step3.items()
                           if k != 'model'}
        save_checkpoint(result_no_model,
                        os.path.join(exp_dir, "step3.pkl"))

    # ── STEP 4 ────────────────────────────────────────────────────
    results = step4.run(data_step3, output_dir=exp_dir)

    # ── Salva metriche ────────────────────────────────────────────
    elapsed = time.time() - start_time
    metrics = {
        "aupr":              results['aupr'],
        "auroc":             results['auroc'],
        "precision":         results['precision'],
        "recall":            results['recall'],
        "f1":                results['f1'],
        "baseline_aupr":     results['baseline_aupr'],
        "improvement":       results['improvement'],
        "best_epoch":        data_step3.get('best_epoch', 0),
        "training_time_sec": round(elapsed, 1),
    }
    save_json(metrics, os.path.join(exp_dir, "metrics.json"))

    # ── Riepilogo ─────────────────────────────────────────────────
    print("\n" + "█" * 55)
    print(f"  RISULTATI — {exp_name}")
    print("█" * 55)
    print(f"  Modello       : {cfg['model_name']}")
    print(f"  AUPR          : {results['aupr']:.4f}")
    print(f"  AUROC         : {results['auroc']:.4f}")
    print(f"  Miglioramento : {results['improvement']:.1f}x vs random")
    print(f"  Tempo         : {elapsed:.1f} secondi")
    print(f"  Output        : {exp_dir}")
    print("█" * 55)

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Determina quali analisi eseguire
    if args.analysis == 'all':
        to_run = list(ANALYSES.keys())
        if args.exp_name:
            print("⚠️  --exp_name ignorato con --analysis all")
            print("   Ogni analisi usa il suo nome di default.")
    else:
        to_run = [args.analysis]

    # Stampa piano di esecuzione
    print("\n" + "█" * 55)
    print("  GRN-HGNN-DREAM5 — Pipeline Unificata")
    print("█" * 55)
    print(f"\n  Analisi da eseguire: {len(to_run)}")
    for key in to_run:
        cfg      = ANALYSES[key]
        exp_name = (args.exp_name if len(to_run) == 1 and args.exp_name
                    else cfg['default_name'])
        exp_dir  = os.path.join(args.output_base, exp_name)
        status   = "✅ già fatto" if (
            args.skip_existing and
            os.path.exists(os.path.join(exp_dir, "metrics.json"))
        ) else "🔄 da fare"
        print(f"    {status}  {key:<25} → {exp_name}")

    # Carica dati una sola volta per tutti gli esperimenti
    print("\n  Carico dati DREAM5 (una sola volta)...")
    data_step1 = step1.run(args.network_path)

    # Esegui ogni analisi
    global_start = time.time()
    completed    = []
    skipped      = []

    for key in to_run:
        cfg      = ANALYSES[key]
        exp_name = (args.exp_name if len(to_run) == 1 and args.exp_name
                    else cfg['default_name'])

        result = run_experiment(key, exp_name, args, data_step1)

        if result is None:
            skipped.append(exp_name)
        else:
            completed.append((exp_name, result['aupr']))

    # Riepilogo finale
    total_time = time.time() - global_start
    print("\n" + "█" * 55)
    print("  RIEPILOGO FINALE")
    print("█" * 55)

    if completed:
        print(f"\n  Completati ({len(completed)}):")
        for name, aupr in sorted(completed,
                                  key=lambda x: x[1], reverse=True):
            print(f"    {name:<35} AUPR = {aupr:.4f}")

    if skipped:
        print(f"\n  Skippati ({len(skipped)}): {', '.join(skipped)}")

    print(f"\n  Tempo totale : {total_time:.1f} secondi")
    print(f"  Output in   : {args.output_base}")
    print("█" * 55)
    print("\n  Ora puoi eseguire:")
    print("  python compare_experiments.py "
          f"--output_base {args.output_base}")


if __name__ == "__main__":
    main()
