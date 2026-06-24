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
    parser.add_argument("--use_topo_features", action="store_true",
                        help="Aggiunge grado ed entropia come feature "
                             "aggiuntive in X (805 → 807 feature). "
                             "Di default disattivato.")

    # ── Nuovi parametri suggeriti dal tutor ───────────────────────
    parser.add_argument("--weighted_tf", action="store_true",
                        help="Pesa i TF inversamente al loro grado in Θ. "
                             "Riduce l'influenza di TF dominanti (es. G120). "
                             "Solo per HGNN.")
    parser.add_argument("--use_arc_features", action="store_true",
                        help="Aggiunge grado TF e grado gene come feature "
                             "extra al decoder (arc-level features). "
                             "Il decoder vede: [emb_TF|emb_gene|deg_TF|deg_gene].")
    parser.add_argument("--curriculum", action="store_true",
                        help="Usa curriculum learning per i negativi: "
                             "Fase1 random → Fase2 misto → Fase3 hard. "
                             "Richiede masking hard_stratified o hard_random.")
    parser.add_argument("--curriculum_phase1_ratio", type=float, default=0.33,
                        help="Frazione di epoche per la Fase 1 (negativi random). "
                             "Default 0.33 → 33/33/33. "
                             "Es. 0.60 → Fase1=60%% Fase2=20%% Fase3=20%%.")
    parser.add_argument("--curriculum_patience", type=str, default=None,
                        help="Patience variabile per fase del curriculum. "
                             "Formato: 'p1,p2,p3' es. '10,25,20'. "
                             "Default None → usa --patience per tutte le fasi.")
    # ── Curriculum sigmoid (opzione C) ────────────────────────────
    parser.add_argument("--curriculum_sigmoid", action="store_true",
                        help="Curriculum con peso sigmoid continuo invece di fasi discrete. "
                             "Default False → comportamento invariato.")
    parser.add_argument("--curriculum_sigmoid_midpoint", type=float, default=0.75,
                        help="Frazione di epoche dove p_hard=50%%. Default 0.75.")
    parser.add_argument("--curriculum_sigmoid_temp", type=float, default=10.0,
                        help="Temperatura sigmoid. Default 10.0.")
    # ── Online Hard Negative Mining (opzione E) ───────────────────
    parser.add_argument("--online_mining", action="store_true",
                        help="Online Hard Negative Mining: ogni --online_mining_freq epoche "
                             "ricalcola i negativi hard usando le predizioni del modello corrente. "
                             "Default False → comportamento invariato.")
    parser.add_argument("--online_mining_freq", type=int, default=10,
                        help="Ogni quante epoche aggiornare i negativi hard. Default 10.")
    parser.add_argument("--online_mining_k", type=int, default=0,
                        help="Quanti hard negatives selezionare. "
                             "Default 0 → uguale al numero di positivi.")
    parser.add_argument("--online_mining_sample", type=int, default=50000,
                        help="Quanti negativi campionare per lo scoring. Default 50000.")
    parser.add_argument("--online_mining_warmup", type=int, default=0,
                        help="Epoche di training su random puro prima del primo mining. "
                             "Default 0 → mining da subito (comportamento precedente). "
                             "Suggerito: 50-70 per stabilizzare gli embeddings.")

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

    # Costruisce model_name dinamico in base ai flag attivi
    model_name = cfg['model_name']
    if args.use_topo_features:
        model_name += "_Topo"
    if args.weighted_tf:
        model_name += "_WeightedTF"
    if args.use_arc_features:
        model_name += "_ArcFeat"
    if args.curriculum:
        pct = int(args.curriculum_phase1_ratio * 100)
        suffix = f"_Curriculum{pct}"
        if args.curriculum_patience:
            suffix += "_VarP"
        model_name += suffix
    if args.curriculum_sigmoid:
        mid = int(args.curriculum_sigmoid_midpoint * 100)
        model_name += f"_Sigmoid{mid}"
    if args.online_mining:
        suffix = f"_OnlineMining{args.online_mining_freq}"
        if args.online_mining_warmup > 0:
            suffix += f"W{args.online_mining_warmup}"
        model_name += suffix

    # Salva config
    config = {
        "exp_name":          exp_name,
        "analysis":          analysis_key,
        "model":             model_name,
        "structure":         cfg['structure'],
        "model_type":        cfg['model_type'],
        "masking":           cfg['masking'],
        "k":                 k,
        "epochs":            args.epochs,
        "lr":                args.lr,
        "hidden_dim":        args.hidden_dim,
        "dropout":           args.dropout,
        "weight_decay":      args.weight_decay,
        "patience":          args.patience,
        "test_ratio":        args.test_ratio,
        "seed":              args.seed,
        "neg_split":         "opzione_B",
        "use_topo_features": args.use_topo_features,
        "input_features":    807 if args.use_topo_features else 805,
        "weighted_tf":       args.weighted_tf,
        "use_arc_features":  args.use_arc_features,
        "curriculum":              args.curriculum,
        "curriculum_phase1_ratio": args.curriculum_phase1_ratio,
        "curriculum_patience":              args.curriculum_patience,
        "curriculum_sigmoid":               args.curriculum_sigmoid,
        "curriculum_sigmoid_midpoint":      args.curriculum_sigmoid_midpoint,
        "curriculum_sigmoid_temp":          args.curriculum_sigmoid_temp,
        "online_mining":                    args.online_mining,
        "online_mining_freq":               args.online_mining_freq,
        "online_mining_k":                  args.online_mining_k,
        "online_mining_sample":             args.online_mining_sample,
        "online_mining_warmup":             args.online_mining_warmup,
    }
    save_json(config, os.path.join(exp_dir, "config.json"))

    # ── STEP 2 ────────────────────────────────────────────────────
    run_kwargs = dict(
        structure         = cfg['structure'],
        model_type        = cfg['model_type'],
        masking           = cfg['masking'] or 'random',
        k                 = k,
        test_ratio        = args.test_ratio,
        seed              = args.seed,
        use_topo_features = args.use_topo_features,
    )
    data_step2 = s2.run(data_step1, **run_kwargs)

    if args.save_checkpoints:
        save_checkpoint(data_step2,
                        os.path.join(exp_dir, "step2.pkl"))

    # ── STEP 3 ────────────────────────────────────────────────────
    data_step3 = s3.run(data_step2,
                        model_type        = cfg['model_type'],
                        epochs            = args.epochs,
                        lr                = args.lr,
                        hidden_dim        = args.hidden_dim,
                        dropout           = args.dropout,
                        weight_decay      = args.weight_decay,
                        patience          = args.patience,
                        weighted_tf              = args.weighted_tf,
                        use_arc_features         = args.use_arc_features,
                        curriculum               = args.curriculum,
                        curriculum_phase1_ratio  = args.curriculum_phase1_ratio,
                        phase_patience                  = [int(p) for p in
                                                          args.curriculum_patience.split(",")]
                                                         if args.curriculum_patience else None,
                        curriculum_sigmoid              = args.curriculum_sigmoid,
                        curriculum_sigmoid_midpoint     = args.curriculum_sigmoid_midpoint,
                        curriculum_sigmoid_temp         = args.curriculum_sigmoid_temp,
                        online_mining                   = args.online_mining,
                        online_mining_freq              = args.online_mining_freq,
                        online_mining_k                 = args.online_mining_k,
                        online_mining_sample            = args.online_mining_sample,
                        online_mining_warmup            = args.online_mining_warmup)

    if args.save_checkpoints:
        # Salva TUTTO incluso il model (serve per analyze_shap.py)
        # Il model PyTorch è picklable → può stare nel pkl
        save_checkpoint(data_step3,
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
    print(f"  Topo features : {'✅ attive' if args.use_topo_features else '❌ non usate'}")
    print(f"  Weighted TF   : {'✅ attivo' if args.weighted_tf else '❌ non usato'}")
    print(f"  Arc features  : {'✅ attive' if args.use_arc_features else '❌ non usate'}")
    print(f"  Curriculum    : {'✅ attivo' if args.curriculum else '❌ non usato'}")
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
