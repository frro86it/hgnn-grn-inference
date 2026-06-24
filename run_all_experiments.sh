#!/bin/bash
# ================================================================
#  GRN-HGNN-DREAM5 — Script Completo Riproduzione Esperimenti
#  Versione: src_XIV (22 giugno 2026)
#  Autori  : Francesco Rollin, Fabio
#  Tutor   : Gerardo (Dino) — Unisa Bioinformatica
# ================================================================
#
#  STORIA VERSIONI:
#    src_4  (apr 2026): ablation study + GCN/Rosario + edge prediction
#                       Flag disponibili: --analysis, --exp_name
#    src_5  (17 giu):   aggiunge hard negative sampling
#                       Flag aggiunto: --use_topo_features
#    src_XIV(22 giu):   aggiunge analisi tutor (curriculum, SHAP, mining)
#                       Flag aggiunti: --weighted_tf, --use_arc_features,
#                       --curriculum, --curriculum_phase1_ratio,
#                       --curriculum_patience, --curriculum_sigmoid,
#                       --online_mining, --online_mining_warmup,
#                       --save_checkpoints (ora salva model per SHAP)
#
#  PREREQUISITI:
#    cd /percorso/src_XIV
#    source grn_hgnn_env/bin/activate
#    pip install shap --break-system-packages   (una sola volta)
#
#  USO:
#    chmod +x run_all_experiments.sh
#    ./run_all_experiments.sh
#
#  Ogni esperimento crea ./output/<exp_name>/ con config.json,
#  metrics.json, results.png. Non sovrascrive mai i risultati
#  esistenti (grazie al flag --skip_existing nel main.py).
# ================================================================

NET="/Users/francescorollin/Desktop/SoSe25/Unisa/Corsi/BioInformatica/DREAM5_network_inference_challenge/Network1"

source grn_hgnn_env/bin/activate

run() {
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  ▶  $1"
    echo "════════════════════════════════════════════════════"
    eval "$2"
    if [ $? -ne 0 ]; then
        echo "❌  ERRORE in: $1"
        exit 1
    fi
    echo "✅  Completato: $1"
}


# ================================================================
#  BLOCCO 0 — ANALISI STRUTTURALI (no training)
#  Esegui una volta sola. Non dipendono da nessun checkpoint.
# ================================================================

# Topologia ipergrafo: grado, entropia, centralità TF
# Output: ./output/topology_analysis/
run "Analisi topologica" \
    "python analyze_topology.py \
     --network_path $NET \
     --output_dir ./output/topology_analysis"

# Sparsità delle 4 strutture H: gold standard vs similarità (k=10)
# Output: ./output/sparsity_analysis/
run "Analisi sparsità" \
    "python analyze_sparsity.py \
     --network_path $NET \
     --k 10 \
     --output_dir ./output/sparsity_analysis"


# ================================================================
#  BLOCCO 1 — ABLATION STUDY IPERPARAMETRI
#  Origine: src_4. Usa --analysis hgnn_gold.
#  Varia un parametro alla volta, tutti gli altri fissi.
# ================================================================

run "exp01 — HGNN baseline (dropout=0.5 patience=10)" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp01_baseline \
     --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 10"

run "exp02 — dropout=0.6" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp02_dropout06 \
     --epochs 100 --dropout 0.6 --weight_decay 5e-4 --patience 10"

run "exp03 — dropout=0.7 (troppo: peggiora)" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp03_dropout07 \
     --epochs 100 --dropout 0.7 --weight_decay 5e-4 --patience 10"

run "exp04 — weight_decay=1e-3" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp04_wd1e3 \
     --epochs 100 --dropout 0.5 --weight_decay 1e-3 --patience 10"

run "exp05 — patience=5 (early stopping aggressivo)" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp05_patience05 \
     --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 5"

run "exp06 — lr=0.0001 (convergenza lenta)" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp06_lr0001 \
     --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 10 \
     --lr 0.0001"

run "exp07 — combinazione migliore (dropout=0.6 200ep)" \
    "python main.py --network_path $NET \
     --analysis hgnn_gold --exp_name exp07_best \
     --epochs 200 --dropout 0.6 --weight_decay 1e-3 --patience 15"


# ================================================================
#  BLOCCO 2 — GCN BASELINE E APPROCCIO ROSARIO (similarità)
#  Origine: src_4. Tutto via main.py --analysis.
# ================================================================

run "GCN gold standard (grafo biologico)" \
    "python main.py --network_path $NET \
     --analysis gcn_gold --exp_name gcn_baseline \
     --epochs 100 --dropout 0.5"

# ⚠️  Data leakage implicito: co-espressione ≈ co-regolazione
run "GCN Rosario k=10 (similarità genetica)" \
    "python main.py --network_path $NET \
     --analysis gcn_statistical --exp_name gcn_rosario_k10 \
     --k 10 --epochs 100"

run "HGNN Rosario k=10 (similarità genetica)" \
    "python main.py --network_path $NET \
     --analysis hgnn_statistical --exp_name hgnn_rosario_k10 \
     --k 10 --epochs 100"


# ================================================================
#  BLOCCO 3 — EDGE PREDICTION
#  Origine: src_4 (base), src_5 (hard negative).
#  H costruita con il 100% degli archi del gold standard.
# ================================================================

run "HGNN edge stratified ← MIGLIOR BASELINE (AUPR≈0.247)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified --exp_name edge_hgnn_stratified \
     --epochs 100"

run "HGNN edge random" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_random --exp_name edge_hgnn_random \
     --epochs 100"

run "GCN edge stratified" \
    "python main.py --network_path $NET \
     --analysis gcn_edge_stratified --exp_name edge_gcn_stratified \
     --epochs 100"

run "GCN edge random" \
    "python main.py --network_path $NET \
     --analysis gcn_edge_random --exp_name edge_gcn_random \
     --epochs 100"

# Hard negative: negativi con espressione simile al target reale
# Senza curriculum: AUPR ≈ 0.068 (troppo difficili subito)
run "HGNN hard stratified senza curriculum (AUPR≈0.068)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_edge_hard_stratified \
     --epochs 100"

run "HGNN hard random senza curriculum" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_random \
     --exp_name hgnn_edge_hard_random \
     --epochs 100"

run "GCN hard stratified" \
    "python main.py --network_path $NET \
     --analysis gcn_edge_hard_stratified \
     --exp_name gcn_edge_hard_stratified \
     --epochs 100"

run "GCN hard random" \
    "python main.py --network_path $NET \
     --analysis gcn_edge_hard_random \
     --exp_name gcn_edge_hard_random \
     --epochs 100"


# ================================================================
#  BLOCCO 4 — FEATURE TOPOLOGICHE IN X  (suggerimento tutor)
#  Origine: src_XIV.
#  Aggiunge grado ed entropia come feature esplicite in X (805→807).
#  --save_checkpoints: salva step3.pkl con model per SHAP (Blocco 10)
# ================================================================

# Risultato: AUPR ≈ 0.240 (-2.6%) — Θ già codifica la topologia
run "Topo features in X + save checkpoint (per SHAP)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified \
     --exp_name edge_hgnn_stratified_topo \
     --use_topo_features --save_checkpoints \
     --epochs 100"

run "Hard stratified + topo features" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_edge_hard_stratified_topo \
     --use_topo_features \
     --epochs 100"


# ================================================================
#  BLOCCO 5 — WEIGHTED TF IN Θ  (suggerimento tutor)
#  Origine: src_XIV.
#  W = 1/sqrt(grado_TF) — riduce influenza di G120 (26% dei target)
#  Risultato: AUPR ≈ 0.065 — AUROC alto ma calibrazione rotta
# ================================================================

run "Weighted TF in Theta (W=1/sqrt(grado))" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified \
     --exp_name hgnn_weighted_tf \
     --weighted_tf \
     --epochs 100"


# ================================================================
#  BLOCCO 6 — ARC FEATURES NEL DECODER  (suggerimento tutor)
#  Origine: src_XIV.
#  Decoder: [emb_TF | emb_gene | deg_TF | deg_gene] (258 input)
#  Risultato: AUPR ≈ 0.236 — lieve peggioramento
# ================================================================

run "Arc features nel decoder" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified \
     --exp_name hgnn_arc_features \
     --use_arc_features \
     --epochs 100"


# ================================================================
#  BLOCCO 7 — CURRICULUM LEARNING  (suggerimento tutor)
#  Origine: src_XIV.
#  Introduce i negativi hard GRADUALMENTE invece che subito:
#    Fase 1 (ratio × epochs):  negativi random (facili)
#    Fase 2:                   50% random + 50% hard (medi)
#    Fase 3:                   negativi hard (difficili)
# ================================================================

# ratio=0.33 (default), epochs=100 → AUPR ≈ 0.143 (+112% vs hard fisso)
run "Curriculum 33% epochs=100" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum \
     --curriculum \
     --epochs 100"

# ratio=0.60, epochs=150 → AUPR ≈ 0.240 (+255%)
run "Curriculum 60% epochs=150" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_60 \
     --curriculum --curriculum_phase1_ratio 0.6 \
     --epochs 150"

# ratio=0.60, epochs=200 → AUPR ≈ 0.220 (peggiora: overfitting Fase3)
run "Curriculum 60% epochs=200" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_60_ep200 \
     --curriculum --curriculum_phase1_ratio 0.6 \
     --epochs 200"

# ratio=0.70, epochs=150 → AUPR ≈ 0.266 ← MIGLIOR RISULTATO
# Supera DREAM5 CLR (0.255) per la prima volta!
run "Curriculum 70% epochs=150 ← MIGLIOR RISULTATO" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_70 \
     --curriculum --curriculum_phase1_ratio 0.7 \
     --epochs 150"

# ratio=0.80, epochs=150 → AUPR ≈ 0.266 — plateau identico a 70%
run "Curriculum 80% epochs=150 (plateau)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_80 \
     --curriculum --curriculum_phase1_ratio 0.8 \
     --epochs 150"


# ================================================================
#  BLOCCO 8 — VARIANTI CURRICULUM
#  Origine: src_XIV.  Nessuna supera curriculum_70.
# ================================================================

# Patience variabile per fase: F1=10, F2=25, F3=20
# Risultato: AUPR ≈ 0.253 — peggiora (patience non è il problema)
run "Curriculum 70% + patience variabile (10,25,20)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_70_varp \
     --curriculum --curriculum_phase1_ratio 0.7 \
     --curriculum_patience '10,25,20' \
     --epochs 150"

# Curriculum 70% + arc features nel decoder
# Risultato: AUPR ≈ 0.228 — interferenza tra i due metodi
run "Curriculum 70% + arc features" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_70_arc \
     --curriculum --curriculum_phase1_ratio 0.7 \
     --use_arc_features \
     --epochs 150"

# Curriculum continuo sigmoid: p_hard cresce da 0% a 100% senza salti
# sigmoid_midpoint=0.75, sigmoid_temp=10.0
# Risultato: AUPR ≈ 0.224 — gradualità non è il problema
run "Curriculum sigmoid (midpoint=0.75 temp=10)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_hard_stratified \
     --exp_name hgnn_curriculum_sigmoid \
     --curriculum_sigmoid \
     --curriculum_sigmoid_midpoint 0.75 \
     --curriculum_sigmoid_temp 10.0 \
     --epochs 150"


# ================================================================
#  BLOCCO 9 — ONLINE HARD NEGATIVE MINING
#  Origine: src_XIV.
#  Ogni 10 epoche ricalcola i negativi più difficili per il modello.
#  Sample: 50000 negativi scorati ad ogni aggiornamento.
# ================================================================

# Senza warmup: crolla (ep.10 primo mining, embeddings casuali)
# Risultato: AUPR ≈ 0.024
run "Online mining senza warmup (freq=10 sample=50000)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified \
     --exp_name hgnn_online_mining \
     --online_mining \
     --online_mining_freq 10 \
     --online_mining_sample 50000 \
     --epochs 150"

# Con warmup=60: ep.1-60 random poi mining ogni 10 ep
# Risultato: AUPR ≈ 0.208 (si ferma a ep.60)
run "Online mining warmup=60 (freq=10 sample=50000)" \
    "python main.py --network_path $NET \
     --analysis hgnn_edge_stratified \
     --exp_name hgnn_online_mining_w60 \
     --online_mining \
     --online_mining_warmup 60 \
     --online_mining_freq 10 \
     --online_mining_sample 50000 \
     --epochs 150"


# ================================================================
#  BLOCCO 10 — ANALISI SHAP (feature importance)
#  Origine: src_XIV.
#  Domanda tutor: grado ed entropia in X contribuiscono o sono rumore?
#  Metodo: Integrated Gradients (Sundararajan 2017) — compatibile GNN.
#  INPUT: step3.pkl di edge_hgnn_stratified_topo (Blocco 4, --save_checkpoints)
#  OUTPUT: ./output/shap_analysis/shap_analysis.png + ig_importance.csv
# ================================================================

run "Analisi SHAP feature importance" \
    "python analyze_shap.py \
     --network_path $NET \
     --exp_dir ./output/edge_hgnn_stratified_topo \
     --output_dir ./output/shap_analysis"


# ================================================================
#  BLOCCO FINALE — GRAFICO COMPARATIVO
#  Legge tutti i config.json e metrics.json in ./output/
#  Genera comparison.png e experiments_summary.csv
# ================================================================

run "Grafico comparativo finale" \
    "python compare_experiments.py --output_base ./output"


# ================================================================
echo ""
echo "████████████████████████████████████████████████████████"
echo "  ✅  TUTTI GLI ESPERIMENTI COMPLETATI!"
echo "████████████████████████████████████████████████████████"
echo ""
echo "  Grafico    : ./output/comparison.png"
echo "  Tabella    : ./output/experiments_summary.csv"
echo "  SHAP       : ./output/shap_analysis/shap_analysis.png"
echo "  Topologia  : ./output/topology_analysis/"
echo "  Sparsità   : ./output/sparsity_analysis/"
echo ""
echo "  RISULTATI FINALI (da CSV):"
echo "    curriculum_70:          AUPR ≈ 0.266  ← NOSTRO TOP"
echo "    curriculum_80:          AUPR ≈ 0.266  ← plateau"
echo "    DREAM5 CLR (benchmark): AUPR = 0.255  ← SUPERATO ✅"
echo "    edge_hgnn_stratified:   AUPR ≈ 0.247  ← baseline"
echo "    DREAM5 GENIE3:          AUPR = 0.291  ← prossimo target"
echo "████████████████████████████████████████████████████████"
