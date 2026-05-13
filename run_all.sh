#!/bin/bash
# ================================================================
# run_all.sh — Esegue tutti gli esperimenti in sequenza
# ================================================================
# Uso:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Per eseguire solo alcuni blocchi commenta le righe che non vuoi
# ================================================================

# ── Percorso dati ─────────────────────────────────────────────────
NET="/Users/francescorollin/Desktop/SoSe25/Unisa/Corsi/BioInformatica/DREAM5_network_inference_challenge/Network1"

# ── Attiva ambiente virtuale ──────────────────────────────────────
source grn_hgnn_env/bin/activate

# ── Funzione helper ───────────────────────────────────────────────
run() {
    echo ""
    echo "════════════════════════════════════════"
    echo "  Avvio: $1"
    echo "════════════════════════════════════════"
    eval "$1"
    if [ $? -ne 0 ]; then
        echo "❌ Errore in: $1"
        exit 1
    fi
    echo "✅ Completato: $1"
}

# ── ABLATION STUDY HGNN ──────────────────────────────────────────
run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp01_baseline --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 10"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp02_dropout06 --epochs 100 --dropout 0.6 --weight_decay 5e-4 --patience 10"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp03_dropout07 --epochs 100 --dropout 0.7 --weight_decay 5e-4 --patience 10"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp04_wd1e3 --epochs 100 --dropout 0.5 --weight_decay 1e-3 --patience 10"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp05_patience05 --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 5"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp06_lr0001 --epochs 100 --dropout 0.5 --weight_decay 5e-4 --patience 10 --lr 0.0001"

run "python main.py --network_path $NET --analysis hgnn_gold --exp_name exp07_best --epochs 200 --dropout 0.6 --weight_decay 1e-3 --patience 15"

# ── GCN GOLD STANDARD ────────────────────────────────────────────
run "python main.py --network_path $NET --analysis gcn_gold --exp_name gcn_baseline --epochs 100 --dropout 0.5"

# ── APPROCCIO STATISTICO (Rosario) ───────────────────────────────
run "python main.py --network_path $NET --analysis gcn_statistical --exp_name gcn_rosario_k10 --k 10 --epochs 100"

run "python main.py --network_path $NET --analysis hgnn_statistical --exp_name hgnn_rosario_k10 --k 10 --epochs 100"

# ── EDGE PREDICTION ──────────────────────────────────────────────
run "python main.py --network_path $NET --analysis hgnn_edge_random --exp_name edge_hgnn_random --epochs 100"

run "python main.py --network_path $NET --analysis hgnn_edge_stratified --exp_name edge_hgnn_stratified --epochs 100"

run "python main.py --network_path $NET --analysis gcn_edge_random --exp_name edge_gcn_random --epochs 100"

run "python main.py --network_path $NET --analysis gcn_edge_stratified --exp_name edge_gcn_stratified --epochs 100"

# ── CONFRONTO FINALE ─────────────────────────────────────────────
run "python compare_experiments.py --output_base ./output"

echo ""
echo "████████████████████████████████████████"
echo "  TUTTI GLI ESPERIMENTI COMPLETATI!"
echo "  Risultati in: ./output/"
echo "████████████████████████████████████████"

# ── HARD NEGATIVE SAMPLING ───────────────────────────────────────
run "python main.py --network_path $NET --analysis hgnn_edge_hard_random --exp_name hgnn_edge_hard_random --epochs 100"

run "python main.py --network_path $NET --analysis hgnn_edge_hard_stratified --exp_name hgnn_edge_hard_stratified --epochs 100"

run "python main.py --network_path $NET --analysis gcn_edge_hard_random --exp_name gcn_edge_hard_random --epochs 100"

run "python main.py --network_path $NET --analysis gcn_edge_hard_stratified --exp_name gcn_edge_hard_stratified --epochs 100"
