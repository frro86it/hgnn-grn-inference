"""
utils.py — Funzioni comuni usate da tutti gli step
====================================================
"""

import os
import json
import pickle
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT (.pkl)
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(data, filepath):
    """Salva un dizionario in un file .pkl"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  💾 Checkpoint salvato: {filepath}")


def load_checkpoint(filepath):
    """Carica un checkpoint .pkl"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\n❌ Checkpoint non trovato: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  📂 Checkpoint caricato: {filepath}")
    return data


# ══════════════════════════════════════════════════════════════════
# JSON (config e metrics)
# ══════════════════════════════════════════════════════════════════

def save_json(data, filepath):
    """
    Salva un dizionario in un file .json.
    Usato per config.json e metrics.json di ogni esperimento.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    # Converte numpy/torch in tipi Python standard
    clean = {}
    for k, v in data.items():
        if hasattr(v, 'item'):
            clean[k] = v.item()
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v
    with open(filepath, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"  📄 JSON salvato: {filepath}")


def load_json(filepath):
    """Carica un file .json"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
# ISPEZIONE CHECKPOINT
# ══════════════════════════════════════════════════════════════════

def inspect_checkpoint(filepath):
    """
    Ispeziona il contenuto di un checkpoint.

    Uso da terminale:
        python -c "from utils import inspect_checkpoint; \
                   inspect_checkpoint('./output/exp01/step1.pkl')"
    """
    data = load_checkpoint(filepath)
    print("\n📋 Contenuto del checkpoint:")
    print("=" * 45)
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            print(f"  {key:20s}: DataFrame {value.shape}")
        elif isinstance(value, list):
            print(f"  {key:20s}: lista con {len(value)} elementi")
        elif isinstance(value, np.ndarray):
            print(f"  {key:20s}: array numpy {value.shape}")
        elif value is None:
            print(f"  {key:20s}: None")
        else:
            print(f"  {key:20s}: {type(value).__name__} = {value}")
    print("=" * 45)
