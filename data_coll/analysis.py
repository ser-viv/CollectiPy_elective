"""
analysis.py
===========
Analisi dei dati per gli esperimenti di allocentric flocking.

Struttura attesa:
    EX_01/EX_01/run_1.zip  →  run_1/group_metrics.csv
    EX_02/EX_02/run_1.zip  →  run_1/group_metrics.csv
    ...
    EX_16/EX_16/run_1.zip  →  run_1/group_metrics.csv

Colonne CSV: tick, group, cohesion, polarization, heading_mean_deg

Parametri per esperimento:
    RW = repulsion_weight  ∈ {0, 0.3, 0.6, 1.0}
    ARW = arena_repulsion_weight   ∈ {0, 0.3, 0.6, 1.0}

Analisi prodotte:
    1. Heatmap delle metriche medie (coesione, polarizzazione)
    2. Evoluzione temporale con banda di confidenza (media ± std)
    3. Distribuzione finale delle metriche per esperimento
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

# Cartella radice che contiene EX_01 ... EX_16
BASE_DIR = "."

# Schema parametri: EX_n → [RW, RR]
EXPERIMENTS = {
    "EX_01": [0.0, 0.0],
    "EX_02": [0.0, 0.3],
    "EX_03": [0.0, 0.6],
    "EX_04": [0.0, 1.0],
    "EX_05": [0.3, 0.0],
    "EX_06": [0.3, 0.3],
    "EX_07": [0.3, 0.6],
    "EX_08": [0.3, 1.0],
    "EX_09": [0.6, 0.0],
    "EX_10": [0.6, 0.3],
    "EX_11": [0.6, 0.6],
    "EX_12": [0.6, 1.0],
    "EX_13": [1.0, 0.0],
    "EX_14": [1.0, 0.3],
    "EX_15": [1.0, 0.6],
    "EX_16": [1.0, 1.0],
}

RW_VALUES = [0.0, 0.3, 0.6, 1.0]
ARW_VALUES = [0.0, 0.3, 0.6, 1.0]

# Frazione finale della simulazione usata per calcolare le metriche stazionarie
# (es. 0.5 = ultima metà della simulazione)
STATIONARY_FRACTION = 0.5

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARICAMENTO DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_run_metrics(zip_path: str) -> pd.DataFrame | None:
    """Carica group_metrics.csv da un file run_N.zip."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_candidates = [n for n in zf.namelist() if n.endswith("group_metrics.csv")]
            if not csv_candidates:
                return None
            with zf.open(csv_candidates[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
            return df
    except Exception as e:
        print(f"  [WARN] impossibile leggere {zip_path}: {e}")
        return None


def load_experiment(ex_name: str) -> list[pd.DataFrame]:
    """Carica tutte le run di un esperimento. Restituisce lista di DataFrame."""
    ex_dir = os.path.join(BASE_DIR, ex_name, ex_name)
    if not os.path.isdir(ex_dir):
        # prova senza subdirectory
        ex_dir = os.path.join(BASE_DIR, ex_name)
    if not os.path.isdir(ex_dir):
        print(f"  [WARN] cartella non trovata: {ex_name}")
        return []

    runs = []
    for fname in sorted(os.listdir(ex_dir)):
        if fname.startswith("run_") and fname.endswith(".zip"):
            df = load_run_metrics(os.path.join(ex_dir, fname))
            if df is not None and len(df) > 0:
                runs.append(df)
    print(f"  {ex_name}: {len(runs)} run caricate")
    return runs


def load_all_experiments() -> dict[str, list[pd.DataFrame]]:
    """Carica tutti gli esperimenti."""
    print("Caricamento dati...")
    data = {}
    for ex_name in EXPERIMENTS:
        data[ex_name] = load_experiment(ex_name)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# CALCOLO METRICHE AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────

def stationary_mean(df: pd.DataFrame, col: str, fraction: float = STATIONARY_FRACTION) -> float:
    """Media di una colonna nella fase stazionaria (ultima `fraction` della simulazione)."""
    n = len(df)
    start = int(n * (1 - fraction))
    return df[col].iloc[start:].mean()


def compute_experiment_stats(runs: list[pd.DataFrame]) -> dict:
    """
    Per ogni esperimento calcola:
        - media e std della polarizzazione stazionaria su tutte le run
        - media e std della coesione stazionaria su tutte le run
        - serie temporale media e std (per l'evoluzione temporale)
    """
    if not runs:
        return {
            "polarization_mean": np.nan,
            "polarization_std":  np.nan,
            "cohesion_mean":     np.nan,
            "cohesion_std":      np.nan,
            "timeseries":        None,
        }

    # valori stazionari per run
    pol_vals = [stationary_mean(r, "polarization") for r in runs]
    coh_vals = [stationary_mean(r, "cohesion")     for r in runs]

    # serie temporale: allinea tutte le run sullo stesso asse tick
    # usa il tick set più comune come riferimento
    tick_sets = [set(r["tick"].values) for r in runs]
    common_ticks = sorted(tick_sets[0].intersection(*tick_sets[1:]))

    if common_ticks:
        pol_ts = np.array([r.set_index("tick").loc[common_ticks, "polarization"].values for r in runs])
        coh_ts = np.array([r.set_index("tick").loc[common_ticks, "cohesion"].values     for r in runs])
        timeseries = {
            "ticks":              np.array(common_ticks),
            "polarization_mean":  pol_ts.mean(axis=0),
            "polarization_std":   pol_ts.std(axis=0),
            "cohesion_mean":      coh_ts.mean(axis=0),
            "cohesion_std":       coh_ts.std(axis=0),
        }
    else:
        timeseries = None

    return {
        "polarization_mean": np.nanmean(pol_vals),
        "polarization_std":  np.nanstd(pol_vals),
        "cohesion_mean":     np.nanmean(coh_vals),
        "cohesion_std":      np.nanstd(coh_vals),
        "timeseries":        timeseries,
        "n_runs":            len(runs),
    }


def build_heatmap_matrices(stats: dict[str, dict]) -> dict[str, np.ndarray]:
    """
    Costruisce le matrici 4×4 per le heatmap.
    Righe = RW (0→3), Colonne = ARW (0→3).
    """
    n = len(RW_VALUES)
    pol_mean = np.full((n, n), np.nan)
    pol_std  = np.full((n, n), np.nan)
    coh_mean = np.full((n, n), np.nan)
    coh_std  = np.full((n, n), np.nan)

    for ex_name, (rw, arw) in EXPERIMENTS.items():
        i = RW_VALUES.index(rw)
        j = ARW_VALUES.index(arw)
        s = stats.get(ex_name, {})
        pol_mean[i, j] = s.get("polarization_mean", np.nan)
        pol_std[i, j]  = s.get("polarization_std",  np.nan)
        coh_mean[i, j] = s.get("cohesion_mean",     np.nan)
        coh_std[i, j]  = s.get("cohesion_std",      np.nan)

    return {
        "pol_mean": pol_mean,
        "pol_std":  pol_std,
        "coh_mean": coh_mean,
        "coh_std":  coh_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1: HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(matrices: dict[str, np.ndarray]):
    """
    4 heatmap: polarizzazione media, std polarizzazione,
               coesione media, std coesione.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Metriche stazionarie al variare di RW e ARW", fontsize=15, fontweight="bold")

    configs = [
        (axes[0, 0], matrices["pol_mean"], "Polarizzazione media",  "YlOrRd",   True),
        (axes[0, 1], matrices["pol_std"],  "Polarizzazione std",     "Blues",    False),
        (axes[1, 0], matrices["coh_mean"], "Coesione media",         "YlGnBu_r", True),
        (axes[1, 1], matrices["coh_std"],  "Coesione std",           "Purples",  False),
    ]

    for ax, mat, title, cmap, annotate in configs:
        im = ax.imshow(mat, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Arena Repulsion Weight (ARW)", fontsize=10)
        ax.set_ylabel("Repulsion Weight (RW)", fontsize=10)
        ax.set_xticks(range(len(ARW_VALUES)))
        ax.set_yticks(range(len(RW_VALUES)))
        ax.set_xticklabels(ARW_VALUES)
        ax.set_yticklabels(RW_VALUES)

        # annotazioni con il valore numerico
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color="black" if val < np.nanmax(mat) * 0.7 else "white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_metriche.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2: EVOLUZIONE TEMPORALE
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_evolution(stats: dict[str, dict], metric: str = "polarization"):
    """
    Griglia 4×4 di subplot, uno per esperimento.
    Ogni subplot mostra la media ± std della metrica nel tempo su tutte le run.
    """
    metric_label = {
        "polarization": "Polarizzazione",
        "cohesion":     "Coesione",
    }.get(metric, metric)

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=False, sharey=True)
    fig.suptitle(f"Evoluzione temporale — {metric_label} (media ± std su N run)",
                 fontsize=14, fontweight="bold")

    rw_list = sorted(set(v[0] for v in EXPERIMENTS.values()))
    arw_list = sorted(set(v[1] for v in EXPERIMENTS.values()))

    # mappa (rw, rr) → ax
    ax_map = {}
    for i, rw in enumerate(rw_list):
        for j, arw in enumerate(arw_list):
            ax_map[(rw, arw)] = axes[i, j]

    for ex_name, (rw, arw) in EXPERIMENTS.items():
        ax = ax_map[(rw, arw)]
        s  = stats.get(ex_name, {})
        ts = s.get("timeseries")
        n  = s.get("n_runs", 0)

        ax.set_title(f"RW={rw} ARW={arw}\n(n={n})", fontsize=8)
        ax.set_xlabel("tick", fontsize=7)
        ax.tick_params(labelsize=7)

        if ts is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            continue

        ticks = ts["ticks"]
        mean  = ts[f"{metric}_mean"]
        std   = ts[f"{metric}_std"]

        ax.plot(ticks, mean, color="#1f77b4", linewidth=1.2, label="media")
        ax.fill_between(ticks, mean - std, mean + std,
                         alpha=0.25, color="#1f77b4", label="±std")
        ax.set_ylim(0, 1.05)

        # linea verticale che separa transitorio da stazionario
        cutoff = ticks[int(len(ticks) * (1 - STATIONARY_FRACTION))]
        ax.axvline(cutoff, color="red", linewidth=0.8, linestyle="--", alpha=0.6)

    # etichette sulle righe e colonne della griglia
    for i, rw in enumerate(rw_list):
        axes[i, 0].set_ylabel(f"RW={rw}\n{metric_label}", fontsize=8)
    for j, arw in enumerate(arw_list):
        axes[0, j].set_title(f"ARW={arw}", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"evoluzione_temporale_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 3: BOXPLOT PER ESPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots(data: dict[str, list[pd.DataFrame]]):
    """
    Due boxplot (polarizzazione e coesione) con un box per esperimento,
    ordinati per numero di esperimento.
    I valori sono quelli stazionari per ogni run.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle("Distribuzione delle metriche stazionarie per esperimento",
                 fontsize=13, fontweight="bold")

    labels = []
    pol_data = []
    coh_data = []

    for ex_name in sorted(EXPERIMENTS.keys()):
        rw, arw = EXPERIMENTS[ex_name]
        runs = data.get(ex_name, [])
        pol_vals = [stationary_mean(r, "polarization") for r in runs if len(r) > 0]
        coh_vals = [stationary_mean(r, "cohesion")     for r in runs if len(r) > 0]
        labels.append(f"{ex_name}\nRW={rw} ARW={arw}")
        pol_data.append(pol_vals if pol_vals else [np.nan])
        coh_data.append(coh_vals if coh_vals else [np.nan])

    x = range(len(labels))

    bp1 = ax1.boxplot(pol_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#AED6F1")
        patch.set_alpha(0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Polarizzazione", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="soglia 0.5")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    bp2 = ax2.boxplot(coh_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#A9DFBF")
        patch.set_alpha(0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax2.set_ylabel("Coesione (dist. media dal centroide)", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplot_metriche.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 4: EVOLUZIONE TEMPORALE — CONFRONTO PER RIGA RW FISSA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rw_comparison(stats: dict[str, dict], metric: str = "polarization"):
    """
    4 subplot, uno per valore di RW.
    Ogni subplot mostra le 4 curve (una per ARW) della metrica nel tempo.
    Utile per vedere come varia ARW a parità di RW.
    """
    metric_label = {
        "polarization": "Polarizzazione",
        "cohesion":     "Coesione",
    }.get(metric, metric)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle(f"Effetto di ARW su {metric_label} al variare di RW",
                 fontsize=13, fontweight="bold")

    for idx, rw in enumerate(RW_VALUES):
        ax = axes[idx]
        ax.set_title(f"RW = {rw}", fontsize=11, fontweight="bold")
        ax.set_xlabel("tick", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        for cidx, arw in enumerate(ARW_VALUES):
            # trova l'esperimento corrispondente
            ex_name = next((k for k, v in EXPERIMENTS.items() if v == [rw, arw]), None)
            if ex_name is None:
                continue
            s  = stats.get(ex_name, {})
            ts = s.get("timeseries")
            n  = s.get("n_runs", 0)
            if ts is None:
                continue
            ticks = ts["ticks"]
            mean  = ts[f"{metric}_mean"]
            std   = ts[f"{metric}_std"]
            ax.plot(ticks, mean, color=colors[cidx], linewidth=1.5,
                    label=f"ARW={arw} (n={n})")
            ax.fill_between(ticks, mean - std, mean + std,
                             alpha=0.15, color=colors[cidx])

        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confronto_rw_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Analisi Allocentric Flocking")
    print("=" * 60)

    # 1. Carica dati
    data = load_all_experiments()

    # 2. Calcola statistiche per esperimento
    print("\nCalcolo statistiche...")
    stats = {ex: compute_experiment_stats(runs) for ex, runs in data.items()}

    # Stampa riepilogo
    print("\nRiepilogo metriche stazionarie:")
    print(f"{'Exp':<8} {'RW':>5} {'ARW':>5} {'Pol_mean':>10} {'Pol_std':>9} {'Coh_mean':>10} {'N_runs':>7}")
    print("-" * 60)
    for ex in sorted(stats.keys()):
        rw, arw = EXPERIMENTS[ex]
        s = stats[ex]
        print(f"{ex:<8} {rw:>5.1f} {arw:>5.1f} "
              f"{s['polarization_mean']:>10.4f} "
              f"{s['polarization_std']:>9.4f} "
              f"{s['cohesion_mean']:>10.4f} "
              f"{s.get('n_runs', 0):>7d}")

    # 3. Costruisce matrici per heatmap
    print("\nGenerazione figure...")
    matrices = build_heatmap_matrices(stats)

    # 4. Figure
    print("  [1/4] Heatmap metriche stazionarie")
    plot_heatmaps(matrices)

    print("  [2/4] Evoluzione temporale — polarizzazione")
    plot_temporal_evolution(stats, metric="polarization")

    print("  [3/4] Evoluzione temporale — coesione")
    plot_temporal_evolution(stats, metric="cohesion")

    print("  [4/4] Boxplot distribuzione per esperimento")
    plot_boxplots(data)

    print("  [extra] Confronto RW — polarizzazione")
    plot_rw_comparison(stats, metric="polarization")

    print("  [extra] Confronto RW — coesione")
    plot_rw_comparison(stats, metric="cohesion")

    print(f"\nTutte le figure salvate in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
