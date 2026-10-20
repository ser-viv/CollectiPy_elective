"""
analysis.py
===========
Analisi dei dati per gli esperimenti di allocentric flocking.

Struttura attesa:
    EX_01/EX_01/run_1.zip  →  run_1/group_metrics.csv
    EX_02/EX_02/run_1.zip  →  run_1/group_metrics.csv
    ...
    EX_25/EX_25/run_1.zip  →  run_1/group_metrics.csv

Colonne CSV: tick, group, cohesion, polarization, heading_mean_deg

Parametri per esperimento:
    AW = align_weight       fissato a 0.2 per tutta la batch
    EW = edge_weight        ∈ {0, 0.25, 0.5, 0.75, 1}
    RW = repulsion_weight   ∈ {0, 0.25, 0.5, 0.75, 1}

Griglia 5×5 = 25 esperimenti, ad align_weight fisso (vedi EXPERIMENTS).

Analisi prodotte:
    1. Heatmap delle metriche medie (EW × RW) — solo le medie, niente
       pannelli di deviazione standard
    2. Evoluzione temporale con banda di confidenza (media ± std), griglia
       EW × RW
    3. Distribuzione finale delle metriche per esperimento (tutti e 25)
    4. Confronto RW a EW fissa
    5. Collisioni/affollamento tra agenti (EW × RW), calcolate dalle
       posizioni grezze salvate in run_N.zip (movable_0.m#<i>.pkl):
         - NND media (nearest-neighbor distance): distanza media dal
           vicino più vicino, per agente, nella fase stazionaria
         - tasso di collisione: frazione di tick stazionari in cui la
           distanza minima tra due agenti scende sotto AGENT_DIAMETER
"""

import os
import io
import pickle
import zipfile
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────────────────

# Cartella radice che contiene EX_01 ... EX_25
BASE_DIR = "./data"

# align_weight è fissato per l'intera batch: non è più un asse del grid,
# resta solo per etichettare titoli e nomi dei file.
ALIGN_WEIGHT_FIXED = 0.2

# Valori usati per EW e RW (lo stesso range per entrambi)
EW_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
RW_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# Schema parametri: EX_n → [align_weight, edge_weight, repulsion_weight]
# Generato dall'incrocio pieno di EW_VALUES × RW_VALUES, con align_weight
# sempre uguale a ALIGN_WEIGHT_FIXED.
EXPERIMENTS = {
    f"EX_{i + 1:02d}": [ALIGN_WEIGHT_FIXED, ew, rw]
    for i, (ew, rw) in enumerate(itertools.product(EW_VALUES, RW_VALUES))
}

# Frazione finale della simulazione usata per calcolare le metriche stazionarie
# (es. 0.5 = ultima metà della simulazione)
STATIONARY_FRACTION = 0.5

# Diametro reale dell'agente (da config): due agenti sono considerati "in
# collisione"/sovrapposti quando la distanza tra i loro centri scende sotto
# questa soglia.
AGENT_DIAMETER = 0.033

# Calcola anche le metriche di collisione/affollamento dalle posizioni
# grezze (movable_0.m#<i>.pkl dentro ogni run_N.zip). Si può disattivare
# per velocizzare un run di sola CSV (polarizzazione/coesione).
COMPUTE_COLLISIONS = True

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_experiment(ew: float, rw: float) -> str | None:
    """Trova il nome EX_n corrispondente alla coppia (edge_weight, repulsion_weight)."""
    for ex_name, (a, e, r) in EXPERIMENTS.items():
        if e == ew and r == rw:
            return ex_name
    return None


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
# COLLISIONI / AFFOLLAMENTO (dalle posizioni grezze in run_N.zip)
# ─────────────────────────────────────────────────────────────────────────────

def _read_pickle_rows(zf: zipfile.ZipFile, name: str) -> list[dict]:
    """Legge un file .pkl 'a record' (un pickle per riga, come i file
    movable_0.m#<i>.pkl) e ne estrae le righe di tipo 'row'."""
    rows = []
    u = pickle.Unpickler(io.BytesIO(zf.read(name)))
    while True:
        try:
            rows.append(u.load())
        except EOFError:
            break
    return [r["value"] for r in rows if r.get("type") == "row"]


def load_run_positions(zip_path: str) -> np.ndarray | None:
    """
    Carica le posizioni (x, y) di tutti gli agenti da un run_N.zip, leggendo
    i file 'movable_0.m#<i>.pkl' (esclusi quelli '_spins', che contengono gli
    spin del ring attractor e non le posizioni).

    Ritorna un array (n_tick, n_agenti, 2), oppure None se non trovate.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            agent_files = sorted(
                n for n in zf.namelist()
                if n.endswith(".pkl") and "_spins" not in n
            )
            if not agent_files:
                return None
            series = []
            for name in agent_files:
                rows = _read_pickle_rows(zf, name)
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                series.append(df[["pos x", "pos y"]].to_numpy())
            n = min(len(s) for s in series)
            if n == 0:
                return None
            pos = np.stack([s[:n] for s in series], axis=1)  # (n_tick, n_agenti, 2)
            return pos
    except Exception as e:
        print(f"  [WARN] impossibile leggere le posizioni da {zip_path}: {e}")
        return None


def pairwise_distance_series(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    pos: array (n_tick, n_agenti, 2).

    Ritorna, per ogni tick:
        - min_dist:  distanza minima tra una qualunque coppia di agenti
        - nnd_mean:  media, sugli agenti, della distanza dal vicino più
                     vicino (nearest-neighbor distance)
    """
    n_tick, n_agents, _ = pos.shape
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    idx = np.arange(n_agents)
    dist[:, idx, idx] = np.inf  # ignora la distanza di un agente da se stesso
    min_dist = dist.min(axis=(1, 2))
    nnd = dist.min(axis=2)          # (n_tick, n_agenti)
    nnd_mean = nnd.mean(axis=1)
    return min_dist, nnd_mean


def collision_stats_for_run(pos: np.ndarray, fraction: float = STATIONARY_FRACTION) -> dict:
    """Statistiche di collisione/affollamento per una singola run, calcolate
    sulla fase stazionaria (stessa convenzione di STATIONARY_FRACTION usata
    per polarizzazione/coesione)."""
    min_dist, nnd_mean = pairwise_distance_series(pos)
    n = len(min_dist)
    start = int(n * (1 - fraction))
    min_dist_stat = min_dist[start:]
    nnd_stat = nnd_mean[start:]
    return {
        "nnd_mean":       float(np.mean(nnd_stat)),
        "collision_rate": float(np.mean(min_dist_stat < AGENT_DIAMETER)),
    }


def compute_experiment_collision_stats(ex_name: str) -> dict:
    """
    Carica le run di un esperimento e calcola, nella fase stazionaria:
        - nnd_mean:            NND media su run e agenti
        - collision_rate_mean: tasso medio di collisione sulle run
        - nnd_vals / coll_vals: valori per singola run (per i boxplot)
    """
    ex_dir = os.path.join(BASE_DIR, ex_name, ex_name)
    if not os.path.isdir(ex_dir):
        ex_dir = os.path.join(BASE_DIR, ex_name)
    if not os.path.isdir(ex_dir):
        return {"nnd_mean": np.nan, "collision_rate_mean": np.nan,
                "nnd_vals": [], "coll_vals": [], "n_runs": 0}

    nnd_vals, coll_vals = [], []
    for fname in sorted(os.listdir(ex_dir)):
        if fname.startswith("run_") and fname.endswith(".zip"):
            pos = load_run_positions(os.path.join(ex_dir, fname))
            if pos is None:
                continue
            s = collision_stats_for_run(pos)
            nnd_vals.append(s["nnd_mean"])
            coll_vals.append(s["collision_rate"])

    return {
        "nnd_mean":            float(np.nanmean(nnd_vals)) if nnd_vals else np.nan,
        "collision_rate_mean": float(np.nanmean(coll_vals)) if coll_vals else np.nan,
        "nnd_vals":            nnd_vals,
        "coll_vals":           coll_vals,
        "n_runs":              len(nnd_vals),
    }


def load_all_collision_stats() -> dict[str, dict]:
    """Calcola le statistiche di collisione per tutti gli esperimenti."""
    print("Calcolo statistiche di collisione/affollamento (dalle posizioni)...")
    out = {}
    for ex_name in EXPERIMENTS:
        out[ex_name] = compute_experiment_collision_stats(ex_name)
        print(f"  {ex_name}: {out[ex_name]['n_runs']} run con posizioni valide")
    return out


def build_collision_matrices(collision_stats: dict[str, dict]) -> dict[str, np.ndarray]:
    """Matrici 5×5 (EW × RW) di NND media e tasso di collisione medio."""
    n_ew, n_rw = len(EW_VALUES), len(RW_VALUES)
    nnd_mean = np.full((n_ew, n_rw), np.nan)
    coll_mean = np.full((n_ew, n_rw), np.nan)

    for ex_name, (a, ew, rw) in EXPERIMENTS.items():
        i = EW_VALUES.index(ew)
        j = RW_VALUES.index(rw)
        s = collision_stats.get(ex_name, {})
        nnd_mean[i, j] = s.get("nnd_mean", np.nan)
        coll_mean[i, j] = s.get("collision_rate_mean", np.nan)

    return {"nnd_mean": nnd_mean, "collision_rate_mean": coll_mean}


def plot_collision_heatmaps(matrices: dict[str, np.ndarray]):
    """
    Due heatmap affiancate (NND media, tasso di collisione medio),
    indicizzate da EW (righe) × RW (colonne), ad align_weight fisso.
    """
    metric_cols = [
        ("nnd_mean", "NND media (distanza dal vicino più vicino)", "viridis"),
        ("collision_rate_mean", "Tasso di collisione\n(frazione tempo con dist. < diametro agente)", "Reds"),
    ]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(6.8 * len(metric_cols), 5.5))
    fig.suptitle(f"Collisioni/affollamento al variare di edge_weight (EW) e repulsion_weight (RW)\n"
                 f"align_weight fisso = {ALIGN_WEIGHT_FIXED}, soglia collisione = {AGENT_DIAMETER} m",
                 fontsize=14, fontweight="bold")

    for col, (key, title, cmap) in enumerate(metric_cols):
        ax = axes[col]
        mat = matrices[key]
        im = ax.imshow(mat, cmap=cmap, aspect="auto",
                        vmin=np.nanmin(mat) if np.any(~np.isnan(mat)) else 0,
                        vmax=np.nanmax(mat) if np.any(~np.isnan(mat)) else 1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Repulsion Weight (RW)", fontsize=10)
        ax.set_ylabel("Edge Weight (EW)", fontsize=10)
        ax.set_xticks(range(len(RW_VALUES)))
        ax.set_yticks(range(len(EW_VALUES)))
        ax.set_xticklabels(RW_VALUES)
        ax.set_yticklabels(EW_VALUES)

        vmax = np.nanmax(mat) if np.any(~np.isnan(mat)) else 1
        vmin = np.nanmin(mat) if np.any(~np.isnan(mat)) else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.3f}" if key == "nnd_mean" else f"{val:.1%}"
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=9, color="black" if val < vmin + 0.7 * (vmax - vmin) else "white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_collisioni.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


def plot_collision_boxplots(collision_stats: dict[str, dict]):
    """
    Due boxplot (NND e tasso di collisione) con un box per esperimento,
    ordinati per numero di esperimento (quindi raggruppati per EW, poi RW).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle(f"Distribuzione delle metriche di collisione per esperimento "
                 f"(align_weight={ALIGN_WEIGHT_FIXED}, soglia={AGENT_DIAMETER} m)",
                 fontsize=13, fontweight="bold")

    labels = []
    nnd_data = []
    coll_data = []
    ew_boundaries = []

    ordered_names = sorted(EXPERIMENTS.keys(), key=lambda n: int(n.split("_")[1]))
    prev_ew = None
    for idx, ex_name in enumerate(ordered_names):
        align, ew, rw = EXPERIMENTS[ex_name]
        if prev_ew is not None and ew != prev_ew:
            ew_boundaries.append(idx - 0.5)
        prev_ew = ew

        s = collision_stats.get(ex_name, {})
        nnd_vals = s.get("nnd_vals") or [np.nan]
        coll_vals = s.get("coll_vals") or [np.nan]
        labels.append(f"{ex_name}\nEW={ew} RW={rw}")
        nnd_data.append(nnd_vals)
        coll_data.append(coll_vals)

    x = range(len(labels))

    bp1 = ax1.boxplot(nnd_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#D7BDE2")
        patch.set_alpha(0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax1.set_ylabel("NND media (m)", fontsize=10)
    ax1.axhline(AGENT_DIAMETER, color="red", linewidth=0.8, linestyle="--",
                alpha=0.6, label=f"diametro agente ({AGENT_DIAMETER} m)")
    for b in ew_boundaries:
        ax1.axvline(b, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    bp2 = ax2.boxplot(coll_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#F5B7B1")
        patch.set_alpha(0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax2.set_ylabel("Tasso di collisione", fontsize=10)
    ax2.set_ylim(0, max(1.0, np.nanmax([v for vs in coll_data for v in vs] or [0]) * 1.1))
    for b in ew_boundaries:
        ax2.axvline(b, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplot_collisioni.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


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
        - media della polarizzazione stazionaria su tutte le run
        - media della coesione stazionaria su tutte le run
        - serie temporale media e std (serve ai grafici di evoluzione
          temporale, che mostrano una banda ± std: è una cosa diversa
          dalle heatmap di deviazione standard, rimosse)
    """
    if not runs:
        return {
            "polarization_mean": np.nan,
            "cohesion_mean":     np.nan,
            "timeseries":        None,
            "n_runs":            0,
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
        "cohesion_mean":     np.nanmean(coh_vals),
        "timeseries":        timeseries,
        "n_runs":            len(runs),
    }


def build_heatmap_matrices(stats: dict[str, dict]) -> dict[str, np.ndarray]:
    """
    Costruisce le matrici 5×5 (EW × RW) delle metriche medie stazionarie,
    ad align_weight fisso.

    Ritorna: {"pol_mean": ..., "coh_mean": ...}
    """
    n_ew, n_rw = len(EW_VALUES), len(RW_VALUES)
    pol_mean = np.full((n_ew, n_rw), np.nan)
    coh_mean = np.full((n_ew, n_rw), np.nan)

    for ex_name, (a, ew, rw) in EXPERIMENTS.items():
        i = EW_VALUES.index(ew)
        j = RW_VALUES.index(rw)
        s = stats.get(ex_name, {})
        pol_mean[i, j] = s.get("polarization_mean", np.nan)
        coh_mean[i, j] = s.get("cohesion_mean",     np.nan)

    return {"pol_mean": pol_mean, "coh_mean": coh_mean}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 1: HEATMAP (solo medie, niente deviazione standard)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(matrices: dict[str, np.ndarray]):
    """
    Due heatmap affiancate (polarizzazione media, coesione media),
    indicizzate da EW (righe) × RW (colonne), ad align_weight fisso.
    I pannelli di deviazione standard sono stati rimossi.
    """
    metric_cols = [
        ("pol_mean", "Polarizzazione media", "YlOrRd"),
        ("coh_mean", "Coesione media",       "YlGnBu_r"),
    ]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(6.5 * len(metric_cols), 5.5))
    fig.suptitle(f"Metriche stazionarie al variare di edge_weight (EW) e repulsion_weight (RW)\n"
                 f"align_weight fisso = {ALIGN_WEIGHT_FIXED}",
                 fontsize=14, fontweight="bold")

    for col, (key, title, cmap) in enumerate(metric_cols):
        ax = axes[col]
        mat = matrices[key]
        im = ax.imshow(mat, cmap=cmap, aspect="auto",
                        vmin=np.nanmin(mat) if np.any(~np.isnan(mat)) else 0,
                        vmax=np.nanmax(mat) if np.any(~np.isnan(mat)) else 1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Repulsion Weight (RW)", fontsize=10)
        ax.set_ylabel("Edge Weight (EW)", fontsize=10)
        ax.set_xticks(range(len(RW_VALUES)))
        ax.set_yticks(range(len(EW_VALUES)))
        ax.set_xticklabels(RW_VALUES)
        ax.set_yticklabels(EW_VALUES)

        vmax = np.nanmax(mat) if np.any(~np.isnan(mat)) else 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, color="black" if val < vmax * 0.7 else "white")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "heatmap_metriche.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 2: EVOLUZIONE TEMPORALE (griglia EW × RW, ad align_weight fisso)
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_evolution(stats: dict[str, dict], metric: str = "polarization"):
    """
    Griglia 5×5 di subplot (righe = EW, colonne = RW). Ogni subplot mostra
    la media ± std della metrica nel tempo su tutte le run, ad align_weight
    fisso.
    """
    metric_label = {
        "polarization": "Polarizzazione",
        "cohesion":     "Coesione",
    }.get(metric, metric)

    fig, axes = plt.subplots(len(EW_VALUES), len(RW_VALUES),
                              figsize=(3.6 * len(RW_VALUES), 2.8 * len(EW_VALUES)),
                              sharex=False, sharey=True)
    fig.suptitle(f"Evoluzione temporale — {metric_label} "
                 f"(align_weight={ALIGN_WEIGHT_FIXED}, media ± std su N run)",
                 fontsize=13, fontweight="bold")

    for i, ew in enumerate(EW_VALUES):
        for j, rw in enumerate(RW_VALUES):
            ax = axes[i, j]
            ex_name = find_experiment(ew, rw)
            s = stats.get(ex_name, {}) if ex_name else {}
            ts = s.get("timeseries")
            n = s.get("n_runs", 0)

            ax.set_title(f"EW={ew} RW={rw}\n(n={n})", fontsize=7)
            ax.set_xlabel("tick", fontsize=6)
            ax.tick_params(labelsize=6)

            if ts is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            ticks = ts["ticks"]
            mean  = ts[f"{metric}_mean"]
            std   = ts[f"{metric}_std"]

            ax.plot(ticks, mean, color="#1f77b4", linewidth=1.0, label="media")
            ax.fill_between(ticks, mean - std, mean + std,
                             alpha=0.25, color="#1f77b4", label="±std")
            ax.set_ylim(0, 1.05)

            # linea verticale che separa transitorio da stazionario
            cutoff = ticks[int(len(ticks) * (1 - STATIONARY_FRACTION))]
            ax.axvline(cutoff, color="red", linewidth=0.7, linestyle="--", alpha=0.6)

    for i, ew in enumerate(EW_VALUES):
        axes[i, 0].set_ylabel(f"EW={ew}\n{metric_label}", fontsize=7)

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
    ordinati per numero di esperimento (quindi raggruppati per EW, poi RW).
    I valori sono quelli stazionari per ogni run, ad align_weight fisso.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle(f"Distribuzione delle metriche stazionarie per esperimento "
                 f"(align_weight={ALIGN_WEIGHT_FIXED})",
                 fontsize=13, fontweight="bold")

    labels = []
    pol_data = []
    coh_data = []
    ew_boundaries = []  # indici in cui cambia EW, per le linee verticali

    ordered_names = sorted(EXPERIMENTS.keys(), key=lambda n: int(n.split("_")[1]))
    prev_ew = None
    for idx, ex_name in enumerate(ordered_names):
        align, ew, rw = EXPERIMENTS[ex_name]
        if prev_ew is not None and ew != prev_ew:
            ew_boundaries.append(idx - 0.5)
        prev_ew = ew

        runs = data.get(ex_name, [])
        pol_vals = [stationary_mean(r, "polarization") for r in runs if len(r) > 0]
        coh_vals = [stationary_mean(r, "cohesion")     for r in runs if len(r) > 0]
        labels.append(f"{ex_name}\nEW={ew} RW={rw}")
        pol_data.append(pol_vals if pol_vals else [np.nan])
        coh_data.append(coh_vals if coh_vals else [np.nan])

    x = range(len(labels))

    bp1 = ax1.boxplot(pol_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#AED6F1")
        patch.set_alpha(0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax1.set_ylabel("Polarizzazione", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="soglia 0.5")
    for b in ew_boundaries:
        ax1.axvline(b, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    bp2 = ax2.boxplot(coh_data, positions=list(x), widths=0.6,
                       patch_artist=True, showfliers=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#A9DFBF")
        patch.set_alpha(0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    ax2.set_ylabel("Coesione (dist. media dal centroide)", fontsize=10)
    for b in ew_boundaries:
        ax2.axvline(b, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "boxplot_metriche.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvata: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURA 4: EVOLUZIONE TEMPORALE — CONFRONTO RW A EW FISSA
# ─────────────────────────────────────────────────────────────────────────────

def plot_rw_comparison(stats: dict[str, dict], metric: str = "polarization"):
    """
    Una figura con un subplot per valore di EW; ogni subplot mostra le
    curve (una per RW) della metrica nel tempo, ad align_weight fisso.
    """
    metric_label = {
        "polarization": "Polarizzazione",
        "cohesion":     "Coesione",
    }.get(metric, metric)

    colors = plt.cm.viridis(np.linspace(0, 1, len(RW_VALUES)))

    fig, axes = plt.subplots(1, len(EW_VALUES), figsize=(4.2 * len(EW_VALUES), 4.5), sharey=True)
    fig.suptitle(f"Effetto di RW su {metric_label} al variare di EW "
                 f"(align_weight={ALIGN_WEIGHT_FIXED})",
                 fontsize=13, fontweight="bold")

    for idx, ew in enumerate(EW_VALUES):
        ax = axes[idx]
        ax.set_title(f"EW = {ew}", fontsize=11, fontweight="bold")
        ax.set_xlabel("tick", fontsize=9)
        if idx == 0:
            ax.set_ylabel(metric_label, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        for cidx, rw in enumerate(RW_VALUES):
            ex_name = find_experiment(ew, rw)
            if ex_name is None:
                continue
            s  = stats.get(ex_name, {})
            ts = s.get("timeseries")
            n  = s.get("n_runs", 0)
            if ts is None:
                continue
            ticks = ts["ticks"]
            mean  = ts[f"{metric}_mean"]
            ax.plot(ticks, mean, color=colors[cidx], linewidth=1.5,
                    label=f"RW={rw} (n={n})")

        ax.legend(fontsize=7)

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
    print(f"  (align_weight fisso = {ALIGN_WEIGHT_FIXED}, griglia EW x RW 5x5)")
    print("=" * 60)

    # 1. Carica dati
    data = load_all_experiments()

    # 2. Calcola statistiche per esperimento
    print("\nCalcolo statistiche...")
    stats = {ex: compute_experiment_stats(runs) for ex, runs in data.items()}

    # Stampa riepilogo
    print("\nRiepilogo metriche stazionarie:")
    print(f"{'Exp':<8} {'EW':>5} {'RW':>5} {'Pol_mean':>10} {'Coh_mean':>10} {'N_runs':>7}")
    print("-" * 52)
    for ex in sorted(stats.keys(), key=lambda n: int(n.split("_")[1])):
        align, ew, rw = EXPERIMENTS[ex]
        s = stats[ex]
        print(f"{ex:<8} {ew:>5.2f} {rw:>5.2f} "
              f"{s['polarization_mean']:>10.4f} "
              f"{s['cohesion_mean']:>10.4f} "
              f"{s.get('n_runs', 0):>7d}")

    # 3. Costruisce matrici per heatmap
    print("\nGenerazione figure...")
    matrices = build_heatmap_matrices(stats)

    # 4. Figure
    print("  [1/4] Heatmap metriche stazionarie (solo medie, EW x RW)")
    plot_heatmaps(matrices)

    print("  [2/4] Evoluzione temporale — polarizzazione")
    plot_temporal_evolution(stats, metric="polarization")

    print("  [3/4] Evoluzione temporale — coesione")
    plot_temporal_evolution(stats, metric="cohesion")

    print("  [4/4] Boxplot distribuzione per esperimento (tutti e 25)")
    plot_boxplots(data)

    print("  [extra] Confronto RW — polarizzazione")
    plot_rw_comparison(stats, metric="polarization")

    print("  [extra] Confronto RW — coesione")
    plot_rw_comparison(stats, metric="cohesion")

    # 5. Collisioni/affollamento (dalle posizioni grezze)
    if COMPUTE_COLLISIONS:
        collision_stats = load_all_collision_stats()

        print("\nRiepilogo metriche di collisione (fase stazionaria):")
        print(f"{'Exp':<8} {'EW':>5} {'RW':>5} {'NND_mean':>10} {'Coll_rate':>10} {'N_runs':>7}")
        print("-" * 52)
        for ex in sorted(collision_stats.keys(), key=lambda n: int(n.split("_")[1])):
            align, ew, rw = EXPERIMENTS[ex]
            s = collision_stats[ex]
            print(f"{ex:<8} {ew:>5.2f} {rw:>5.2f} "
                  f"{s['nnd_mean']:>10.4f} "
                  f"{s['collision_rate_mean']:>10.2%} "
                  f"{s.get('n_runs', 0):>7d}")

        coll_matrices = build_collision_matrices(collision_stats)
        print("\n  [5/5] Heatmap collisioni (NND e tasso di collisione, EW x RW)")
        plot_collision_heatmaps(coll_matrices)

        print("  [extra] Boxplot collisioni per esperimento (tutti e 25)")
        plot_collision_boxplots(collision_stats)

    print(f"\nTutte le figure salvate in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
