"""
mini_analysis.py
================
Analysis of the exploratory mini-batch for the allocentric flocking
experiments: a coarse, full-factorial screening of all three gains
(align_weight, edge_weight, repulsion_weight) jointly.

Expected folder structure:
    EX_01/EX_01/run_1.zip  ->  run_1/group_metrics.csv
    ...
    EX_27/EX_27/run_1.zip  ->  run_1/group_metrics.csv

CSV columns: tick, group, cohesion, polarization, heading_mean_deg

Parameters per experiment:
    AW = align_weight       in {0, 0.2, 0.5}
    EW = edge_weight        in {0.2, 0.4, 0.6}
    RW = repulsion_weight   in {0.2, 0.5, 1.0}

Style, scales and fonts are defined in the STYLE section below (identical
in analysis.py), so every metric uses the same colour map and range as in
the final batch.

Figures produced (in OUTPUT_DIR):
    heatmap_minibatch.png           Φ (top row) and C (bottom row), one column
                                    per AW, one shared colour bar per row
                                    (Φ/Φ*, C/C*)
    temporal_minibatch_aw<AW>.png   Φ and C over time, 2 x 3 (columns EW,
                                    curves RW), one figure per AW
    boxplot_minibatch.png           per-run stationary Φ and C, 2 x 3
                                    (columns AW, groups EW, colours RW)
"""

import os
import io
import zipfile
import itertools
import warnings

import numpy as np
import pandas as pd

# =============================================================================
# STILE DELLE FIGURE (ex figstyle.py)
# Identico in analysis.py e mini_analysis.py: se cambiate una scala o un colore
# qui, cambiatelo anche nell'altro file, così le figure restano coerenti.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONI E FONT
# ─────────────────────────────────────────────────────────────────────────────

FIG_WIDTH = 7.0     # pollici ≈ larghezza del testo in A4: includere con width=\textwidth
ANNOT_SIZE = 8.5    # valori scritti nelle celle delle heatmap

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# TEMPO E STATISTICA
# ─────────────────────────────────────────────────────────────────────────────

TICK_DT = 0.2               # s per tick (5 Hz)
SMOOTHING_WINDOW_S = 60.0   # media mobile centrata delle serie temporali
CI_Z = 1.96                 # IC al 95% della media

# ─────────────────────────────────────────────────────────────────────────────
# RIFERIMENTI PER LE SCALE CORRETTE PER IL CASO
# ─────────────────────────────────────────────────────────────────────────────

PHI_CHANCE = 0.40                        # E[Φ] per 5 orientazioni indipendenti
ARENA_DIAMETER = 2.0                     # m
D_RAND = 0.583                           # distanza media dal centroide, 5 agenti casuali
C_RAND = 1.0 - D_RAND / ARENA_DIAMETER   # ≈ 0.7085, corrisponde a C* = 0
ORDER_THRESHOLD = 0.8

# ─────────────────────────────────────────────────────────────────────────────
# STILE PER METRICA: stessa colormap e stesso intervallo ovunque
# ─────────────────────────────────────────────────────────────────────────────

METRIC_STYLE = {
    "polarization":   dict(label=r"Polarisation $\Phi$", cmap="viridis",
                           lim=(PHI_CHANCE, 1.0), fmt="{:.3f}"),
    "cohesion":       dict(label=r"Cohesion $C$", cmap="cividis",
                           lim=(C_RAND, 1.0), fmt="{:.3f}"),
    "fraction":       dict(label="Fraction", cmap="magma",
                           lim=(0.0, 1.0), fmt="{:.2f}"),
    "nnd":            dict(label="Mean NND (m)", cmap="Blues",
                           lim=(0.0, 0.5), fmt="{:.3f}"),
    "collision_rate": dict(label="Collision rate", cmap="Reds",
                           lim=(0.0, 1.0), fmt="{:.1%}"),
    "distance_bl":    dict(label="Distance (BL)", cmap="Greens",
                           lim=(0.0, 16.0), fmt="{:.1f}"),
}

# Scale secondarie (trasformazioni lineari) mostrate sulla stessa colorbar
SECONDARY_SCALE = {
    "polarization": (
        r"$\Phi^*$",
        lambda x: (x - PHI_CHANCE) / (1.0 - PHI_CHANCE),
        lambda y: PHI_CHANCE + y * (1.0 - PHI_CHANCE),
    ),
    "cohesion": (
        r"$C^*$",
        lambda c: 1.0 - ARENA_DIAMETER * (1.0 - c) / D_RAND,
        lambda s: 1.0 - (1.0 - s) * D_RAND / ARENA_DIAMETER,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI SUPPORTO
# ─────────────────────────────────────────────────────────────────────────────

def series_colours(n: int) -> np.ndarray:
    """Colori sequenziali per n curve ordinate (es. valori di RW); niente giallo chiaro."""
    return plt.get_cmap("plasma")(np.linspace(0.0, 0.85, n))


def _finite(values) -> np.ndarray:
    return np.asarray([x for x in np.ravel(values) if np.isfinite(x)], dtype=float)


def check_range(values, lim, what: str):
    """Avvisa se dei valori escono dall'intervallo fisso della metrica."""
    v = _finite(values)
    if v.size and (v.min() < lim[0] or v.max() > lim[1]):
        print(f"  [WARN] {what}: valori in [{v.min():.3f}, {v.max():.3f}] "
              f"fuori dall'intervallo fisso {lim}")


def extend_mode(values, lim) -> str:
    v = _finite(values)
    if not v.size:
        return "neither"
    lo, hi = v.min() < lim[0], v.max() > lim[1]
    return "both" if lo and hi else "min" if lo else "max" if hi else "neither"


def text_colour(rgba) -> str:
    r, g, b, _ = rgba
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "white"


def save(fig, path: str):
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP E COLORBAR
# ─────────────────────────────────────────────────────────────────────────────

def annotated_heatmap(ax, mat: np.ndarray, style_key: str, xvals, yvals,
                      xlabel: str | None = "Repulsion weight $RW$",
                      ylabel: str | None = "Edge weight $EW$",
                      title: str | None = None, aspect: str = "equal",
                      fmt: str | None = None):
    """Heatmap con la scala fissa della metrica e i valori scritti nelle celle."""
    st = METRIC_STYLE[style_key]
    lim = st["lim"]
    fmt = fmt or st["fmt"]
    check_range(mat, lim, (title or style_key).replace("\n", " "))

    cmap = plt.get_cmap(st["cmap"])
    norm = mcolors.Normalize(vmin=lim[0], vmax=lim[1])
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect=aspect)

    ax.set_xticks(range(len(xvals)))
    ax.set_yticks(range(len(yvals)))
    ax.set_xticklabels([f"{v:g}" for v in xvals])
    ax.set_yticklabels([f"{v:g}" for v in yvals])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=text_colour(cmap(norm(val))))
    return im


def style_colorbar(fig, im, ax, style_key: str, values, label: str | None = None, **kwargs):
    """
    Colorbar della metrica. Per Φ e C aggiunge la scala corretta per il caso
    (Φ*, C*): scala grezza a sinistra della barra, scala corretta a destra.
    `ax` può essere un singolo asse o una lista (colorbar condivisa).
    """
    st = METRIC_STYLE[style_key]
    has_secondary = style_key in SECONDARY_SCALE
    kwargs.setdefault("fraction", 0.046)
    kwargs.setdefault("pad", 0.16 if has_secondary else 0.04)
    cb = fig.colorbar(im, ax=ax, extend=extend_mode(values, st["lim"]), **kwargs)
    cb.set_label(label or st["label"])
    if style_key == "collision_rate":
        cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if has_secondary:
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_label_position("left")
        lab, f, finv = SECONDARY_SCALE[style_key]
        sec = cb.ax.secondary_yaxis("right", functions=(f, finv))
        sec.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        sec.set_ylabel(lab)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# SERIE TEMPORALI
# ─────────────────────────────────────────────────────────────────────────────

def build_timeseries(runs: list[pd.DataFrame], stationary_fraction: float,
                     metrics=("polarization", "cohesion")) -> dict | None:
    """
    Allinea le run sui tick comuni, liscia ogni run con una media mobile
    centrata di SMOOTHING_WINDOW_S e calcola media e semi-ampiezza
    dell'IC al 95% (CI_Z · σ/√n) tra le run.
    """
    if not runs:
        return None
    tick_sets = [set(r["tick"].values) for r in runs]
    common = sorted(tick_sets[0].intersection(*tick_sets[1:]))
    if len(common) < 2:
        return None

    time_s = np.asarray(common, dtype=float) * TICK_DT
    win = max(1, int(round(SMOOTHING_WINDOW_S / float(np.median(np.diff(time_s))))))
    out = {
        "time_s": time_s,
        "window_samples": win,
        "cutoff_s": time_s[int(len(time_s) * (1 - stationary_fraction))],
    }
    for metric in metrics:
        arr = np.array([r.set_index("tick").loc[common, metric].to_numpy(dtype=float)
                        for r in runs])
        smooth = (pd.DataFrame(arr.T)
                  .rolling(win, center=True, min_periods=1).mean()
                  .to_numpy().T)
        n = smooth.shape[0]
        mean = np.nanmean(smooth, axis=0)
        se = (np.nanstd(smooth, axis=0, ddof=1) / np.sqrt(n)) if n > 1 else np.zeros_like(mean)
        out[f"{metric}_mean"] = mean
        out[f"{metric}_ci"] = CI_Z * se
    return out


def plot_temporal_grid(get_ts, col_values, curve_values, col_name: str,
                       curve_name: str, path: str):
    """
    Figura 2 × len(col_values): Φ sopra, C sotto; una colonna per valore
    di col_values (es. EW); in ogni pannello una curva per valore di
    curve_values (es. RW). get_ts(col, curve) restituisce la serie temporale
    prodotta da build_timeseries, oppure None.
    """
    metrics = ("polarization", "cohesion")
    colours = series_colours(len(curve_values))
    fig, axes = plt.subplots(len(metrics), len(col_values),
                             figsize=(FIG_WIDTH, 3.9), sharex=True, sharey="row",
                             squeeze=False)

    cutoff_s, t_max = None, None
    for r, metric in enumerate(metrics):
        lo, hi = METRIC_STYLE[metric]["lim"]
        pad = 0.04 * (hi - lo)
        for c, cv in enumerate(col_values):
            ax = axes[r, c]
            for k, kv in enumerate(curve_values):
                ts = get_ts(cv, kv)
                if ts is None:
                    continue
                t, m, h = ts["time_s"], ts[f"{metric}_mean"], ts[f"{metric}_ci"]
                ax.fill_between(t, m - h, m + h, color=colours[k], alpha=0.25, linewidth=0)
                ax.plot(t, m, color=colours[k], linewidth=1.1, label=f"{kv:g}")
                cutoff_s, t_max = ts["cutoff_s"], t[-1]

            if cutoff_s is not None:
                ax.axvline(cutoff_s, color="0.35", linewidth=0.8)
            if metric == "polarization":
                ax.axhline(PHI_CHANCE, color="0.35", linewidth=0.8, linestyle="--")
            ax.set_ylim(lo - pad, hi)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(f"{col_name} = {cv:g}")
            if c == 0:
                ax.set_ylabel(METRIC_STYLE[metric]["label"])
            if r == len(metrics) - 1:
                ax.set_xlabel("Time (s)")

    if t_max is not None:
        axes[0, 0].set_xlim(0, t_max)
        for ax in axes[-1]:
            ax.set_xticks([0, 900, 1800] if t_max >= 1800 else [0, t_max / 2, t_max])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, title=curve_name, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# BOXPLOT RAGGRUPPATI
# ─────────────────────────────────────────────────────────────────────────────

def grouped_boxplot(ax, values: dict[tuple, list], groups, series, style_key: str,
                    ylabel: str | None, group_name: str):
    """
    Un gruppo di box per ogni valore di `groups` (es. EW), un box per ogni
    valore di `series` (es. RW) colorato in modo sequenziale.
    values[(g, s)] = lista dei valori per run. Ritorna l'intervallo fisso.
    """
    colours = series_colours(len(series))
    n_s, gap = len(series), 1
    lim = METRIC_STYLE[style_key]["lim"]
    all_vals = []

    for gi, g in enumerate(groups):
        for k, s in enumerate(series):
            vals = [v for v in values.get((g, s), []) if np.isfinite(v)]
            if not vals:
                continue
            all_vals.extend(vals)
            bp = ax.boxplot([vals], positions=[gi * (n_s + gap) + k], widths=0.7,
                            patch_artist=True, showfliers=True, manage_ticks=False,
                            medianprops=dict(color="black", linewidth=1.0),
                            flierprops=dict(marker=".", markersize=3,
                                            markerfacecolor=colours[k],
                                            markeredgecolor="none"))
            bp["boxes"][0].set_facecolor(colours[k])
            bp["boxes"][0].set_alpha(0.85)

    check_range(all_vals, lim, ylabel or style_key)
    ax.set_xticks([gi * (n_s + gap) + (n_s - 1) / 2 for gi in range(len(groups))])
    ax.set_xticklabels([f"{group_name} = {g:g}" for g in groups])
    for gi in range(1, len(groups)):
        ax.axvline(gi * (n_s + gap) - (gap + 1) / 2, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_xlim(-1, len(groups) * (n_s + gap) - gap)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if style_key == "collision_rate":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    return lim


def series_legend(fig, series, title: str):
    """Legenda unica, fuori dai pannelli, per i colori delle serie."""
    handles = [Patch(facecolor=c, alpha=0.85, label=f"{s:g}")
               for c, s in zip(series_colours(len(series)), series)]
    fig.legend(handles=handles, title=title, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)

# =============================================================================
# FINE STILE
# =============================================================================

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

BASE_DIR = "./data"
OUTPUT_DIR = "./figures_mini"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# AW varies slowest, then EW, then RW (RW varies fastest).
ALIGN_VALUES = [0.0, 0.2, 0.5]
EW_VALUES = [0.2, 0.4, 0.6]
RW_VALUES = [0.2, 0.5, 1.0]

EXPERIMENTS = {
    f"EX_{i + 1:02d}": [aw, ew, rw]
    for i, (aw, ew, rw) in enumerate(itertools.product(ALIGN_VALUES, EW_VALUES, RW_VALUES))
}

STATIONARY_FRACTION = 0.5   # last half of the simulation


def find_experiment(aw: float, ew: float, rw: float) -> str | None:
    for ex_name, (a, e, r) in EXPERIMENTS.items():
        if a == aw and e == ew and r == rw:
            return ex_name
    return None


def _path(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_run_metrics(zip_path: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_candidates = [n for n in zf.namelist() if n.endswith("group_metrics.csv")]
            if not csv_candidates:
                return None
            with zf.open(csv_candidates[0]) as f:
                return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] could not read {zip_path}: {e}")
        return None


def load_experiment(ex_name: str) -> list[pd.DataFrame]:
    for ex_dir in (os.path.join(BASE_DIR, ex_name, ex_name), os.path.join(BASE_DIR, ex_name)):
        if os.path.isdir(ex_dir):
            break
    else:
        print(f"  [WARN] folder not found: {ex_name}")
        return []

    runs = []
    for fname in sorted(os.listdir(ex_dir)):
        if fname.startswith("run_") and fname.endswith(".zip"):
            df = load_run_metrics(os.path.join(ex_dir, fname))
            if df is not None and len(df) > 0:
                runs.append(df.drop_duplicates("tick").reset_index(drop=True))
    print(f"  {ex_name}: {len(runs)} runs loaded")
    return runs


def load_all_experiments() -> dict[str, list[pd.DataFrame]]:
    print("Loading data...")
    return {ex_name: load_experiment(ex_name) for ex_name in EXPERIMENTS}


# -----------------------------------------------------------------------------
# AGGREGATE METRICS
# -----------------------------------------------------------------------------

def compute_experiment_stats(runs: list[pd.DataFrame]) -> dict:
    """Per-run stationary Φ and C, their means, and smoothed time series with 95% CI."""
    pol_vals, coh_vals = [], []
    for r in runs:
        start = int(len(r) * (1 - STATIONARY_FRACTION))
        pol = r["polarization"].iloc[start:].to_numpy(dtype=float)
        coh = r["cohesion"].iloc[start:].to_numpy(dtype=float)
        pol, coh = pol[np.isfinite(pol)], coh[np.isfinite(coh)]
        if pol.size:
            pol_vals.append(pol.mean())
        if coh.size:
            coh_vals.append(coh.mean())

    return {
        "polarization_mean": float(np.mean(pol_vals)) if pol_vals else np.nan,
        "polarization_std":  float(np.std(pol_vals)) if pol_vals else np.nan,
        "cohesion_mean":     float(np.mean(coh_vals)) if coh_vals else np.nan,
        "cohesion_std":      float(np.std(coh_vals)) if coh_vals else np.nan,
        "pol_vals":          pol_vals,
        "coh_vals":          coh_vals,
        "timeseries":        build_timeseries(runs, STATIONARY_FRACTION),
        "n_runs":            len(runs),
    }


def build_matrix(stats: dict[str, dict], aw: float, key: str) -> np.ndarray:
    """EW (rows) x RW (columns) matrix of `key` at a given AW."""
    mat = np.full((len(EW_VALUES), len(RW_VALUES)), np.nan)
    for ex_name, (a, ew, rw) in EXPERIMENTS.items():
        if a == aw:
            mat[EW_VALUES.index(ew), RW_VALUES.index(rw)] = stats.get(ex_name, {}).get(key, np.nan)
    return mat


# -----------------------------------------------------------------------------
# FIGURE 1: HEATMAPS (rows = metric, columns = AW, shared colour bar per row)
# -----------------------------------------------------------------------------

def plot_heatmaps(stats: dict[str, dict]):
    rows = (("polarization_mean", "polarization"), ("cohesion_mean", "cohesion"))
    fig, axes = plt.subplots(len(rows), len(ALIGN_VALUES), figsize=(FIG_WIDTH, 4.9),
                             layout="constrained")

    for r, (key, style) in enumerate(rows):
        mats = [build_matrix(stats, aw, key) for aw in ALIGN_VALUES]
        im = None
        for c, (aw, mat) in enumerate(zip(ALIGN_VALUES, mats)):
            im = annotated_heatmap(
                axes[r, c], mat, style, RW_VALUES, EW_VALUES,
                xlabel="Repulsion weight $RW$" if r == len(rows) - 1 else None,
                ylabel="Edge weight $EW$" if c == 0 else None,
                title=f"$AW$ = {aw:g}" if r == 0 else None,
                aspect="auto",
            )
        style_colorbar(fig, im, list(axes[r]), style, mats, fraction=0.04, pad=0.06)

    save(fig, _path("heatmap_minibatch.png"))


# -----------------------------------------------------------------------------
# FIGURE 2: TEMPORAL EVOLUTION (one 2 x 3 figure per AW)
# -----------------------------------------------------------------------------

def plot_temporal(stats: dict[str, dict], aw: float):
    def get_ts(ew, rw):
        ex = find_experiment(aw, ew, rw)
        return stats.get(ex, {}).get("timeseries") if ex else None

    suffix = f"{aw:g}".replace(".", "")
    plot_temporal_grid(get_ts, EW_VALUES, RW_VALUES, "$EW$", "$RW$",
                       _path(f"temporal_minibatch_aw{suffix}.png"))


# -----------------------------------------------------------------------------
# FIGURE 3: BOXPLOTS (rows = metric, columns = AW, groups = EW, colours = RW)
# -----------------------------------------------------------------------------

def plot_boxplots(stats: dict[str, dict]):
    rows = (("pol_vals", "polarization"), ("coh_vals", "cohesion"))
    fig, axes = plt.subplots(len(rows), len(ALIGN_VALUES), figsize=(FIG_WIDTH, 4.8),
                             sharey="row")

    for r, (key, style) in enumerate(rows):
        lo, hi = METRIC_STYLE[style]["lim"]
        for c, aw in enumerate(ALIGN_VALUES):
            ax = axes[r, c]
            values = {(ew, rw): stats[find_experiment(aw, ew, rw)][key]
                      for ew in EW_VALUES for rw in RW_VALUES}
            grouped_boxplot(ax, values, EW_VALUES, RW_VALUES, style,
                            METRIC_STYLE[style]["label"] if c == 0 else None, "$EW$")
            if style == "polarization":
                # at AW = 0 the values lie around chance level: show them
                # just below the common lower limit rather than clipping them
                ax.axhline(PHI_CHANCE, color="0.35", linewidth=0.8, linestyle="--")
                ax.set_ylim(lo - 0.1 * (hi - lo), hi)
            else:
                ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(f"$AW$ = {aw:g}")
            ax.tick_params(axis="x", labelsize=7.5)

    fig.tight_layout()
    series_legend(fig, RW_VALUES, "$RW$")
    save(fig, _path("boxplot_minibatch.png"))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Allocentric Flocking Analysis -- Exploratory Mini-Batch")
    print("  (align_weight x edge_weight x repulsion_weight, 3x3x3 grid)")
    print("=" * 60)

    data = load_all_experiments()

    print("\nComputing statistics...")
    stats = {ex: compute_experiment_stats(runs) for ex, runs in data.items()}

    print("\nStationary metrics summary:")
    print(f"{'Exp':<8} {'AW':>5} {'EW':>5} {'RW':>5} {'Pol_mean':>10} {'Pol_std':>9} "
          f"{'Coh_mean':>10} {'Coh_std':>9} {'N_runs':>7}")
    print("-" * 70)
    for ex in sorted(stats, key=lambda n: int(n.split("_")[1])):
        aw, ew, rw = EXPERIMENTS[ex]
        s = stats[ex]
        print(f"{ex:<8} {aw:>5.2f} {ew:>5.2f} {rw:>5.2f} "
              f"{s['polarization_mean']:>10.4f} {s['polarization_std']:>9.4f} "
              f"{s['cohesion_mean']:>10.4f} {s['cohesion_std']:>9.4f} {s['n_runs']:>7d}")

    print("\nGenerating figures...")
    plot_heatmaps(stats)
    for aw in ALIGN_VALUES:
        plot_temporal(stats, aw)
    plot_boxplots(stats)

    print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
