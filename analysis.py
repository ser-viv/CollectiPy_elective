"""
analysis.py
===========
Analysis of the allocentric flocking experiments (EW x RW grid at fixed AW).

Expected layout:
    data/EX_01/EX_01/run_1.zip  ->  run_1/group_metrics.csv
                                    run_1/movable_0.m#<i>.pkl   (positions)
    ...
    data/EX_25/EX_25/run_N.zip

Parameters per experiment:
    AW = align_weight       fixed to 0.2 for the whole batch
    EW = edge_weight        in {0, 0.25, 0.5, 0.75, 1}
    RW = repulsion_weight   in {0, 0.25, 0.5, 0.75, 1}

Style, colour scales and fonts are defined in the STYLE section below
(identical in mini_analysis.py). The METRICS are
computed exactly as in the previous version of this script; only the
figures have changed.

Sources
-------
* group_metrics.csv (written by the simulator, 1 row per simulated second):
    - polarization  Phi_body = |mean_i (cos theta_i, sin theta_i)|, from the
      agents' BODY orientation
    - cohesion      C = 1 - mean_i |r_i - c| / D_arena
* movable_0.m#<i>.pkl (raw positions, 1 snapshot per simulated second).
  All the additional metrics below are computed here, offline, from the
  positions: no new simulation is needed.

Metrics
-------
Polarization
    Phi_body      as in the CSV (body orientation)
    Phi_vel       |mean_i u_i|, u_i = unit displacement over VELOCITY_LAG
                  snapshots; agents that moved less than MIN_DISPLACEMENT
                  contribute u_i = 0 (an agent turning on the spot does not
                  count as aligned)
    Phi*          (Phi - Phi_0) / (1 - Phi_0), Phi_0 = chance level for N
                  random headings (Monte Carlo, ~0.40 for N = 5)
    P(Phi > 0.8)  fraction of stationary time spent in an ordered state
    Milling       Couzin et al. (2002): M = |mean_i (r_hat_i x u_i)|, with
                  r_hat_i the unit vector from the rotation centre to agent i.
                  Computed around the group centroid (M_group, classic
                  milling) and around the arena centre (M_arena, circling
                  along the wall).
    Phi_vel far from wall   Phi_vel restricted to snapshots in which every
                  agent is farther than WALL_MARGIN_BL body lengths from the
                  wall (excludes forced turns at the boundary)
Cohesion / group structure (BL = AGENT_DIAMETER = 0.033 m)
    C             as in the CSV
    C*            1 - d / d_random, d = mean distance from centroid,
                  d_random = same quantity for N agents uniformly placed in
                  the arena (Monte Carlo). 0 = random placement, 1 = compact
    d_centroid    mean distance from centroid, in BL
    R_g           radius of gyration sqrt(mean |r_i - c|^2), in BL
    d_pair        mean distance over all agent pairs, in BL
    Largest cluster   fraction of agents in the largest connected cluster,
                  two agents being connected if closer than
                  CLUSTER_THRESHOLD_BL body lengths
    Near-wall time    fraction of stationary time with at least one agent
                  closer than WALL_MARGIN_BL body lengths to the wall
Crowding
    NND           mean nearest-neighbour distance (m)
    Collision rate    fraction of stationary snapshots with min pairwise
                  distance < AGENT_DIAMETER

All stationary quantities are averaged over the last STATIONARY_FRACTION of
each run, then averaged over runs.

Outputs (in OUTPUT_DIR)
-----------------------
Figures for the main text:
    heatmap_stationary_metrics.png   Phi (top) and C (bottom), with Phi* and C*
                                     on the same colour bars
    heatmap_structure.png            largest cluster, milling around the arena
                                     centre, fraction of time with Phi > 0.8
    distribution_polarisation.png    distribution of stationary Phi, 5 RW at EW = 0.5
    temporal_combined.png            Phi and C over time, 2 x 5 (columns EW,
                                     curves RW), moving average + 95% CI
    boxplot_metrics.png              per-run stationary Phi and C
    heatmap_collisions.png           mean NND and collision rate
    boxplot_collisions.png           per-run NND and collision rate
Figures for the appendix:
    appendix_structure.png           Phi_v, Phi_v away from the wall, distances
                                     in BL, milling around the centroid,
                                     near-wall time
    appendix_distributions.png       distributions of Phi for every EW (1 x 5)
Tables:
    summary_metrics.csv, random_baselines.txt
"""

import io
import os
import pickle
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = "./data"          # folder containing EX_01 ... EX_25
OUTPUT_DIR = "./figures"

ALIGN_WEIGHT_FIXED = 0.2
EW_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
RW_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

# EX_n -> [align_weight, edge_weight, repulsion_weight] (RW varies fastest)
EXPERIMENTS = {
    f"EX_{i + 1:02d}": [ALIGN_WEIGHT_FIXED, ew, rw]
    for i, (ew, rw) in enumerate(itertools.product(EW_VALUES, RW_VALUES))
}

STATIONARY_FRACTION = 0.5     # last 50% of each run

AGENT_DIAMETER = 0.033        # m, = 1 body length (BL)
ARENA_DIAMETER = 2.0          # m
ARENA_CENTER = None           # (x, y) in m, or None to detect it from the data
N_AGENTS_DEFAULT = 5          # used only if it cannot be read from the data

VELOCITY_LAG = 1              # snapshots (1 snapshot = 1 s = 5 ticks)
MIN_DISPLACEMENT = 0.001      # m over VELOCITY_LAG; below this an agent is "still"
CLUSTER_THRESHOLD_BL = 5.0    # connection threshold for the largest cluster
WALL_MARGIN_BL = 6.0          # 6 BL ~ 0.2 m, range of the wall repulsion
DIST_EW_MAIN = 0.5            # EW shown in the main-text distribution figure

MC_SAMPLES = 200_000          # Monte Carlo samples for the random baselines
RNG_SEED = 0

COMPUTE_POSITION_METRICS = True   # False -> only the CSV metrics

os.makedirs(OUTPUT_DIR, exist_ok=True)

BL = AGENT_DIAMETER
ARENA_RADIUS = ARENA_DIAMETER / 2.0


def find_experiment(ew: float, rw: float) -> str | None:
    """Return the EX_n name for a given (EW, RW) pair."""
    for ex_name, (_, e, r) in EXPERIMENTS.items():
        if e == ew and r == rw:
            return ex_name
    return None


def ordered_experiments() -> list[str]:
    return sorted(EXPERIMENTS.keys(), key=lambda n: int(n.split("_")[1]))


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM BASELINES (Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def random_polarization(n_agents: int, samples: int = MC_SAMPLES) -> float:
    """Expected |mean of n random unit vectors| (chance-level polarization)."""
    rng = np.random.default_rng(RNG_SEED)
    a = rng.uniform(0, 2 * np.pi, size=(samples, n_agents))
    return float(np.abs(np.exp(1j * a).mean(axis=1)).mean())


def random_centroid_distance(n_agents: int, radius: float,
                             samples: int = MC_SAMPLES) -> float:
    """Expected mean distance from the centroid for n points uniform in a disc."""
    rng = np.random.default_rng(RNG_SEED + 1)
    r = radius * np.sqrt(rng.random((samples, n_agents)))
    t = rng.uniform(0, 2 * np.pi, size=(samples, n_agents))
    z = r * np.exp(1j * t)
    return float(np.abs(z - z.mean(axis=1, keepdims=True)).mean())


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def experiment_dir(ex_name: str) -> str | None:
    for d in (os.path.join(BASE_DIR, ex_name, ex_name), os.path.join(BASE_DIR, ex_name)):
        if os.path.isdir(d):
            return d
    return None


def list_runs(ex_name: str) -> list[str]:
    d = experiment_dir(ex_name)
    if d is None:
        return []
    names = [f for f in os.listdir(d) if f.startswith("run_") and f.endswith(".zip")]
    names.sort(key=lambda f: int("".join(ch for ch in f if ch.isdigit()) or 0))
    return [os.path.join(d, f) for f in names]


def _read_pickle_rows(zf: zipfile.ZipFile, name: str) -> list[dict]:
    """Read a record-style .pkl (one pickle per row) and keep the 'row' entries."""
    rows = []
    u = pickle.Unpickler(io.BytesIO(zf.read(name)))
    while True:
        try:
            rows.append(u.load())
        except EOFError:
            break
    return [r["value"] for r in rows if isinstance(r, dict) and r.get("type") == "row"]


def read_csv_metrics(zf: zipfile.ZipFile) -> pd.DataFrame | None:
    names = [n for n in zf.namelist() if n.endswith("group_metrics.csv")]
    if not names:
        return None
    with zf.open(names[0]) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))
    return df if len(df) else None


def read_positions(zf: zipfile.ZipFile) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Positions of all agents from the movable_*.pkl files (the *_spins files
    are skipped). Agents are aligned on the tick column.
    Returns (ticks (T,), pos (T, N, 2)) or None.
    """
    files = sorted(n for n in zf.namelist() if n.endswith(".pkl") and "_spins" not in n)
    if not files:
        return None
    per_agent = []
    for name in files:
        rows = _read_pickle_rows(zf, name)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if not {"pos x", "pos y"}.issubset(df.columns):
            return None
        if "tick" not in df.columns:
            df["tick"] = np.arange(len(df))
        per_agent.append(df.drop_duplicates("tick").set_index("tick")[["pos x", "pos y"]])
    common = per_agent[0].index
    for df in per_agent[1:]:
        common = common.intersection(df.index)
    common = common.sort_values()
    if len(common) <= VELOCITY_LAG + 1:
        return None
    pos = np.stack([df.loc[common].to_numpy(dtype=float) for df in per_agent], axis=1)
    return np.asarray(common), pos


def resolve_center(pos: np.ndarray) -> np.ndarray:
    """Arena centre: configured, origin if the data fit around it, else midpoint."""
    if ARENA_CENTER is not None:
        return np.asarray(ARENA_CENTER, dtype=float)
    if np.nanmax(np.hypot(pos[..., 0], pos[..., 1])) <= ARENA_RADIUS * 1.05:
        return np.zeros(2)
    flat = pos.reshape(-1, 2)
    return (np.nanmin(flat, axis=0) + np.nanmax(flat, axis=0)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PER-RUN METRICS
# ─────────────────────────────────────────────────────────────────────────────

def stationary_slice(n: int, fraction: float = STATIONARY_FRACTION) -> slice:
    return slice(int(n * (1 - fraction)), n)


def largest_cluster_fraction(dist: np.ndarray, threshold: float) -> np.ndarray:
    """dist: (T, N, N). Fraction of agents in the largest connected component."""
    t, n, _ = dist.shape
    adj = (dist < threshold).astype(np.int64)
    adj[:, np.arange(n), np.arange(n)] = 1
    reach = adj.copy()
    for _ in range(n - 1):  # transitive closure (N is small)
        reach = (np.matmul(reach, adj) > 0).astype(np.int64)
    return reach.sum(axis=2).max(axis=1) / n


def milling(rel: np.ndarray, unit_v: np.ndarray) -> np.ndarray:
    """Couzin et al. (2002) rotation order parameter |mean_i (r_hat_i x u_i)|."""
    norm = np.linalg.norm(rel, axis=2, keepdims=True)
    r_hat = np.divide(rel, norm, out=np.zeros_like(rel), where=norm > 1e-12)
    cross = r_hat[..., 0] * unit_v[..., 1] - r_hat[..., 1] * unit_v[..., 0]
    return np.abs(cross.mean(axis=1))


def position_series(ticks: np.ndarray, pos: np.ndarray, d_random: float) -> pd.DataFrame:
    """Time series of all position-based metrics for one run."""
    lag = VELOCITY_LAG
    disp = pos[lag:] - pos[:-lag]                # (T', N, 2)
    p = pos[lag:]                                # positions at the end of each interval
    n_agents = p.shape[1]
    center = resolve_center(pos)

    speed = np.linalg.norm(disp, axis=2, keepdims=True)
    unit_v = np.divide(disp, speed, out=np.zeros_like(disp), where=speed > MIN_DISPLACEMENT)
    pol_vel = np.linalg.norm(unit_v.mean(axis=1), axis=1)

    c = p.mean(axis=1, keepdims=True)
    rel = p - c
    dist_c = np.linalg.norm(rel, axis=2)
    d_centroid = dist_c.mean(axis=1)
    r_g = np.sqrt((dist_c ** 2).mean(axis=1))

    diff = p[:, :, None, :] - p[:, None, :, :]
    dist = np.linalg.norm(diff, axis=3)
    iu = np.triu_indices(n_agents, 1)
    pair = dist[:, iu[0], iu[1]]
    d_pair = pair.mean(axis=1)
    min_dist = pair.min(axis=1)
    dist_nn = dist.copy()
    dist_nn[:, np.arange(n_agents), np.arange(n_agents)] = np.inf
    nnd = dist_nn.min(axis=2).mean(axis=1)
    largest = largest_cluster_fraction(dist, CLUSTER_THRESHOLD_BL * BL)

    m_group = milling(rel, unit_v)
    m_arena = milling(p - center, unit_v)

    wall_dist = ARENA_RADIUS - np.linalg.norm(p - center, axis=2)
    near_wall = (wall_dist < WALL_MARGIN_BL * BL).any(axis=1)

    return pd.DataFrame({
        "tick":          ticks[lag:],
        "pol_vel":       pol_vel,
        "cohesion_star": 1.0 - d_centroid / d_random,
        "d_centroid_bl": d_centroid / BL,
        "rg_bl":         r_g / BL,
        "d_pair_bl":     d_pair / BL,
        "largest_cluster": largest,
        "milling_group": m_group,
        "milling_arena": m_arena,
        "near_wall":     near_wall.astype(float),
        "nnd_m":         nnd,
        "collision":     (min_dist < AGENT_DIAMETER).astype(float),
    })


def summarize_csv(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    s = stationary_slice(len(df))
    pol = df["polarization"].to_numpy(dtype=float)
    coh = df["cohesion"].to_numpy(dtype=float)
    scalars = {
        "pol_body":         float(np.nanmean(pol[s])),
        "cohesion":         float(np.nanmean(coh[s])),
        "pol_body_ordered": float(np.nanmean(pol[s] > ORDER_THRESHOLD)),
    }
    return scalars, pol[s]


def summarize_positions(ts: pd.DataFrame) -> tuple[dict, np.ndarray]:
    st = ts.iloc[stationary_slice(len(ts))]
    far = st.loc[st["near_wall"] == 0, "pol_vel"]
    scalars = {
        "pol_vel":          float(st["pol_vel"].mean()),
        "pol_vel_ordered":  float((st["pol_vel"] > ORDER_THRESHOLD).mean()),
        "pol_vel_far_wall": float(far.mean()) if len(far) else np.nan,
        "cohesion_star":    float(st["cohesion_star"].mean()),
        "d_centroid_bl":    float(st["d_centroid_bl"].mean()),
        "rg_bl":            float(st["rg_bl"].mean()),
        "d_pair_bl":        float(st["d_pair_bl"].mean()),
        "largest_cluster":  float(st["largest_cluster"].mean()),
        "milling_group":    float(st["milling_group"].mean()),
        "milling_arena":    float(st["milling_arena"].mean()),
        "near_wall":        float(st["near_wall"].mean()),
        "nnd_m":            float(st["nnd_m"].mean()),
        "collision_rate":   float(st["collision"].mean()),
    }
    return scalars, st["pol_vel"].to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# PER-EXPERIMENT AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def process_experiment(ex_name: str, d_random: float) -> dict:
    per_run = []                       # list of dicts of scalars
    csv_runs = []                      # raw CSVs, for the time series
    stationary_values = {"pol_body": [], "pol_vel": []}

    for zip_path in list_runs(ex_name):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv = read_csv_metrics(zf)
                positions = read_positions(zf) if COMPUTE_POSITION_METRICS else None
        except Exception as e:
            print(f"  [WARN] cannot read {zip_path}: {e}")
            continue
        run = {}
        if csv is not None:
            sc, sv = summarize_csv(csv)
            run.update(sc)
            csv_runs.append(csv.drop_duplicates("tick").reset_index(drop=True))
            stationary_values["pol_body"].append(sv)
        if positions is not None:
            ts = position_series(*positions, d_random=d_random)
            sc, sv = summarize_positions(ts)
            run.update(sc)
            stationary_values["pol_vel"].append(sv)
        if run:
            per_run.append(run)

    runs_df = pd.DataFrame(per_run)
    return {
        "n_runs":     len(per_run),
        "runs":       runs_df,
        "mean":       runs_df.mean(numeric_only=True).to_dict() if len(runs_df) else {},
        "std":        runs_df.std(ddof=0, numeric_only=True).to_dict() if len(runs_df) else {},
        # smoothed mean and 95% CI of Phi and C over time (see build_timeseries)
        "timeseries": build_timeseries(csv_runs, STATIONARY_FRACTION),
        "stationary_values": {k: (np.concatenate(v) if v else np.array([]))
                              for k, v in stationary_values.items()},
    }


def add_chance_corrected(stats: dict, phi0: float):
    """Phi* = (Phi - Phi_0) / (1 - Phi_0), per run and per experiment."""
    for s in stats.values():
        runs = s["runs"]
        for src in ("pol_body", "pol_vel"):
            if src in runs:
                runs[f"{src}_star"] = (runs[src] - phi0) / (1 - phi0)
                s["mean"][f"{src}_star"] = float(runs[f"{src}_star"].mean())
                s["std"][f"{src}_star"] = float(runs[f"{src}_star"].std(ddof=0))


def matrix(stats: dict, key: str) -> np.ndarray:
    m = np.full((len(EW_VALUES), len(RW_VALUES)), np.nan)
    for ex_name, (_, ew, rw) in EXPERIMENTS.items():
        m[EW_VALUES.index(ew), RW_VALUES.index(rw)] = stats[ex_name]["mean"].get(key, np.nan)
    return m


def per_cell(stats: dict, key: str) -> dict[tuple, list]:
    """Per-run values of `key` for every (EW, RW) cell (for the boxplots)."""
    out = {}
    for ew in EW_VALUES:
        for rw in RW_VALUES:
            runs = stats[find_experiment(ew, rw)]["runs"]
            out[(ew, rw)] = runs[key].dropna().tolist() if key in runs else []
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def _path(name: str) -> str:
    return os.path.join(OUTPUT_DIR, name)


def plot_heatmaps(stats: dict):
    """Phi (top) and C (bottom), each with its chance-corrected scale on the colour bar."""
    fig, axes = plt.subplots(2, 1, figsize=(0.8 * FIG_WIDTH, 5.6))
    for ax, key, style, xlabel in ((axes[0], "pol_body", "polarization", None),
                                   (axes[1], "cohesion", "cohesion", "Repulsion weight $RW$")):
        mat = matrix(stats, key)
        im = annotated_heatmap(ax, mat, style, RW_VALUES, EW_VALUES, xlabel=xlabel, aspect="auto")
        style_colorbar(fig, im, ax, style, mat)
    fig.tight_layout()
    save(fig, _path("heatmap_stationary_metrics.png"))


def plot_heatmap_rows(rows: list[tuple], name: str, ncols: int = 3):
    """
    Grid of heatmaps, one row per family of metrics with a single colour bar
    per row (metrics in a row share the scale). Each row is an independent
    sub-figure, so all panels have the same width.
    rows = [(style_key, colour_bar_label, [(key, title), ...], fmt), ...]
    """
    fig = plt.figure(figsize=(FIG_WIDTH, 2.5 * len(rows)), layout="constrained")
    subfigs = np.atleast_1d(fig.subfigures(len(rows), 1))
    for sub, (style, cb_label, panels, fmt) in zip(subfigs, rows):
        axes = np.atleast_1d(sub.subplots(1, ncols))
        im, mats = None, []
        for c, ax in enumerate(axes):
            if c >= len(panels):
                ax.set_axis_off()
                continue
            key, title = panels[c]
            mat = matrix(_STATS, key)
            mats.append(mat)
            im = annotated_heatmap(ax, mat, style, RW_VALUES, EW_VALUES,
                                   xlabel="$RW$", ylabel="$EW$" if c == 0 else None,
                                   title=title, aspect="auto", fmt=fmt)
        style_colorbar(sub, im, list(axes[:len(panels)]), style, mats,
                       label=cb_label, fraction=0.05, pad=0.03)
    save(fig, _path(name))


def plot_structure():
    """Main text, 1 x 3: largest cluster, milling around the arena centre, Phi > 0.8."""
    plot_heatmap_rows([
        ("fraction", "Fraction", [
            ("largest_cluster", "Largest-cluster\nfraction"),
            ("milling_arena", "Milling around\narena centre"),
            ("pol_body_ordered", f"Fraction of time\nwith $\\Phi > {ORDER_THRESHOLD:g}$"),
        ], "{:.2f}"),
    ], "heatmap_structure.png")


def plot_appendix_structure():
    plot_heatmap_rows([
        ("polarization", r"Velocity polarisation $\Phi_v$", [
            ("pol_vel", r"$\Phi_v$"),
            ("pol_vel_far_wall", "$\\Phi_v$ away\nfrom wall"),
        ], "{:.2f}"),
        ("distance_bl", "Distance (BL)", [
            ("d_centroid_bl", "Mean distance\nfrom centroid"),
            ("rg_bl", "Radius of\ngyration"),
            ("d_pair_bl", "Mean pairwise\ndistance"),
        ], None),
        ("fraction", "Milling / fraction of time", [
            ("milling_group", "Milling around\ngroup centroid"),
            ("near_wall", f"Time within\n{WALL_MARGIN_BL:g} BL of wall"),
        ], "{:.2f}"),
    ], "appendix_structure.png")


def plot_collision_heatmaps(stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.2))
    for ax, key, style in ((axes[0], "nnd_m", "nnd"), (axes[1], "collision_rate", "collision_rate")):
        mat = matrix(stats, key)
        im = annotated_heatmap(ax, mat, style, RW_VALUES, EW_VALUES)
        style_colorbar(fig, im, ax, style, mat)
    fig.tight_layout()
    save(fig, _path("heatmap_collisions.png"))


def plot_temporal_combined(stats: dict):
    plot_temporal_grid(lambda ew, rw: stats[find_experiment(ew, rw)]["timeseries"],
                       EW_VALUES, RW_VALUES, "$EW$", "$RW$", _path("temporal_combined.png"))


DIST_BINS = np.linspace(0.0, 1.0, 51)


def _plot_distributions(ax, stats: dict, ew: float):
    colours = series_colours(len(RW_VALUES))
    centres = 0.5 * (DIST_BINS[1:] + DIST_BINS[:-1])
    for k, rw in enumerate(RW_VALUES):
        vals = stats[find_experiment(ew, rw)]["stationary_values"].get("pol_body", np.array([]))
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        dens, _ = np.histogram(vals, bins=DIST_BINS, density=True)
        ax.plot(centres, dens, color=colours[k], linewidth=1.3, label=f"{rw:g}")
    ax.axvline(PHI_CHANCE, color="0.35", linewidth=0.8, linestyle=":")
    ax.axvline(ORDER_THRESHOLD, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)


def plot_distributions(stats: dict):
    # main text: one value of EW
    fig, ax = plt.subplots(figsize=(0.6 * FIG_WIDTH, 2.8))
    _plot_distributions(ax, stats, DIST_EW_MAIN)
    ax.set_xlabel(METRIC_STYLE["polarization"]["label"])
    ax.set_ylabel("Probability density")
    ax.set_title(f"$EW$ = {DIST_EW_MAIN:g}")
    ax.legend(title="$RW$", frameon=False)
    fig.tight_layout()
    save(fig, _path("distribution_polarisation.png"))

    # appendix: every EW, 1 x 5
    fig, axes = plt.subplots(1, len(EW_VALUES), figsize=(FIG_WIDTH, 2.3), sharey=True)
    for c, ew in enumerate(EW_VALUES):
        _plot_distributions(axes[c], stats, ew)
        axes[c].set_title(f"$EW$ = {ew:g}")
        axes[c].set_xticks([0, 0.4, 0.8])
    axes[0].set_ylabel("Probability density")
    axes[len(EW_VALUES) // 2].set_xlabel(METRIC_STYLE["polarization"]["label"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, title="$RW$", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    save(fig, _path("appendix_distributions.png"))


def plot_boxplots(stats: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 5.2))
    lo, hi = grouped_boxplot(ax1, per_cell(stats, "pol_body"), EW_VALUES, RW_VALUES,
                             "polarization", METRIC_STYLE["polarization"]["label"], "$EW$")
    ax1.axhline(PHI_CHANCE, color="0.35", linewidth=0.8, linestyle="--")
    ax1.set_ylim(lo - 0.04 * (hi - lo), hi)
    lo, hi = grouped_boxplot(ax2, per_cell(stats, "cohesion"), EW_VALUES, RW_VALUES,
                             "cohesion", METRIC_STYLE["cohesion"]["label"], "$EW$")
    ax2.set_ylim(lo, hi)
    fig.tight_layout()
    series_legend(fig, RW_VALUES, "$RW$")
    save(fig, _path("boxplot_metrics.png"))


def plot_collision_boxplots(stats: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 5.2))
    lo, hi = grouped_boxplot(ax1, per_cell(stats, "nnd_m"), EW_VALUES, RW_VALUES,
                             "nnd", METRIC_STYLE["nnd"]["label"], "$EW$")
    ax1.axhline(AGENT_DIAMETER, color="#c0392b", linewidth=0.9, linestyle="--",
                label=f"Body diameter ({AGENT_DIAMETER} m)")
    ax1.set_ylim(lo, hi)
    ax1.legend(loc="upper left", frameon=False)
    lo, hi = grouped_boxplot(ax2, per_cell(stats, "collision_rate"), EW_VALUES, RW_VALUES,
                             "collision_rate", METRIC_STYLE["collision_rate"]["label"], "$EW$")
    ax2.set_ylim(lo, hi)
    fig.tight_layout()
    series_legend(fig, RW_VALUES, "$RW$")
    save(fig, _path("boxplot_collisions.png"))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_KEYS = [
    "pol_body", "pol_body_star", "pol_body_ordered",
    "pol_vel", "pol_vel_star", "pol_vel_ordered", "pol_vel_far_wall",
    "milling_group", "milling_arena",
    "cohesion", "cohesion_star", "d_centroid_bl", "rg_bl", "d_pair_bl",
    "largest_cluster", "near_wall", "nnd_m", "collision_rate",
]

_STATS: dict = {}   # filled in main(), read by plot_heatmap_rows


def detect_n_agents() -> int:
    for ex in ordered_experiments():
        for zip_path in list_runs(ex):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    n = len([n for n in zf.namelist() if n.endswith(".pkl") and "_spins" not in n])
                if n > 1:
                    return n
            except Exception:
                continue
    return N_AGENTS_DEFAULT


def main():
    print("=" * 70)
    print(f"  Allocentric flocking analysis (AW = {ALIGN_WEIGHT_FIXED}, EW x RW 5x5 grid)")
    print("=" * 70)

    n_agents = detect_n_agents()
    phi0 = random_polarization(n_agents)
    d_random = random_centroid_distance(n_agents, ARENA_RADIUS)
    d_random_inner = random_centroid_distance(n_agents, ARENA_RADIUS - WALL_MARGIN_BL * BL)
    print(f"\nRandom baselines for N = {n_agents}:")
    print(f"  chance polarization Phi_0           = {phi0:.3f}")
    print(f"  random mean distance from centroid  = {d_random:.3f} m "
          f"(C = {1 - d_random / ARENA_DIAMETER:.3f})")
    print(f"  same, excluding {WALL_MARGIN_BL:g} BL wall band = {d_random_inner:.3f} m "
          f"(C = {1 - d_random_inner / ARENA_DIAMETER:.3f})")
    # the colour bars use the reference values in the STYLE section: check they match
    if abs(phi0 - PHI_CHANCE) > 0.005 or abs(d_random - D_RAND) > 0.005:
        print(f"  [WARN] baselines differ from the STYLE section (PHI_CHANCE={PHI_CHANCE}, "
              f"D_RAND={D_RAND}): update them there so that Phi* and C* on the colour "
              f"bars match the values in the tables")

    print("\nLoading and processing runs...")
    stats = {}
    for ex in ordered_experiments():
        stats[ex] = process_experiment(ex, d_random)
        print(f"  {ex}: {stats[ex]['n_runs']} runs")
    add_chance_corrected(stats, phi0)
    _STATS.update(stats)

    # ── summary table (console + CSV) ─────────────────────────────────────
    rows = []
    for ex in ordered_experiments():
        _, ew, rw = EXPERIMENTS[ex]
        row = {"experiment": ex, "EW": ew, "RW": rw, "n_runs": stats[ex]["n_runs"]}
        for k in SUMMARY_KEYS:
            row[f"{k}_mean"] = stats[ex]["mean"].get(k, np.nan)
            row[f"{k}_std"] = stats[ex]["std"].get(k, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False, float_format="%.5f")
    with open(os.path.join(OUTPUT_DIR, "random_baselines.txt"), "w") as fh:
        fh.write(f"N = {n_agents}\nPhi_0 = {phi0:.5f}\n"
                 f"d_random = {d_random:.5f} m (C_random = {1 - d_random / ARENA_DIAMETER:.5f})\n"
                 f"d_random_excluding_wall_band = {d_random_inner:.5f} m "
                 f"(C = {1 - d_random_inner / ARENA_DIAMETER:.5f})\n")

    cols = ["pol_body", "pol_vel", "pol_vel_ordered", "milling_arena",
            "cohesion", "cohesion_star", "rg_bl", "largest_cluster", "collision_rate"]
    print("\nStationary means:")
    print(f"{'Exp':<6}{'EW':>5}{'RW':>6}" + "".join(f"{c[:11]:>12}" for c in cols) + f"{'runs':>6}")
    for r in rows:
        print(f"{r['experiment']:<6}{r['EW']:>5.2f}{r['RW']:>6.2f}"
              + "".join(f"{r.get(c + '_mean', np.nan):>12.3f}" for c in cols)
              + f"{r['n_runs']:>6d}")
    print(f"\n  Saved: {summary_path}")

    ts0 = next((s["timeseries"] for s in stats.values() if s["timeseries"] is not None), None)
    if ts0 is not None:
        print(f"\nMoving average: {ts0['window_samples']} samples (~{SMOOTHING_WINDOW_S:.0f} s); "
              f"stationary window starts at {ts0['cutoff_s']:.0f} s")

    # ── figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_heatmaps(stats)
    plot_temporal_combined(stats)
    plot_distributions(stats)
    plot_boxplots(stats)
    if COMPUTE_POSITION_METRICS:
        plot_structure()
        plot_collision_heatmaps(stats)
        plot_collision_boxplots(stats)
        plot_appendix_structure()

    print(f"\nAll outputs saved in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
