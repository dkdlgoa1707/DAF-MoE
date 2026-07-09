"""
Critical Difference (CD) Diagram
=================================
Reads results/summarize_performance/final_summary.csv, performs the
Friedman test followed by pairwise Wilcoxon signed-rank tests with
Holm step-down correction, and plots a CD diagram for the paper rebuttal.

Usage:
    python analysis/plot_cd_diagram.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV  = "results/summarize_performance/final_summary.csv"
OUTPUT_DIR = "results/summarize_performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model display names (raw CSV → pretty label) ──────────────────────────────
MODEL_NAME_MAP = {
    "daf_moe":       "DAF-MoE (Ours)",
    "ft_transformer":"FT-Transformer",
    "resnet":        "ResNet",
    "mlp":           "MLP",
    # "xgboost":       "XGBoost",
    # "catboost":      "CatBoost",
}

OUR_MODEL = "DAF-MoE (Ours)"


# ── 1. Load & filter ──────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# Exclude TabM (experiments not yet complete)
df = df[df["Model"].str.lower() != "tabm"].copy()

# DL-only comparison: exclude tree-based models
df = df[~df["Model"].str.lower().isin(["xgboost", "catboost"])].copy()

# Map raw model names to display names
df["Model"] = df["Model"].map(MODEL_NAME_MAP).fillna(df["Model"])


# ── 2. Build rank matrix  (rows = datasets, cols = models) ────────────────────
datasets = sorted(df["Dataset"].unique())
models   = sorted(df["Model"].unique())

rank_matrix = pd.DataFrame(index=datasets, columns=models, dtype=float)

for dataset in datasets:
    sub = df[df["Dataset"] == dataset].copy()
    metric = sub["Metric"].iloc[0].upper()

    # Lower RMSE is better → ascending rank; higher ACC/AUPRC is better → descending
    ascending = (metric == "RMSE")
    sub = sub.sort_values("Mean Score", ascending=ascending).reset_index(drop=True)

    for rank_val, row in enumerate(sub.itertuples(), start=1):
        rank_matrix.loc[dataset, row.Model] = float(rank_val)

# Drop columns that have any NaN (datasets where a model has no result)
rank_matrix = rank_matrix.dropna(axis=1)
models_used = list(rank_matrix.columns)

print("=" * 60)
print(f"Datasets  : {len(rank_matrix)}")
print(f"Models    : {models_used}")
print("\nRank matrix:")
print(rank_matrix.to_string())


# ── 3. Friedman test ──────────────────────────────────────────────────────────
rank_cols = [rank_matrix[m].values for m in models_used]
stat, p_friedman = friedmanchisquare(*rank_cols)

print("\n" + "=" * 60)
print(f"Friedman test:  χ² = {stat:.4f},  p = {p_friedman:.6f}")
if p_friedman < 0.05:
    print("→ Significant differences exist (p < 0.05). Proceeding with post-hoc.")
else:
    print("→ No significant global difference (p ≥ 0.05).")


# ── 4. Wilcoxon signed-rank with Holm correction ─────────────────────────────
# posthoc_wilcoxon expects long-format: one row per (dataset, model) observation
rank_long = rank_matrix.reset_index().melt(
    id_vars="index", var_name="Model", value_name="Rank"
).rename(columns={"index": "Dataset"})

p_values = sp.posthoc_wilcoxon(
    rank_long,
    val_col="Rank",
    group_col="Model",
    p_adjust="holm",
    zero_method="wilcox",
)

print("\nWilcoxon-Holm p-value matrix:")
print(p_values.round(4).to_string())


# ── 5. CD diagram ─────────────────────────────────────────────────────────────
avg_ranks = rank_matrix.mean().sort_values()   # ascending → best rank on left

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"]   = 12

fig, ax = plt.subplots(figsize=(10, 3.5))

# ── axis range ────────────────────────────────────────────────────────────────
n_models  = len(avg_ranks)
rank_min  = avg_ranks.min()
rank_max  = avg_ranks.max()
pad       = 0.6
ax_min    = max(1.0, rank_min - pad)
ax_max    = min(float(n_models), rank_max + pad)

ax.set_xlim(ax_min, ax_max)
ax.set_ylim(-0.5, 1.0)
ax.axis("off")

# ── main horizontal axis line ─────────────────────────────────────────────────
axis_y = 0.55
ax.plot([ax_min, ax_max], [axis_y, axis_y], color="black", lw=1.5,
        transform=ax.transData, clip_on=False)

# tick marks
for r in np.arange(np.ceil(ax_min), np.floor(ax_max) + 1):
    ax.plot([r, r], [axis_y - 0.02, axis_y + 0.02],
            color="black", lw=1.2, transform=ax.transData)
    ax.text(r, axis_y + 0.06, str(int(r)),
            ha="center", va="bottom", fontsize=10,
            transform=ax.transData)

ax.text((ax_min + ax_max) / 2, axis_y + 0.25,
        "Average Rank  (lower = better →)",
        ha="center", va="bottom", fontsize=11, style="italic")

# ── draw model ticks + labels ─────────────────────────────────────────────────
n = len(avg_ranks)
half = (n + 1) // 2          # top-half models (left of midpoint)

label_y_top    = axis_y + 0.42   # y for labels above
label_y_bottom = axis_y - 0.40   # y for labels below
stem_top       = axis_y + 0.20
stem_bottom    = axis_y - 0.18

top_models    = list(avg_ranks.index[:half])
bottom_models = list(avg_ranks.index[half:])

def label_color(name):
    return "#b22222" if name == OUR_MODEL else "black"   # firebrick red for ours

def label_weight(name):
    return "bold" if name == OUR_MODEL else "normal"

for name in top_models:
    x = avg_ranks[name]
    ax.plot([x, x], [axis_y, stem_top], color="black", lw=1.0)
    is_ours = (name == OUR_MODEL)
    txt = ax.text(x, label_y_top, name,
                  ha="center", va="bottom",
                  fontsize=10, color=label_color(name),
                  fontweight=label_weight(name),
                  rotation=30, rotation_mode="anchor")
    if is_ours:
        txt.set_path_effects([
            pe.withStroke(linewidth=2, foreground="white")
        ])

for name in bottom_models:
    x = avg_ranks[name]
    ax.plot([x, x], [stem_bottom, axis_y], color="black", lw=1.0)
    txt = ax.text(x, label_y_bottom, name,
                  ha="center", va="top",
                  fontsize=10, color=label_color(name),
                  fontweight=label_weight(name),
                  rotation=-30, rotation_mode="anchor")

# ── draw significance cliques (thick lines for p >= 0.05) ────────────────────
# Group models that are NOT significantly different (p >= 0.05) into cliques,
# then draw a horizontal bar spanning the min–max avg rank of each clique.
ALPHA     = 0.05
bar_y     = axis_y - 0.10
bar_step  = 0.055
drawn     = set()
bar_level = 0

# Collect all non-significant pairs
ns_pairs = []
for i, m1 in enumerate(models_used):
    for j, m2 in enumerate(models_used):
        if j <= i:
            continue
        if m1 not in p_values.index or m2 not in p_values.columns:
            continue
        if p_values.loc[m1, m2] >= ALPHA:
            ns_pairs.append((m1, m2))

# Build maximal cliques via greedy grouping
cliques = []
used_in_clique = set()
for m1, m2 in ns_pairs:
    placed = False
    for clique in cliques:
        # m1 and m2 must both be non-significantly different from all members
        if all(
            (a == m1 or a == m2 or
             (a in p_values.index and m1 in p_values.columns and p_values.loc[a, m1] >= ALPHA) and
             (a in p_values.index and m2 in p_values.columns and p_values.loc[a, m2] >= ALPHA))
            for a in clique
        ):
            clique.add(m1)
            clique.add(m2)
            placed = True
            break
    if not placed:
        cliques.append({m1, m2})

# Remove subsets
cliques = [c for i, c in enumerate(cliques)
           if not any(c < other for j, other in enumerate(cliques) if i != j)]

clique_color = "#3a86ff"
for clique in cliques:
    xs = sorted([avg_ranks[m] for m in clique])
    y  = bar_y - bar_level * bar_step
    ax.plot([xs[0], xs[-1]], [y, y],
            color=clique_color, lw=4.5, solid_capstyle="round",
            alpha=0.75, zorder=3)
    bar_level += 1

# ── legend note ───────────────────────────────────────────────────────────────
note = (f"Friedman: χ²={stat:.2f}, p={p_friedman:.4f}\n"
        f"Thick bars: groups with p ≥ 0.05 (Wilcoxon–Holm)\n"
        f"Red bold = DAF-MoE (Ours)   |   TabM excluded (pending)")
ax.text(0.01, 0.01, note,
        transform=ax.transAxes,
        fontsize=8.5, va="bottom", ha="left",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.8))

ax.set_title("Critical Difference Diagram (Wilcoxon–Holm, α=0.05)",
             fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()

save_pdf = os.path.join(OUTPUT_DIR, "cd_diagram_wilcoxon_DL.pdf")
save_png = os.path.join(OUTPUT_DIR, "cd_diagram_wilcoxon_DL.png")
plt.savefig(save_pdf, format="pdf", bbox_inches="tight")
plt.savefig(save_png, dpi=300, bbox_inches="tight")

print(f"\n✅ CD diagram saved:")
print(f"   PDF → {save_pdf}")
print(f"   PNG → {save_png}")
