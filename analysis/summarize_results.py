"""
Performance Summary Script
==========================

Description:
    Aggregates experimental results from JSON logs found in `results/scores`.
    It generates the main performance comparison tables (Table 2 and Table 4 in the paper)
    and visualizes the average rank of DAF-MoE against baselines.

    Key Features:
    - Filters results for the 9 target benchmark datasets.
    - Aggregates metrics (RMSE/ACC/AUPRC) across multiple random seeds.
    - Generates a 'Result (Mean ± Std)' summary CSV.
    - Plots a ranking chart comparing DAF-MoE with Deep Learning and GBDT baselines.

Usage:
    python analysis/summarize_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ==========================================
# 1. Configuration
# ==========================================
SCORE_DIR = "results/scores"
OUTPUT_DIR = "results/summarize_performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target datasets for the paper (9 Benchmarks)
TARGET_DATASETS = [
    "California Housing",
    "Adult Census Income",
    "Higgs small", 
    "Covertype",
    "Allstate",
    "BNP Paribas",
    "NHANES",
    "MIMIC-III Mortality",
    "MIMIC-IV Mortality"
]

MODEL_NAME_MAP = {
    "daf_moe": "DAF-MoE (Ours)",
    "ft_transformer": "FT-Transformer",
    "resnet": "ResNet",
    "mlp": "MLP",
    "tabm": "TabM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost"
}

# Color Scheme for Plots
COLOR_OURS = "#b2df8a"   # Light Green
COLOR_DL   = "#fdbf6f"   # Light Orange
COLOR_GBDT = "#a6cee3"   # Light Blue

COLOR_MAP = {
    "DAF-MoE (Ours)": COLOR_OURS,
    "FT-Transformer": COLOR_DL,
    "ResNet": COLOR_DL,
    "MLP": COLOR_DL,
    "XGBoost": COLOR_GBDT,
    "CatBoost": COLOR_GBDT
}

# ==========================================
# 2. Rank Calculation
# ==========================================
def calculate_ranks(df):
    """Calculates the rank of each model per dataset based on the mean score."""
    datasets = df['Dataset'].unique()
    ranks = defaultdict(list)

    for dataset in datasets:
        ds_data = df[df['Dataset'] == dataset].copy()
        
        # Determine sorting direction (RMSE: Lower is better, others: Higher is better)
        metric_name = ds_data.iloc[0]['Metric'].upper()
        ascending = True if 'RMSE' in metric_name else False
        
        ds_data = ds_data.sort_values(by="Mean Score", ascending=ascending)
        
        for rank, row in enumerate(ds_data.itertuples(), start=1):
            mapped_name = MODEL_NAME_MAP.get(row.Model, row.Model)
            ranks[mapped_name].append(rank)
    return ranks

# ==========================================
# 3. Visualization
# ==========================================
def plot_rank_chart(rank_data):
    """Generates and saves the average rank comparison chart."""
    if not rank_data: return
    print("\n🎨 Generating Rank Chart...")

    # Calculate statistics
    model_stats = []
    for model, rank_list in rank_data.items():
        if not rank_list: continue
        model_stats.append({
            "Model": model,
            "Mean Rank": np.mean(rank_list),
            "Std Rank": np.std(rank_list)
        })
    
    df_plot = pd.DataFrame(model_stats)
    df_plot = df_plot.sort_values("Mean Rank", ascending=True) 

    # Plot Settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = [COLOR_MAP.get(m, "#999999") for m in df_plot["Model"]]
    y_pos = np.arange(len(df_plot))
    
    bars = ax.barh(y_pos, df_plot["Mean Rank"], 
                   align='center', color=colors, alpha=0.9, 
                   edgecolor='black', linewidth=1.0, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Model"], fontsize=12, fontweight='bold')
    ax.set_xlabel('Rank (↓)', fontsize=13, fontweight='bold')
    
    ax.set_xlim(0, 6.5)
    ax.set_xticks(np.arange(1, 7))
    ax.tick_params(axis='x', labelsize=11)

    for bar, mean, std in zip(bars, df_plot["Mean Rank"], df_plot["Std Rank"]):
        width = bar.get_width()
        text_str = f"{mean:.1f} ± {std:.1f}"
        ax.text(width - 0.2, bar.get_y() + bar.get_height()/2, text_str, 
                ha='right', va='center', fontsize=11, fontweight='bold', color='black')

    ax.grid(False) 

    ax.set_title('Performance ranks', fontsize=15, fontweight='bold', pad=15)
    dataset_count = len(rank_data[next(iter(rank_data))])
    ax.text(0.5, 1.01, f'On {dataset_count} datasets',
            transform=ax.transAxes, ha='center', fontsize=11)

    legend_elements = [
        Patch(facecolor=COLOR_OURS, edgecolor='black', label='DAF-MoE (Ours)'),
        Patch(facecolor=COLOR_DL,   edgecolor='black', label='Deep Learning'),
        Patch(facecolor=COLOR_GBDT, edgecolor='black', label='GBDT')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
              fontsize=11, frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()

    save_pdf = os.path.join(OUTPUT_DIR, "rank_chart.pdf")
    save_png = os.path.join(OUTPUT_DIR, "rank_chart.png")
    
    plt.savefig(save_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(save_png, dpi=600, bbox_inches='tight')
    
    print(f"✅ Chart saved:\n  - PDF: {save_pdf}\n  - PNG: {save_png}")

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    files = glob.glob(os.path.join(SCORE_DIR, "*.json"))
    if not files:
        print(f"❌ No result files found in {SCORE_DIR}")
        return

    results = {}
    print(f"📂 Found {len(files)} result files. Filtering & Parsing...")
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset = data['dataset']
            model = data['model']
            metrics = data['metrics']
            config_metric = data.get('config', {}).get('optimize_metric', None)
            
            # Filtering: Skip datasets not in the target list
            if dataset not in TARGET_DATASETS:
                continue

            key = (dataset, model)
            if key not in results:
                results[key] = {'metrics': [], 'target_metric': config_metric}
            results[key]['metrics'].append(metrics)
            
        except Exception as e:
            print(f"⚠️ Error reading {fpath}: {e}")

    summary_data = []
    for (dataset, model), info in results.items():
        metrics_list = info['metrics']
        target_metric = info['target_metric']
        n_seeds = len(metrics_list)
        first_metric = metrics_list[0]
        
        # Determine Primary Metric
        if target_metric and target_metric in first_metric:
            scores = [m[target_metric] for m in metrics_list]
            main_metric_name = target_metric.upper()
        elif 'rmse' in first_metric:
            scores = [m['rmse'] for m in metrics_list]
            main_metric_name = 'RMSE'
        elif 'auprc' in first_metric and any(name in dataset.lower() for name in ['credit','mimic']):
            scores = [m['auprc'] for m in metrics_list]
            main_metric_name = 'AUPRC'
        else:
            scores = [m.get('acc', 0) for m in metrics_list]
            main_metric_name = 'ACC'
            
        summary_data.append({
            "Dataset": dataset,
            "Model": model,
            "Seeds": n_seeds,
            "Metric": main_metric_name,
            "Mean Score": np.mean(scores),
            "Std Dev": np.std(scores),
            "Result (Mean ± Std)": f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
        })

    df = pd.DataFrame(summary_data)
    
    if not df.empty:
        # Sort: Dataset Name -> Model Name
        df = df.sort_values(by=["Dataset", "Model"])

        print("\n" + "="*90)
        print(f"📊 Final Experiment Summary (Filtered: {len(df['Dataset'].unique())} Datasets)")
        print("="*90)
        
        # Output columns excluding inference time
        print(df[["Dataset", "Model", "Seeds", "Metric", "Result (Mean ± Std)"]].to_string(index=False))
        print("="*90)

        csv_path = os.path.join(OUTPUT_DIR, "final_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Summary saved to: {csv_path}")

        ranks = calculate_ranks(df)
        plot_rank_chart(ranks)

    else:
        print("⚠️ No valid data found for the target datasets.")

if __name__ == "__main__":
    main()