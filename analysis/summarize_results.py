"""
Experiment Results Summarizer
=============================

Description:
    This script aggregates the evaluation logs (JSON files) generated during training.
    It groups results by (Dataset, Model), calculates the Mean and Standard Deviation 
    of the primary metric across all random seeds, and reports the average inference time.

    The final summary is saved as a CSV file for easy inclusion in the paper.

Usage:
    python analysis/summarize_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import glob
from collections import defaultdict

# ==========================================
# ⚙️ Configuration
# ==========================================
SCORE_DIR = "results/scores"
OUTPUT_DIR = "results/analysis"
OUTPUT_FILE = "final_summary.csv"

# Define primary metrics for each dataset to ensure consistency
METRIC_RULES = {
    "Adult Census Income": "acc",
    "Higgs small": "acc",
    "Covertype": "acc",
    "BNP Paribas": "acc",
    "NHANES": "acc",
    "Credit Card Fraud": "auprc",
    "MIMIC-III Mortality": "auprc",
    "MIMIC-IV Mortality": "auprc",
    "California Housing": "rmse",
    "YearPrediction": "rmse",
    "Allstate": "rmse"
}

def get_metric_value(metrics_data, target_metric):
    """Safely extracts value from metrics dict/list."""
    if isinstance(metrics_data, list) and len(metrics_data) > 0:
        return metrics_data[0].get(target_metric)
    elif isinstance(metrics_data, dict):
        return metrics_data.get(target_metric)
    return None

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(SCORE_DIR, "*.json"))
    
    if not files:
        print(f"❌ No result files found in {SCORE_DIR}")
        return

    # Data Collection: { (dataset, model): {'scores': [], 'times': [], 'metric_name': str} }
    results = defaultdict(lambda: {'scores': [], 'times': [], 'metric_name': 'ACC'})
    
    print(f"📂 Found {len(files)} result files. Parsing...")
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            dataset = data.get('dataset', 'Unknown')
            model = data.get('model', 'Unknown')
            metrics = data.get('metrics', {})
            
            # 1. Determine Target Metric
            # Priority: 1. Config 'optimize_metric' -> 2. METRIC_RULES dict -> 3. Fallback to ACC
            config_metric = data.get('config', {}).get('optimize_metric')
            
            if config_metric:
                target_metric = config_metric
            else:
                # Match dataset name prefix (e.g., "Adult Census Income_..." -> "Adult Census Income")
                dataset_key = dataset.split('_')[0] if '_' in dataset else dataset
                # Try exact match first, then partial match if needed
                target_metric = METRIC_RULES.get(dataset, METRIC_RULES.get(dataset_key, 'acc'))

            # 2. Extract Score
            score = get_metric_value(metrics, target_metric)
            
            # 3. Extract Inference Time
            inf_time = metrics.get('inference_time_sec', 0) if isinstance(metrics, dict) else 0
            
            if score is not None:
                key = (dataset, model)
                results[key]['scores'].append(score)
                results[key]['times'].append(inf_time)
                results[key]['metric_name'] = target_metric.upper()
            
        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(fpath)}: {e}")

    # --- Generate Summary Table ---
    summary_data = []
    
    for (dataset, model), info in results.items():
        scores = info['scores']
        times = info['times']
        metric_name = info['metric_name']
        n_seeds = len(scores)
        
        if n_seeds == 0: continue
        
        # Statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        mean_time = np.mean(times)
        
        summary_data.append({
            "Dataset": dataset,
            "Model": model,
            "Seeds": n_seeds,
            "Metric": metric_name,
            "Mean Score": mean_score,
            "Std Dev": std_score,
            "Result (Mean ± Std)": f"{mean_score:.4f} ± {std_score:.4f}",
            "Avg Inference Time (s)": f"{mean_time:.4f}"
        })

    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    if not df.empty:
        # Sort for readability: Dataset -> Model
        df = df.sort_values(by=["Dataset", "Model"])
        
        print("\n" + "="*90)
        print("📊 Final Experiment Summary (All Seeds)")
        print("="*90)
        print(df[["Dataset", "Model", "Seeds", "Metric", "Result (Mean ± Std)", "Avg Inference Time (s)"]].to_string(index=False))
        print("="*90)
        
        save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        df.to_csv(save_path, index=False)
        print(f"\n💾 Summary saved to: {save_path}")
    else:
        print("⚠️ No valid data to summarize.")

if __name__ == "__main__":
    main()