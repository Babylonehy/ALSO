import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_all_metrics(summary_data, key_prefix="train"):
    section = summary_data.get(key_prefix)
    if not section and "results" in summary_data:
        section = summary_data
        
    if not section or "results" not in section:
        print(f"Warning: No results found in {key_prefix}")
        return {}

    # metrics storage: {metric_name: [values]}
    metrics = {
        "Goal": [],
        "Believability": [],
        "Relationship": [],
        "Knowledge": [],
        "Secret": [],
        "Social Rules": [],
        "Financial": [],
        "Overall": []
    }
    
    # Mapping from JSON keys to display names
    key_map = {
        "goal": "Goal",
        "believability": "Believability",
        "relationship": "Relationship",
        "knowledge": "Knowledge",
        "secret": "Secret",
        "social_rules": "Social Rules",
        "financial_and_material_benefits": "Financial",
        "overall_score": "Overall"
    }

    for res in section["results"]:
        if "error" in res:
            continue
            
        r = res.get("result", {})
        final_rewards = r.get("final_rewards", {})
        
        for agent in ["p1", "p2"]:
            p_data = final_rewards.get(agent, {})
            breakdown = p_data.get("breakdown", {})
            
            # Extract each metric
            for json_key, display_name in key_map.items():
                val = breakdown.get(json_key, 0)
                # Fallback for overall if not in breakdown
                if json_key == "overall_score" and "overall" in p_data:
                    val = p_data["overall"]
                
                metrics[display_name].append(val)
        
    return metrics

def main():
    # Paths to results
    baseline_path = "results/generalization/adversarial_baseline_20260127_184542_fast_ctx/final_summary.json"
    test_path = "results/generalization/adversarial_splitA_20260127_193553_test_only_fast/final_summary.json"
    
    print(f"Loading Baseline: {baseline_path}")
    baseline_data = load_summary(baseline_path)
    
    print(f"Loading Test: {test_path}")
    test_data = load_summary(test_path)
    
    baseline_metrics = extract_all_metrics(baseline_data, "baseline")
    test_metrics = extract_all_metrics(test_data, "test")
    
    # Metric order for plotting
    metric_names = ["Goal", "Overall", "Believability", "Relationship", "Knowledge", "Secret", "Social Rules", "Financial"]
    
    baseline_means = []
    baseline_stds = []
    test_means = []
    test_stds = []
    
    print("\n--- Summary Statistics ---")
    print(f"{'Metric':<15} | {'Baseline':<15} | {'Ours (Zero-Shot)':<15}")
    print("-" * 50)
    
    for m in metric_names:
        b_vals = baseline_metrics.get(m, [])
        t_vals = test_metrics.get(m, [])
        
        b_mean = np.mean(b_vals) if b_vals else 0
        b_std = np.std(b_vals) if b_vals else 0
        t_mean = np.mean(t_vals) if t_vals else 0
        t_std = np.std(t_vals) if t_vals else 0
        
        baseline_means.append(b_mean)
        baseline_stds.append(b_std)
        test_means.append(t_mean)
        test_stds.append(t_std)
        
        print(f"{m:<15} | {b_mean:.2f} (+/- {b_std:.2f}) | {t_mean:.2f} (+/- {t_std:.2f})")

    # Plotting
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline (Online)', capsize=5, color='#d62728', alpha=0.9)
    rects2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, label='Ours (Zero-Shot)', capsize=5, color='#1f77b4', alpha=0.9)
    
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_title('Cross-Scenario Generalization: Full Metrics Breakdown', fontsize=16)
    ax.set_xticks(x, metric_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 10.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Only label if significant height to avoid clutter
            if height > 0.5:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    output_file = "generalization_detailed_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved detailed plot to {output_file}")

if __name__ == "__main__":
    main()
