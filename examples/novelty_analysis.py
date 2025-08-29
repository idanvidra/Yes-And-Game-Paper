#!/usr/bin/env python3
"""
Novelty Analysis for Yes-And Story Datasets

This script computes semantic novelty as cosine distance between consecutive
sentences within each story and attributes each distance to the responding
speaker (the second sentence in the pair). It supports HH (human-human) and
AIH (AI-human) stories, evaluates multiple embedding models, and produces
summary tables, significance tests, and a bar plot.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from scipy import stats as scipy_stats
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style to match paper
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("Dependencies loaded successfully.")
print(f"Results will be saved to: {results_dir.absolute()}")

# Data Loading Functions
def load_stories_from_json(file_path):
    """Load stories from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_sentences_from_story(story):
    """Extract sentences from a story dictionary."""
    sentences = []
    for message in story['messages']:
        content = message['content'].strip()
        if content and content != "________________":
            sentences.append(content)
    return sentences

def get_speaker_labels_for_story(story):
    """Generate speaker labels for a specific story."""
    labels = []
    for message in story['messages']:
        content = message['content'].strip()
        if content and content != "________________":
            role = message['role']
            if role in ['human', 'human1', 'human2']:
                label = 'H_AIH' if role == 'human' else 'H_HH'
                labels.append(label)
            elif role == 'ai':
                labels.append('AI_AIH')
    return labels

# Load the datasets (paths resolved relative to project root)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
hh_file = project_root / "Story Datasets" / "hh_stories.json"
aih_file = project_root / "Story Datasets" / "aih_stories.json"

hh_stories = load_stories_from_json(str(hh_file))
aih_stories = load_stories_from_json(str(aih_file))

print(f"Loaded {len(hh_stories)} Human-Human stories")
print(f"Loaded {len(aih_stories)} AI-Human stories")

# Novelty Computation Functions
def compute_consecutive_distances(sentences, model):
    """Compute cosine distances between consecutive sentences."""
    if len(sentences) < 2:
        return np.array([])

    embeddings = model.encode(sentences)
    distances = cosine_distances(embeddings[:-1], embeddings[1:]).diagonal()
    return distances

def analyze_novelty(hh_stories, aih_stories, model_name, sentence_filter=2):
    """
    Compute novelty using distances between truly consecutive contributions
    within each story. Distances are attributed to the responding speaker
    (second in the pair).
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Initialize results for each speaker type
    h_hh_distances = []    # Human responses in HH stories
    h_aih_distances = []   # Human responses in AIH stories  
    ai_aih_distances = []  # AI responses in AIH stories
    
    print(f"Processing {len(hh_stories)} HH stories...")
    # Process HH stories
    for story in hh_stories:
        sentences = extract_sentences_from_story(story)
        if len(sentences) > sentence_filter:
            # Compute consecutive distances within this story
            story_distances = compute_consecutive_distances(sentences, model)
            # In HH, all responses are human (except the first sentence has no predecessor)
            h_hh_distances.extend(story_distances)
    
    print(f"Processing {len(aih_stories)} AIH stories...")
    # Process AIH stories
    for story in aih_stories:
        sentences = extract_sentences_from_story(story)
        if len(sentences) > sentence_filter:
            speaker_labels = get_speaker_labels_for_story(story)
            
            # Compute consecutive distances within this story
            story_distances = compute_consecutive_distances(sentences, model)
            
            # Assign each distance to the RESPONDING speaker (second in pair)
            for i, distance in enumerate(story_distances):
                responding_speaker = speaker_labels[i + 1]  # The speaker who responded
                
                if responding_speaker == 'H_AIH':
                    h_aih_distances.append(distance)
                elif responding_speaker == 'AI_AIH':
                    ai_aih_distances.append(distance)
    
    results = {
        'H_HH': np.array(h_hh_distances),
        'H_AIH': np.array(h_aih_distances),
        'AI_AIH': np.array(ai_aih_distances)
    }
    
    print("Analysis complete.")
    print(f"Sample sizes: H_HH={len(h_hh_distances)}, H_AIH={len(h_aih_distances)}, AI_AIH={len(ai_aih_distances)}")
    
    return results, model_name

def get_novelty_statistics(results):
    """Compute comprehensive statistics for novelty results."""
    stats = {}
    
    for group, values in results.items():
        if len(values) > 0:
            stats[group] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
        else:
            stats[group] = {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q25': np.nan,
                'q75': np.nan
            }
    
    return stats

# Multi-Model Analysis (Replicating Paper)
print("\nRunning multi-model novelty analysis")
print("=" * 60)

# Embedding models used in the analysis
models_to_test = [
    'sentence-transformers/all-roberta-large-v1',  # Similar to roberta-base-nli-stsb-mean-tokens
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
    'sentence-transformers/paraphrase-mpnet-base-v2', 
    'sentence-transformers/all-MiniLM-L6-v2'
]

# Fallback models if the above don't work
fallback_models = [
    'all-roberta-large-v1',
    'distilbert-base-nli-stsb-mean-tokens', 
    'paraphrase-mpnet-base-v2',
    'all-MiniLM-L6-v2'
]

model_display_names = {
    'sentence-transformers/all-roberta-large-v1': 'RoBERTa',
    'all-roberta-large-v1': 'RoBERTa',
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens': 'DistilBERT',
    'distilbert-base-nli-stsb-mean-tokens': 'DistilBERT',
    'sentence-transformers/paraphrase-mpnet-base-v2': 'MPNet',
    'paraphrase-mpnet-base-v2': 'MPNet',
    'sentence-transformers/all-MiniLM-L6-v2': 'MiniLM',
    'all-MiniLM-L6-v2': 'MiniLM'
}

all_results = {}
all_stats = {}

for i, model_name in enumerate(models_to_test):
    print(f"\nModel {i+1}/{len(models_to_test)}: {model_display_names.get(model_name, model_name)}")
    
    try:
        # Try primary model name
        results, used_model = analyze_novelty(
            hh_stories, aih_stories, 
            model_name=model_name,
            sentence_filter=2
        )
        
    except Exception as e:
        print(f"Failed with {model_name}: {e}")
        print(f"Trying fallback: {fallback_models[i]}")
        
        try:
            results, used_model = analyze_novelty(
                hh_stories, aih_stories,
                model_name=fallback_models[i],
                sentence_filter=2
            )
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            continue
    
    # Compute statistics
    stats_by_group = get_novelty_statistics(results)
    
    # Store results
    display_name = model_display_names.get(model_name, model_display_names.get(used_model, used_model))
    all_results[display_name] = results
    all_stats[display_name] = stats_by_group
    
    # Print results for this model
    print(f"\nResults for {display_name}:")
    for group, group_stats in stats_by_group.items():
        if group_stats['count'] > 0:
            print(f"  {group}: μ={group_stats['mean']:.4f} (σ={group_stats['std']:.4f}, n={group_stats['count']})")

print(f"\nCompleted analysis for {len(all_results)} models.")

# Create results table
print("\nResults table")
print("Mean consecutive-sentence distances by group and embedding model (with SD).")
print("=" * 80)

# Create DataFrame for easy manipulation
table_data = []
for model_name, stats_dict in all_stats.items():
    row = {
        'Model': model_name,
        'H_HH': f"{stats_dict['H_HH']['mean']:.3f} ({stats_dict['H_HH']['std']:.3f})" if not np.isnan(stats_dict['H_HH']['mean']) else "N/A",
        'H_AIH': f"{stats_dict['H_AIH']['mean']:.3f} ({stats_dict['H_AIH']['std']:.3f})" if not np.isnan(stats_dict['H_AIH']['mean']) else "N/A",
        'AI_AIH': f"{stats_dict['AI_AIH']['mean']:.3f} ({stats_dict['AI_AIH']['std']:.3f})" if not np.isnan(stats_dict['AI_AIH']['mean']) else "N/A"
    }
    table_data.append(row)

# Create and display table
results_df = pd.DataFrame(table_data)
print(results_df.to_string(index=False))

# Save table to CSV
results_df.to_csv(results_dir / "novelty_table.csv", index=False)
print(f"\n💾 Table saved to: {results_dir / 'novelty_table.csv'}")

# Also create a LaTeX version for easy paper integration
latex_table = results_df.to_latex(index=False, escape=False)
with open(results_dir / "novelty_table.tex", 'w') as f:
    f.write(latex_table)
print(f"📄 LaTeX table saved to: {results_dir / 'novelty_table.tex'}")

# Statistical analysis and significance testing
print("\nStatistical significance testing")
print("=" * 60)

statistical_results = []

for model_name, results in all_results.items():
    print(f"\n🔍 Model: {model_name}")
    print("-" * 40)
    
    # Get data for groups that have data
    groups_with_data = [(name, data) for name, data in results.items() if len(data) > 0]
    
    # Pairwise comparisons
    for i in range(len(groups_with_data)):
        for j in range(i+1, len(groups_with_data)):
            group1_name, group1_data = groups_with_data[i]
            group2_name, group2_data = groups_with_data[j]
            
            # Perform t-test
            t_stat, p_value = scipy_stats.ttest_ind(group1_data, group2_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data) - 1) * np.var(group1_data, ddof=1) + 
                                 (len(group2_data) - 1) * np.var(group2_data, ddof=1)) / 
                                (len(group1_data) + len(group2_data) - 2))
            cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
            
            # Determine significance level
            if p_value < 0.001:
                sig_level = "***"
                sig_text = "p < 0.001"
            elif p_value < 0.01:
                sig_level = "**"
                sig_text = "p < 0.01"
            elif p_value < 0.05:
                sig_level = "*"
                sig_text = "p < 0.05"
            else:
                sig_level = "ns"
                sig_text = "p ≥ 0.05"
            
            print(f"{group1_name} vs {group2_name}:")
            print(f"  Mean diff: {np.mean(group1_data) - np.mean(group2_data):+.4f}")
            print(f"  t({len(group1_data) + len(group2_data) - 2}) = {t_stat:.3f}, {sig_text} {sig_level}")
            print(f"  Cohen's d = {cohens_d:.3f}")
            
            # Store for summary
            statistical_results.append({
                'Model': model_name,
                'Comparison': f"{group1_name} vs {group2_name}",
                'Mean_Diff': np.mean(group1_data) - np.mean(group2_data),
                'T_Stat': t_stat,
                'P_Value': p_value,
                'Cohens_D': cohens_d,
                'Significance': sig_level
            })

# Save statistical results
stats_df = pd.DataFrame(statistical_results)
stats_df.to_csv(results_dir / "statistical_tests.csv", index=False)
print(f"\n💾 Statistical results saved to: {results_dir / 'statistical_tests.csv'}")

# Create novelty bar plot
print("\nCreating novelty bar plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Prepare data for plotting
models = list(all_stats.keys())
groups = ['H_HH', 'H_AIH', 'AI_AIH']
group_labels = ['Human-Human', 'Human-AI', 'AI-Human']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

x = np.arange(len(models))
width = 0.25

# Create bars for each group
for i, (group, label, color) in enumerate(zip(groups, group_labels, colors)):
    means = [all_stats[model][group]['mean'] for model in models]
    stds = [all_stats[model][group]['std'] for model in models]
    
    bars = ax.bar(x + i * width, means, width, label=label, color=color, alpha=0.8, 
                  yerr=stds, capsize=5, error_kw={'linewidth': 1.5})
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Customize plot
ax.set_xlabel('Embedding Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Novelty Score (Cosine Distance)', fontsize=14, fontweight='bold')
ax.set_title('Novelty comparison across embedding models\n' +
            'Cosine distances between consecutive sentences', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(models)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max([max([all_stats[model][group]['mean'] + all_stats[model][group]['std'] 
                        for group in groups if not np.isnan(all_stats[model][group]['mean'])]) 
                   for model in models]) * 1.1)

# Add brief methodological note
ax.text(0.02, 0.98, 'Method: within-story consecutive pairs; distances attributed to respondent', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

# Save the plot
plt.savefig(results_dir / "novelty_barplot.png", dpi=300, bbox_inches='tight')
plt.savefig(results_dir / "novelty_barplot.pdf", bbox_inches='tight')
plt.show()

print("Novelty bar plot saved to:")
print(f"  • {results_dir / 'novelty_barplot.png'}")
print(f"  • {results_dir / 'novelty_barplot.pdf'}")

# Generate a concise summary report
print("\nGenerating summary report...")

report_lines = []
report_lines.append("# Novelty Analysis Report")
report_lines.append("=" * 50)
report_lines.append("")
report_lines.append("## Method overview")
report_lines.append("")
report_lines.append("1. Distances computed between consecutive sentences within each story")
report_lines.append("2. Distances attributed to the responding speaker (second in the pair)")
report_lines.append("3. Within-story only (no cross-story pairs)")
report_lines.append("4. Summary statistics reported per group (H_HH, H_AIH, AI_AIH)")
report_lines.append("")
report_lines.append("## Results:")
report_lines.append("")

# Add results table
report_lines.append("### Mean Novelty Scores by Model and Group:")
report_lines.append("")
for model_name, stats_dict in all_stats.items():
    report_lines.append(f"**{model_name}:**")
    for group in ['H_HH', 'H_AIH', 'AI_AIH']:
        if not np.isnan(stats_dict[group]['mean']):
            report_lines.append(f"  - {group}: {stats_dict[group]['mean']:.4f} (±{stats_dict[group]['std']:.4f}, n={stats_dict[group]['count']})")
    report_lines.append("")

# Add key findings
report_lines.append("## KEY FINDINGS:")
report_lines.append("")

# Calculate average rankings across models
avg_scores = {'H_HH': [], 'H_AIH': [], 'AI_AIH': []}
for model_name, stats_dict in all_stats.items():
    for group in avg_scores.keys():
        if not np.isnan(stats_dict[group]['mean']):
            avg_scores[group].append(stats_dict[group]['mean'])

avg_means = {group: np.mean(scores) for group, scores in avg_scores.items() if scores}
ranking = sorted(avg_means.items(), key=lambda x: x[1])

report_lines.append(f"1. **Overall Ranking (low to high novelty)**: {' < '.join([f'{group} ({mean:.3f})' for group, mean in ranking])}")
report_lines.append("")

if 'AI_AIH' in avg_means and 'H_HH' in avg_means:
    ai_vs_hh = avg_means['AI_AIH'] - avg_means['H_HH']
    report_lines.append(f"2. **AI vs Human-Human**: AI shows {'higher' if ai_vs_hh > 0 else 'lower'} novelty by {abs(ai_vs_hh):.4f} points")

if 'AI_AIH' in avg_means and 'H_AIH' in avg_means:
    ai_vs_haih = avg_means['AI_AIH'] - avg_means['H_AIH']
    report_lines.append(f"3. **AI vs Human-in-AIH**: AI shows {'higher' if ai_vs_haih > 0 else 'lower'} novelty by {abs(ai_vs_haih):.4f} points")

report_lines.append("")
report_lines.append("## STATISTICAL SIGNIFICANCE:")
report_lines.append("")

# Add significance summary
sig_summary = {}
for result in statistical_results:
    comparison = result['Comparison']
    if comparison not in sig_summary:
        sig_summary[comparison] = []
    sig_summary[comparison].append(result['Significance'])

for comparison, significances in sig_summary.items():
    sig_count = sum(1 for s in significances if s != 'ns')
    total_count = len(significances)
    report_lines.append(f"- **{comparison}**: Significant in {sig_count}/{total_count} models")

report_lines.append("")
report_lines.append("## FILES GENERATED:")
report_lines.append("")
report_lines.append("- novelty_table.csv (results table)")
report_lines.append("- novelty_table.tex (LaTeX table)")
report_lines.append("- novelty_barplot.png/.pdf (main figure)")
report_lines.append("- statistical_tests.csv (all statistical tests)")
report_lines.append("- analysis_report.txt (this report)")
report_lines.append("")

# Save report
report_text = "\n".join(report_lines)
with open(results_dir / "analysis_report.txt", 'w') as f:
    f.write(report_text)

print("\nAnalysis complete.")
print("=" * 50)
print(report_text)
print("=" * 50)
print(f"\nComplete report saved to: {results_dir / 'analysis_report.txt'}")
print(f"All results available in: {results_dir.absolute()}")
