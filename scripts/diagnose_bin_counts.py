import sys
import os
import pandas as pd
from collections import defaultdict
import nltk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.utils import load_results, discover_datasets

def count_words(text):
    if not isinstance(text, str): return 0
    return len(text.strip().split())

def get_bin_counts():
    models = ["flamingo_hf", "qwen_omni"]
    experiments = ["adding_mistakes", "early_answering", "random_partial_filler_text", "paraphrasing"]
    results_dir = "/scratch/aynevol/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/results"
    
    summary = []

    for model in models:
        for exp in experiments:
            try:
                datasets = discover_datasets(model, results_dir, restricted=True)
            except FileNotFoundError:
                continue
                
            for dataset in datasets:
                kwargs = {"restricted": True}
                if exp == "adding_mistakes" or exp == "paraphrasing":
                    kwargs["perturbation_source"] = "mistral"
                elif exp == "random_partial_filler_text":
                    kwargs["filler_type"] = "lorem"
                    
                try:
                    df = load_results(model, results_dir, exp, dataset, **kwargs)
                except FileNotFoundError:
                    continue
                    
                if df.empty:
                    continue
                    
                if 'total_sentences_in_chain' in df.columns:
                    df = df[df['total_sentences_in_chain'] > 0].copy()
                
                # Replicate binning logic for each experiment exactly as the plotting scripts do
                if exp == "adding_mistakes":
                    # We need the truth condition to filter samples correctly
                    try:
                        rpf_df = load_results(model, results_dir, 'random_partial_filler_text', dataset, filler_type='lorem', restricted=True)
                        rpf_0 = rpf_df[rpf_df['percent_replaced'] == 0]
                        truth = dict(zip(zip(rpf_0['id'], rpf_0['chain_id']), rpf_0['predicted_choice']))
                        df = df[df.apply(lambda r: (r['id'], r['chain_id']) in truth, axis=1)].copy()
                    except FileNotFoundError:
                        continue
                        
                    df['percent'] = ((df['mistake_position'] - 1) / df['total_sentences_in_chain']) * 100
                    df['bin'] = (df['percent'] / 5).round() * 5
                    
                elif exp == "early_answering":
                    # We need the 100% full CoT truth condition
                    max_per_chain = df.groupby(['id', 'chain_id'])['num_sentences_provided'].max().reset_index()
                    max_per_chain.rename(columns={'num_sentences_provided': 'max_sentences'}, inplace=True)
                    df = df.merge(max_per_chain, on=['id', 'chain_id'])
                    full_cot = df[df['num_sentences_provided'] == df['max_sentences']]
                    truth = dict(zip(zip(full_cot['id'], full_cot['chain_id']), full_cot['predicted_choice']))
                    df = df[df.apply(lambda r: (r['id'], r['chain_id']) in truth, axis=1)].copy()
                    
                    df['percent'] = (df['num_sentences_provided'] / df['total_sentences_in_chain']) * 100
                    df['bin'] = (df['percent'] / 5).round() * 5
                    
                elif exp == "random_partial_filler_text":
                    zero_pct = df[df['percent_replaced'] == 0]
                    truth = dict(zip(zip(zero_pct['id'], zero_pct['chain_id']), zero_pct['predicted_choice']))
                    df = df[df.apply(lambda r: (r['id'], r['chain_id']) in truth, axis=1)].copy()
                    
                    df['bin'] = (df['percent_replaced'] / 5).round() * 5
                    
                elif exp == "paraphrasing":
                    try:
                        baseline_df = load_results(model, results_dir, "baseline", dataset, restricted=True)
                    except FileNotFoundError:
                        continue

                    word_counts_lookup = {}
                    for _, row in baseline_df.iterrows():
                        cot = row.get("sanitized_cot", "")
                        sentences = nltk.sent_tokenize(cot) if isinstance(cot, str) else []
                        word_counts_lists = [count_words(s) for s in sentences]
                        word_counts_lookup[(row["id"], row["chain_id"])] = {
                            "counts": word_counts_lists,
                            "total_words": sum(word_counts_lists),
                        }

                    zero_pct_df = df[df["num_sentences_paraphrased"] == 0]
                    truth = dict(zip(zip(zero_pct_df["id"], zero_pct_df["chain_id"]), zero_pct_df["predicted_choice"]))
                    df = df[df.apply(lambda r: (r['id'], r['chain_id']) in truth, axis=1)].copy()

                    def calculate_percent_words(row):
                        info = word_counts_lookup.get((row["id"], row["chain_id"]))
                        if not info or info["total_words"] == 0:
                            return 0.0
                        n_para = int(row["num_sentences_paraphrased"])
                        words_para = sum(info["counts"][:n_para])
                        return (words_para / info["total_words"]) * 100.0

                    df["percent_words_paraphrased"] = df.apply(calculate_percent_words, axis=1)
                    df['bin'] = (df["percent_words_paraphrased"] / 5).round() * 5
                
                # Compute sample counts per bin
                counts = df.groupby('bin').size()
                
                for b, count in counts.items():
                    summary.append({
                        "Model": model,
                        "Experiment": exp,
                        "Dataset": dataset,
                        "Bin (%)": b,
                        "Sample Count": count
                    })
                    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('/scratch/aynevol/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/results/diagnostics_bin_counts.csv', index=False)
    
    print("\n--- Bin Count Diagnostics ---")
    print("\nOverall Statistics across all plotted bins (Min / Median / Mean / Max):")
    print(summary_df['Sample Count'].describe()[['min', '50%', 'mean', 'max']])
    
    print("\nDistribution of Bin Sizes (e.g. how many bins have exactly N samples):")
    print("Size Range\tNumber of Bins")
    bins = [0, 5, 10, 20, 50, 100, 500, 1000, 5000]
    cuts = pd.cut(summary_df['Sample Count'], bins=bins, right=False)
    print(cuts.value_counts().sort_index())
    
    # Save the full report for the user to review
    with open('/scratch/aynevol/Measuring-Faithfulness-of-CoT-Reasoning-LALMs/results/bin_sample_counts_report.txt', 'w') as f:
        for (model, exp), group in summary_df.groupby(['Model', 'Experiment']):
            f.write(f"\n{'='*80}\n{model.upper()} - {exp.upper()}\n{'='*80}\n")
            pivot = group.pivot(index='Dataset', columns='Bin (%)', values='Sample Count')
            f.write(pivot.fillna(0).astype(int).to_string())
            f.write("\n")
            
    print("\nFull readable report of exact sample counts per point generated at:")
    print(" -> results/bin_sample_counts_report.txt")
    print(" -> results/diagnostics_bin_counts.csv")

if __name__ == "__main__":
    get_bin_counts()
