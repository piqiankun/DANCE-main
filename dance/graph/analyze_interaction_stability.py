from dance.paths import RESULTS_DIR as PROJECT_RESULTS_DIR

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


RESULTS_DIR = PROJECT_RESULTS_DIR
MODEL_NAME = "3.5-27B"


ENHANCE_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5]

def load_results():
    results = {}
    for val in ENHANCE_VALUES:
        filepath = f"{RESULTS_DIR}pairwise_neuron_synergy_{MODEL_NAME}_enhance_{val}.csv"
        try:
            df = pd.read_csv(filepath)
            results[val] = df
            print(f"Loaded enhance={val}: {len(df)} neuron pairs")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
    return results


def compute_correlation_matrix(results):
    print("\n" + "="*60)
    print("1. Synergy correlation matrix (Pearson)")
    print("="*60)

    synergy_series = {}
    for val, df in results.items():
        synergy_series[val] = df['synergy']


    vals = sorted(synergy_series.keys())
    corr_matrix = pd.DataFrame(index=vals, columns=vals, dtype=float)

    for v1 in vals:
        for v2 in vals:
            corr, p_value = stats.pearsonr(synergy_series[v1], synergy_series[v2])
            corr_matrix.loc[v1, v2] = corr

    print(corr_matrix.to_string(float_format="{:.4f}".format))


    print("\n" + "="*60)
    print("2. Synergy rank correlations (Spearman)")
    print("="*60)

    spearman_matrix = pd.DataFrame(index=vals, columns=vals, dtype=float)
    for v1 in vals:
        for v2 in vals:
            corr, p_value = stats.spearmanr(synergy_series[v1], synergy_series[v2])
            spearman_matrix.loc[v1, v2] = corr

    print(spearman_matrix.to_string(float_format="{:.4f}".format))

    return corr_matrix, spearman_matrix


def compute_classification_consistency(results):
    print("\n" + "="*60)
    print("3. Classification consistency (synergy/suppression)")
    print("="*60)

    vals = sorted(results.keys())
    n_pairs = len(results[vals[0]])

    print(f"Total neuron pairs: {n_pairs}")


    merged = results[vals[0]][['layer_A', 'neuron_A', 'layer_B', 'neuron_B', 'relationship']].copy()
    merged = merged.rename(columns={'relationship': f'rel_{vals[0]}'})

    for val in vals[1:]:
        other = results[val][['layer_A', 'neuron_A', 'layer_B', 'neuron_B', 'relationship']].copy()
        other = other.rename(columns={'relationship': f'rel_{val}'})
        merged = merged.merge(other, on=['layer_A', 'neuron_A', 'layer_B', 'neuron_B'])


    consistency_matrix = pd.DataFrame(index=vals, columns=vals, dtype=float)

    for v1 in vals:
        for v2 in vals:
            col1, col2 = f'rel_{v1}', f'rel_{v2}'
            consistent = (merged[col1] == merged[col2]).sum()
            consistency_matrix.loc[v1, v2] = consistent / n_pairs * 100

    print("\nClassification agreement (%):")
    print(consistency_matrix.to_string(float_format="{:.1f}".format))


    print("\nClass distribution at each enhancement strength:")
    for val in vals:
        col = f'rel_{val}'
        synergistic = (merged[col] == 'synergistic').sum()
        suppressive = (merged[col] == 'suppressive').sum()
        independent = (merged[col] == 'independent').sum()
        print(f"  enhance={val}: Synergistic={synergistic} ({synergistic/n_pairs*100:.1f}%), "
              f"Suppressive={suppressive} ({suppressive/n_pairs*100:.1f}%), "
              f"Independent={independent} ({independent/n_pairs*100:.1f}%)")

    return merged, consistency_matrix


def find_stable_pairs_all(merged, vals, results):
    print("\n" + "="*60)
    print("4. Stable neuron pairs (consistent classes across all enhancement strengths)")
    print("="*60)

    rel_cols = [f'rel_{v}' for v in vals]


    merged['all_consistent'] = merged[rel_cols].apply(lambda x: x.nunique() == 1, axis=1)

    stable_pairs = merged[merged['all_consistent']].copy()
    unstable_pairs = merged[~merged['all_consistent']].copy()

    print(f"Stable neuron pairs: {len(stable_pairs)} ({len(stable_pairs)/len(merged)*100:.1f}%)")
    print(f"Unstable neuron pairs: {len(unstable_pairs)} ({len(unstable_pairs)/len(merged)*100:.1f}%)")

    if len(stable_pairs) > 0:

        stable_rel = stable_pairs[f'rel_{vals[0]}'].value_counts()
        print(f"\nClass distribution of stable neuron pairs:")
        for rel, count in stable_rel.items():
            print(f"  {rel}: {count} ({count/len(stable_pairs)*100:.1f}%)")


        synergy_cols = []
        for v in vals:
            syn_col = f'synergy_{v}'
            synergy_cols.append(syn_col)

            syn_df = results[v][['layer_A', 'neuron_A', 'layer_B', 'neuron_B', 'synergy']].copy()
            syn_df = syn_df.rename(columns={'synergy': syn_col})
            stable_pairs = stable_pairs.merge(syn_df, on=['layer_A', 'neuron_A', 'layer_B', 'neuron_B'], how='left')


        stable_pairs['synergy_mean'] = stable_pairs[synergy_cols].mean(axis=1)
        stable_pairs['synergy_std'] = stable_pairs[synergy_cols].std(axis=1)


        stable_synergistic = stable_pairs[stable_pairs[f'rel_{vals[0]}'] == 'synergistic'].sort_values('synergy_mean', ascending=False)

        stable_suppressive = stable_pairs[stable_pairs[f'rel_{vals[0]}'] == 'suppressive'].sort_values('synergy_mean', ascending=True)

        print(f"\n" + "="*60)
        print("Stable synergistic neuron pairs (Top 20)")
        print("="*60)
        if len(stable_synergistic) > 0:
            top_stable_syn = stable_synergistic.head(20)
            print(f"{'Layer_A':>10} {'Neuron_A':>10} {'Layer_B':>10} {'Neuron_B':>10} {'Synergy_Mean':>12} {'Synergy_Std':>12}")
            for _, row in top_stable_syn.iterrows():
                print(f"{row['layer_A']:>10} {row['neuron_A']:>10} {row['layer_B']:>10} {row['neuron_B']:>10} {row['synergy_mean']:>12.8f} {row['synergy_std']:>12.8f}")
        else:
            print("None")

        print(f"\n" + "="*60)
        print("Stable suppressive neuron pairs (Top 20)")
        print("="*60)
        if len(stable_suppressive) > 0:
            top_stable_sup = stable_suppressive.head(20)
            print(f"{'Layer_A':>10} {'Neuron_A':>10} {'Layer_B':>10} {'Neuron_B':>10} {'Synergy_Mean':>12} {'Synergy_Std':>12}")
            for _, row in top_stable_sup.iterrows():
                print(f"{row['layer_A']:>10} {row['neuron_A']:>10} {row['layer_B']:>10} {row['neuron_B']:>10} {row['synergy_mean']:>12.8f} {row['synergy_std']:>12.8f}")
        else:
            print("None")


        save_path = f"{RESULTS_DIR}stable_neuron_pairs_all_enhance.csv"
        stable_pairs_output = stable_pairs[['layer_A', 'neuron_A', 'layer_B', 'neuron_B',
                                            f'rel_{vals[0]}', 'synergy_mean', 'synergy_std'] + synergy_cols]
        stable_pairs_output = stable_pairs_output.rename(columns={f'rel_{vals[0]}': 'relationship'})
        stable_pairs_output.to_csv(save_path, index=False)
        print(f"\nStable neuron pairs saved to: {save_path}")

    return stable_pairs, unstable_pairs


def find_stable_pairs(merged, vals, threshold=90):
    print("\n" + "="*60)
    print(f"4. Stable neuron pairs (agreement >= {threshold}%)")
    print("="*60)


    rel_cols = [f'rel_{v}' for v in vals]
    merged['all_consistent'] = merged[rel_cols].apply(lambda x: x.nunique() == 1, axis=1)

    stable_pairs = merged[merged['all_consistent']]
    unstable_pairs = merged[~merged['all_consistent']]

    print(f"Stable neuron pairs: {len(stable_pairs)} ({len(stable_pairs)/len(merged)*100:.1f}%)")
    print(f"Unstable neuron pairs: {len(unstable_pairs)} ({len(unstable_pairs)/len(merged)*100:.1f}%)")


    if len(unstable_pairs) > 0:
        print(f"\nExample class distribution of unstable pairs:")
        sample = unstable_pairs.head(5)
        for _, row in sample.iterrows():
            rels = [row[f'rel_{v}'] for v in vals]
            pair = (row['layer_A'], row['neuron_A'], row['layer_B'], row['neuron_B'])
            print(f"  {pair}: {[r[:4] for r in rels]}")

    return stable_pairs, unstable_pairs


def analyze_top_pairs(results):
    print("\n" + "="*60)
    print("5. Stability of the top 10 synergistic/suppressive pairs")
    print("="*60)

    vals = sorted(results.keys())


    print("\nTop 10 synergistic pairs at each enhancement strength:")
    for val in vals:
        df = results[val]
        top10 = df.nlargest(10, 'synergy')
        print(f"\n  enhance={val}:")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            pair = (row['layer_A'], row['neuron_A'], row['layer_B'], row['neuron_B'])
            print(f"    {i}. {pair} synergy={row['synergy']:.6f}")


    print("\nOverlap of the top 10 synergistic pairs across enhancement strengths:")
    top_sets = {}
    for val in vals:
        df = results[val]
        top10 = df.nlargest(10, 'synergy')
        top_sets[val] = set()
        for _, row in top10.iterrows():
            pair = (row['layer_A'], row['neuron_A'], row['layer_B'], row['neuron_B'])
            top_sets[val].add(pair)

    for v1 in vals:
        for v2 in vals:
            if v1 < v2:
                intersection = top_sets[v1] & top_sets[v2]
                print(f"  enhance={v1} ∩ enhance={v2}: {len(intersection)} pairs appear in both top-10 lists")


def main(argv=None):
    global RESULTS_DIR, MODEL_NAME, ENHANCE_VALUES
    parser = argparse.ArgumentParser(description="Compare neuron interactions across enhancement strengths.")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--enhance-values", nargs="+", type=float, default=ENHANCE_VALUES)
    args = parser.parse_args(argv)
    RESULTS_DIR = str(Path(args.results_dir)) + os.sep
    MODEL_NAME = args.model_name
    ENHANCE_VALUES = args.enhance_values
    print("="*60)
    print("Sensitivity analysis comparison")
    print("="*60)


    results = load_results()

    if len(results) < 2:
        print("At least two result files are required for comparison")
        return

    vals = sorted(results.keys())


    corr_matrix, spearman_matrix = compute_correlation_matrix(results)


    merged, consistency_matrix = compute_classification_consistency(results)


    stable_pairs, unstable_pairs = find_stable_pairs_all(merged, vals, results)


    analyze_top_pairs(results)


    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    vals = sorted(results.keys())
    avg_spearman = []
    for i in range(len(vals)-1):
        for j in range(i+1, len(vals)):
            v1, v2 = vals[i], vals[j]
            corr, _ = stats.spearmanr(results[v1]['synergy'], results[v2]['synergy'])
            avg_spearman.append((v1, v2, corr))
            print(f"Spearman correlation ({v1}x vs {v2}x): {corr:.4f}")

    if avg_spearman:
        avg_corr = np.mean([c[2] for c in avg_spearman])
        print(f"\nMean Spearman correlation: {avg_corr:.4f}")


if __name__ == '__main__':
    main()

