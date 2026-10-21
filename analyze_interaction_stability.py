
import pandas as pd
import numpy as np
from scipy import stats


RESULTS_DIR = "results/"
MODEL_NAME = "3.5-27B"


ENHANCE_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5]

def load_results():
    results = {}
    for val in ENHANCE_VALUES:
        filepath = f"{RESULTS_DIR}pairwise_neuron_synergy_{MODEL_NAME}_enhance_{val}.csv"
        try:
            df = pd.read_csv(filepath)
            results[val] = df
            print(f"✓ 加载 enhance={val}: {len(df)} 对神经元")
        except FileNotFoundError:
            print(f"✗ 文件不存在: {filepath}")
    return results


def compute_correlation_matrix(results):
    print("\n" + "="*60)
    print("1. Synergy 值相关系数矩阵 (Pearson)")
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
    print("2. Synergy 值排序相关系数 (Spearman)")
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
    print("3. 分类一致性分析 (协同/抑制)")
    print("="*60)

    vals = sorted(results.keys())
    n_pairs = len(results[vals[0]])

    print(f"总神经元对数: {n_pairs}")


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

    print("\n分类一致率 (%):")
    print(consistency_matrix.to_string(float_format="{:.1f}".format))


    print("\n各增强倍数的分类分布:")
    for val in vals:
        col = f'rel_{val}'
        synergistic = (merged[col] == 'synergistic').sum()
        suppressive = (merged[col] == 'suppressive').sum()
        independent = (merged[col] == 'independent').sum()
        print(f"  enhance={val}: 协同={synergistic} ({synergistic/n_pairs*100:.1f}%), "
              f"抑制={suppressive} ({suppressive/n_pairs*100:.1f}%), "
              f"独立={independent} ({independent/n_pairs*100:.1f}%)")

    return merged, consistency_matrix


def find_stable_pairs_all(merged, vals, results):
    print("\n" + "="*60)
    print("4. 稳定神经元对分析 (在所有增强倍数下分类一致)")
    print("="*60)

    rel_cols = [f'rel_{v}' for v in vals]


    merged['all_consistent'] = merged[rel_cols].apply(lambda x: x.nunique() == 1, axis=1)

    stable_pairs = merged[merged['all_consistent']].copy()
    unstable_pairs = merged[~merged['all_consistent']].copy()

    print(f"稳定神经元对: {len(stable_pairs)} ({len(stable_pairs)/len(merged)*100:.1f}%)")
    print(f"不稳定神经元对: {len(unstable_pairs)} ({len(unstable_pairs)/len(merged)*100:.1f}%)")

    if len(stable_pairs) > 0:

        stable_rel = stable_pairs[f'rel_{vals[0]}'].value_counts()
        print(f"\n稳定神经元对的分类分布:")
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
        print("稳定的协同神经元对 (Top 20)")
        print("="*60)
        if len(stable_synergistic) > 0:
            top_stable_syn = stable_synergistic.head(20)
            print(f"{'Layer_A':>10} {'Neuron_A':>10} {'Layer_B':>10} {'Neuron_B':>10} {'Synergy_Mean':>12} {'Synergy_Std':>12}")
            for _, row in top_stable_syn.iterrows():
                print(f"{row['layer_A']:>10} {row['neuron_A']:>10} {row['layer_B']:>10} {row['neuron_B']:>10} {row['synergy_mean']:>12.8f} {row['synergy_std']:>12.8f}")
        else:
            print("无")

        print(f"\n" + "="*60)
        print("稳定的抑制神经元对 (Top 20)")
        print("="*60)
        if len(stable_suppressive) > 0:
            top_stable_sup = stable_suppressive.head(20)
            print(f"{'Layer_A':>10} {'Neuron_A':>10} {'Layer_B':>10} {'Neuron_B':>10} {'Synergy_Mean':>12} {'Synergy_Std':>12}")
            for _, row in top_stable_sup.iterrows():
                print(f"{row['layer_A']:>10} {row['neuron_A']:>10} {row['layer_B']:>10} {row['neuron_B']:>10} {row['synergy_mean']:>12.8f} {row['synergy_std']:>12.8f}")
        else:
            print("无")


        save_path = f"{RESULTS_DIR}stable_neuron_pairs_all_enhance.csv"
        stable_pairs_output = stable_pairs[['layer_A', 'neuron_A', 'layer_B', 'neuron_B',
                                            f'rel_{vals[0]}', 'synergy_mean', 'synergy_std'] + synergy_cols]
        stable_pairs_output = stable_pairs_output.rename(columns={f'rel_{vals[0]}': 'relationship'})
        stable_pairs_output.to_csv(save_path, index=False)
        print(f"\n稳定神经元对已保存到: {save_path}")

    return stable_pairs, unstable_pairs


def find_stable_pairs(merged, vals, threshold=90):
    print("\n" + "="*60)
    print(f"4. 稳定神经元对分析 (一致率 >= {threshold}%)")
    print("="*60)


    rel_cols = [f'rel_{v}' for v in vals]
    merged['all_consistent'] = merged[rel_cols].apply(lambda x: x.nunique() == 1, axis=1)

    stable_pairs = merged[merged['all_consistent']]
    unstable_pairs = merged[~merged['all_consistent']]

    print(f"稳定神经元对: {len(stable_pairs)} ({len(stable_pairs)/len(merged)*100:.1f}%)")
    print(f"不稳定神经元对: {len(unstable_pairs)} ({len(unstable_pairs)/len(merged)*100:.1f}%)")


    if len(unstable_pairs) > 0:
        print(f"\n不稳定对的分类分布示例:")
        sample = unstable_pairs.head(5)
        for _, row in sample.iterrows():
            rels = [row[f'rel_{v}'] for v in vals]
            pair = (row['layer_A'], row['neuron_A'], row['layer_B'], row['neuron_B'])
            print(f"  {pair}: {[r[:4] for r in rels]}")

    return stable_pairs, unstable_pairs


def analyze_top_pairs(results):
    print("\n" + "="*60)
    print("5. Top 10 协同/抑制对的稳定性")
    print("="*60)

    vals = sorted(results.keys())


    print("\n各增强倍数Top 10协同对:")
    for val in vals:
        df = results[val]
        top10 = df.nlargest(10, 'synergy')
        print(f"\n  enhance={val}:")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            pair = (row['layer_A'], row['neuron_A'], row['layer_B'], row['neuron_B'])
            print(f"    {i}. {pair} synergy={row['synergy']:.6f}")


    print("\n各增强倍数Top 10协同对的交集:")
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
                print(f"  enhance={v1} ∩ enhance={v2}: {len(intersection)} 对共同出现在Top 10")


def main():
    print("="*60)
    print("敏感性分析结果对比")
    print("="*60)


    results = load_results()

    if len(results) < 2:
        print("需要至少2个结果文件才能进行对比分析")
        return

    vals = sorted(results.keys())


    corr_matrix, spearman_matrix = compute_correlation_matrix(results)


    merged, consistency_matrix = compute_classification_consistency(results)


    stable_pairs, unstable_pairs = find_stable_pairs_all(merged, vals, results)


    analyze_top_pairs(results)


    print("\n" + "="*60)
    print("总结")
    print("="*60)

    vals = sorted(results.keys())
    avg_spearman = []
    for i in range(len(vals)-1):
        for j in range(i+1, len(vals)):
            v1, v2 = vals[i], vals[j]
            corr, _ = stats.spearmanr(results[v1]['synergy'], results[v2]['synergy'])
            avg_spearman.append((v1, v2, corr))
            print(f"Spearman相关系数 ({v1}x vs {v2}x): {corr:.4f}")

    if avg_spearman:
        avg_corr = np.mean([c[2] for c in avg_spearman])
        print(f"\n平均Spearman相关系数: {avg_corr:.4f}")


if __name__ == '__main__':
    main()

