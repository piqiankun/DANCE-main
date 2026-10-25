
import numpy as np
import pandas as pd
import argparse
import os
from scipy.stats import spearmanr
import networkx as nx


DATA_DIR = 'data'
SAVE_DIR = "results/"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_contribution_matrix(csv_path):
    df = pd.read_csv(csv_path)


    sample_cols = [col for col in df.columns if col.startswith('sample_')]
    contribution_matrix = df[sample_cols].values


    neuron_ids = list(zip(df['layer'].values, df['neuron'].values))

    print(f"Contribution matrix shape: {contribution_matrix.shape} (n_neurons={contribution_matrix.shape[0]}, n_samples={contribution_matrix.shape[1]})")
    return contribution_matrix, neuron_ids


def compute_neuron_importance(contribution_matrix):
    abs_contribution = np.abs(contribution_matrix)
    importance = np.mean(abs_contribution, axis=1)

    importance = importance / importance.max()
    return importance


def compute_dcc_matrix(contribution_matrix):
    n_neurons, n_samples = contribution_matrix.shape
    print(f"\n--- Computing DCC matrix ---")
    print(f"Matrix shape: {n_neurons} neurons × {n_samples} samples")


    importance = compute_neuron_importance(contribution_matrix)
    print(f"Neuron importance computed. Range: [{importance.min():.4f}, {importance.max():.4f}]")

    dcc_matrix = np.zeros((n_neurons, n_neurons))

    for i in range(n_neurons):
        for j in range(i, n_neurons):
            x = contribution_matrix[i, :]
            y = contribution_matrix[j, :]

            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                dcc = 0.0
            else:
                dcc, _ = spearmanr(x, y)

            dcc_matrix[i, j] = dcc
            dcc_matrix[j, i] = dcc

    np.fill_diagonal(dcc_matrix, 1.0)


    upper_tri = dcc_matrix[np.triu_indices(n_neurons, k=1)]
    print(f"DCC matrix computed.")
    print(f"  |DCC| > 0.5: {(np.abs(upper_tri) > 0.5).sum()} ({(np.abs(upper_tri) > 0.5).mean()*100:.1f}%)")
    print(f"  |DCC| > 0.3: {(np.abs(upper_tri) > 0.3).sum()} ({(np.abs(upper_tri) > 0.3).mean()*100:.1f}%)")
    print(f"  |DCC| > 0.1: {(np.abs(upper_tri) > 0.1).sum()} ({(np.abs(upper_tri) > 0.1).mean()*100:.1f}%)")
    print(f"  Mean |DCC|: {np.abs(upper_tri).mean():.4f}")
    print(f"  Max DCC: {upper_tri.max():.4f}, Min DCC: {upper_tri.min():.4f}")

    return dcc_matrix, importance


def build_knn_graph(dcc_matrix, importance, k=10, weight_threshold=0.0):
    n_neurons = len(importance)
    print(f"\n--- Building kNN graph (k={k}, weight_threshold={weight_threshold}) ---")
    print(f"Edge weight formula: w_ij = |DCC_ij| * I_i * I_j")


    weight_matrix = np.abs(dcc_matrix) * np.outer(importance, importance)
    np.fill_diagonal(weight_matrix, 0)

    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))


    for i in range(n_neurons):

        weights = weight_matrix[i, :]

        top_k_idx = np.argsort(weights)[::-1][:k]
        for j in top_k_idx:
            if j != i and weights[j] > weight_threshold:
                G.add_edge(i, j, weight=weights[j], dcc=dcc_matrix[i, j])


    if G.number_of_edges() > 0:
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
        print(f"  Mean weight: {np.mean(weights):.4f}")


        pos_edges = sum(1 for _, _, d in G.edges(data=True) if d.get('dcc', 0) > 0)
        neg_edges = G.number_of_edges() - pos_edges
        print(f"  Positive DCC edges: {pos_edges}, Negative DCC edges: {neg_edges}")
    else:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def detect_communities_louvain(G, resolution=1.0, random_state=42):
    if G.number_of_edges() == 0:
        print("  No edges in graph, returning isolated nodes as separate communities")
        partition = {n: n for n in G.nodes()}
        return partition, 0.0, G.number_of_nodes()

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, weight='weight', resolution=resolution, random_state=random_state)
        mod = community_louvain.modularity(partition, G, weight='weight')
    except ImportError:
        print("  Warning: No community detection library available. Using trivial partition.")
        partition = {n: n for n in G.nodes()}
        mod = 0.0

    n_communities = len(set(partition.values()))
    return partition, mod, n_communities


def analyze_communities(G, partition):
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)


    communities = dict(sorted(communities.items(), key=lambda x: len(x[1]), reverse=True))


    print(f"  Communities: {len(communities)}, nodes: {sum(len(v) for v in communities.values())}")
    for comm_id, nodes in list(communities.items())[:5]:
        print(f"    Comm {comm_id}: {len(nodes)} neurons")

    return communities


def get_community_edges(G, communities, dcc_matrix, neuron_ids, importance, min_dcc=0.0):
    all_edges = []
    for comm_id, nodes in communities.items():
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                if G.has_edge(n1, n2):
                    dcc = dcc_matrix[n1, n2]
                    if abs(dcc) > min_dcc:
                        all_edges.append((
                            neuron_ids[n1], neuron_ids[n2],
                            dcc, importance[n1], importance[n2], comm_id
                        ))


    all_edges.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"  Candidate pairs: {len(all_edges)} (|DCC| > {min_dcc})")

    return all_edges


def save_candidate_results(candidate_edges, model_name, save_dir):
    import datetime

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    if candidate_edges:
        df = pd.DataFrame([
            {
                'layer_i': e[0][0], 'neuron_i': e[0][1],
                'layer_j': e[1][0], 'neuron_j': e[1][1],
                'dcc': e[2],
                'importance_i': e[3],
                'importance_j': e[4],
                'community_id': e[5],
            }
            for e in candidate_edges
        ])

        df = df.iloc[df['dcc'].abs().argsort()[::-1]].reset_index(drop=True)

        save_path = os.path.join(save_dir, f'candidate_neurons_{model_name}_{timestamp}.csv')
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
        print(f"Total candidate pairs: {len(df)}")
        return df
    else:
        print("\nNo candidate pairs found.")
        return None


def print_candidate_summary(community_edges, top_n=20):
    print("\n" + "="*80)
    print("CANDIDATE NEURON PAIRS SUMMARY")
    print("="*80)

    print(f"\nTotal candidate pairs: {len(community_edges)}")

    if community_edges:

        pos_count = sum(1 for e in community_edges if e[2] > 0)
        neg_count = len(community_edges) - pos_count
        print(f"  Positive DCC (synergy): {pos_count} ({pos_count/len(community_edges)*100:.1f}%)")
        print(f"  Negative DCC (inhibition): {neg_count} ({neg_count/len(community_edges)*100:.1f}%)")


        print(f"\n--- Top {top_n} Strongest Pairs (by |DCC|) ---")
        header = f"{'Rank':<6} {'DCC':<10} {'Layer_i':<8} {'Neuron_i':<10} {'Layer_j':<8} {'Neuron_j':<10} {'Comm':<6}"
        print(header)
        print("-" * 70)
        for i, edge in enumerate(community_edges[:top_n]):
            ni, nj, dcc, imp_i, imp_j, comm = edge
            print(f"{i+1:<6} {dcc:+.4f} {ni[0]:<8} {ni[1]:<10} {nj[0]:<8} {nj[1]:<10} {comm:<6}")


def main():
    parser = argparse.ArgumentParser(
        description='Neuron functional-module identification through community detection'
    )
    parser.add_argument('--input', type=str,
                        default=os.path.join(DATA_DIR, 'neuron_bias_contributions.csv'),
                        help='Input CSV file path')
    parser.add_argument('--k_neighbors', type=int, default=10,
                        help='Number of neighbors in the kNN graph (default: 10)')
    parser.add_argument('--weight_threshold', type=float, default=0.0,
                        help='Edge-weight threshold (default: 0.0; retain only edges with w_ij > threshold)')
    parser.add_argument('--resolution', type=float, default=1.0,
                        help='Louvain resolution (default: 1.0)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str,
                        default=SAVE_DIR,
                        help='Output directory')
    parser.add_argument('--min_dcc', type=float, default=0.1,
                        help='Minimum |DCC| threshold (default: 0.1; retain candidate pairs with |DCC| > threshold)')

    args = parser.parse_args()


    model_name = 'llama3.1-8b-instruct'
    if 'llama' in args.input.lower():
        model_name = 'llama3.1-8b-instruct'

    print("="*60)
    print("Neuron functional-module identification (DCC + Louvain community detection)")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"k neighbors: {args.k_neighbors}")
    print(f"Weight threshold: {args.weight_threshold}")
    print(f"Resolution: {args.resolution}")
    print(f"Random state: {args.random_state}")


    contribution_matrix, neuron_ids = load_contribution_matrix(args.input)


    dcc_matrix, importance = compute_dcc_matrix(contribution_matrix)


    G = build_knn_graph(dcc_matrix, importance, k=args.k_neighbors, weight_threshold=args.weight_threshold)


    print(f"\n{'='*60}")
    print("COMMUNITY DETECTION")
    print(f"{'='*60}")
    partition, mod, n_communities = detect_communities_louvain(
        G, resolution=args.resolution, random_state=args.random_state
    )
    print(f"Detected {n_communities} communities, modularity: {mod:.4f}")
    communities = analyze_communities(G, partition)


    candidate_edges = get_community_edges(G, communities, dcc_matrix, neuron_ids, importance, min_dcc=args.min_dcc)


    save_candidate_results(candidate_edges, model_name, args.output_dir)


    print_candidate_summary(candidate_edges)


if __name__ == '__main__':
    main()


