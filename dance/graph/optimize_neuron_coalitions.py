from dance.paths import PROJECT_ROOT

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def rank_normalize(values, *, keep_zero=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank normalization requires a finite one-dimensional array")
    if keep_zero and np.any(values < 0):
        raise ValueError("Coalition influence must be nonnegative")
    if len(values) == 0:
        return values.copy()
    if len(values) == 1:
        result = np.full(1, 0.5)
    else:
        result = (pd.Series(values).rank(method="average").to_numpy() - 1) / (len(values) - 1)
    if keep_zero:
        result[values == 0] = 0.0
    return result


def candidate_pairs_from_profiles(profiles):
    import networkx as nx

    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] < 2 or not np.isfinite(profiles).all():
        raise ValueError("profiles must be a finite N-by-M array with M >= 2")
    n = len(profiles)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    if n < 2:
        return [], graph, [{i} for i in range(n)]

    ranked = pd.DataFrame(profiles).rank(axis=1, method="average").to_numpy()
    centered = ranked - ranked.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    unit = np.divide(centered, norms[:, None], out=np.zeros_like(centered), where=norms[:, None] > 0)
    corr = unit @ unit.T
    np.fill_diagonal(corr, -np.inf)
    nearest = np.argmax(corr, axis=1)
    for i, j in enumerate(nearest):
        if i < j and nearest[j] == i and corr[i, j] > 0:
            graph.add_edge(i, int(j), weight=float(corr[i, j]))
    if graph.number_of_edges() == 0:
        communities = [{i} for i in range(n)]
    else:
        warnings.warn("The mutual 1-nearest-neighbor graph forms at most two-node pairs; Louvain cannot recover larger connected communities.", stacklevel=2)
        communities = list(nx.community.louvain_communities(graph, weight="weight", seed=0))
    communities = sorted((set(c) for c in communities), key=lambda c: min(c))
    pairs = sorted(pair for c in communities for pair in combinations(sorted(c), 2))
    return pairs, graph, communities


def verify_interactions(candidate_pairs, ratios, evaluate):
    ratios = np.asarray(list(ratios), dtype=float)
    if ratios.ndim != 1 or len(ratios) == 0 or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError("ratios must contain finite positive activation multipliers")
    pairs = sorted({tuple(sorted(pair)) for pair in candidate_pairs})
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError("Each candidate pair must contain two distinct neurons")

    def score(interventions):
        value = float(evaluate(interventions))
        if not np.isfinite(value):
            raise ValueError("evaluate returned a non-finite score")
        return value

    baseline = score({})
    singles = {}
    edges, types, records = [], [], []
    for i, j in pairs:
        effects = []
        for ratio in ratios:
            ratio = float(ratio)
            for neuron in (i, j):
                key = (neuron, ratio)
                if key not in singles:
                    singles[key] = score({neuron: ratio})
            joint = score({i: ratio, j: ratio})
            effect = joint - singles[i, ratio] - singles[j, ratio] + baseline
            effects.append(effect)
            records.append(dict(i=i, j=j, ratio=ratio, P0=baseline,
                                Pi=singles[i, ratio], Pj=singles[j, ratio],
                                Pij=joint, interaction=effect))
        if all(effect > 0 for effect in effects):
            edges.append((i, j))
            types.append(0)
        elif all(effect < 0 for effect in effects):
            edges.append((i, j))
            types.append(1)
    return np.asarray(edges, dtype=int).reshape(-1, 2).T, np.asarray(types, dtype=int), pd.DataFrame(records)


@dataclass
class Coalition:
    id: int
    type: int
    seed_neuron: int
    members: set = field(default_factory=set)
    seed_neurons: set = field(default_factory=set)
    member_strength: dict = field(default_factory=dict)
    growth_trace: list = field(default_factory=list)
    score: float = 0.0
    debias_score: float = 0.0
    general_score: float = 0.0
    crosslang_score: float = 0.0

    def __post_init__(self):
        self.members.add(self.seed_neuron)
        self.seed_neurons.add(self.seed_neuron)

    def size(self):
        return len(self.members)


class CoalitionGrowing:

    def __init__(self, num_neurons, node_features, edge_index, edge_type,
                 raw_features, device="cpu"):
        self.raw = np.array(_as_numpy(raw_features), dtype=float, copy=True)
        if self.raw.shape != (num_neurons, 3) or not np.isfinite(self.raw).all():
            raise ValueError("raw_features must be a finite N-by-3 array")
        self.N = num_neurons
        self.device = device
        self.node_features = np.column_stack([rank_normalize(self.raw[:, k]) for k in range(3)])
        self.node_contributions = self.node_features.mean(axis=1)
        edges, types = _as_numpy(edge_index), _as_numpy(edge_type)
        if edges.ndim != 2 or edges.shape[0] != 2 or types.shape != (edges.shape[1],):
            raise ValueError("edge_index must have shape [2, E] and edge_type must have shape [E]")
        if not np.isfinite(edges).all() or np.any(edges != np.floor(edges)):
            raise ValueError("Edge indices must be integers")
        if np.any(edges < 0) or np.any(edges >= self.N) or not np.isin(types, [0, 1]).all():
            raise ValueError("Edge index out of bounds or invalid edge type")
        canonical = {}
        for (i, j), t in zip(edges.T.astype(int), types.astype(int)):
            if i == j:
                raise ValueError("Self-loops are not allowed in the signed graph")
            pair = tuple(sorted((int(i), int(j))))
            if pair in canonical and canonical[pair] != t:
                raise ValueError(f"An edge cannot be both synergistic and suppressive: {pair}")
            canonical[pair] = int(t)
        pairs = sorted(canonical)
        self.edge_index = np.asarray(pairs, dtype=int).reshape(-1, 2).T
        self.edge_type = np.array([canonical[pair] for pair in pairs], dtype=int)
        self.syn_neighbors = [set() for _ in range(self.N)]
        self.sup_neighbors = [set() for _ in range(self.N)]
        adjacency = np.zeros((self.N, self.N))
        for (i, j), t in zip(self.edge_index.T, self.edge_type):
            i, j = int(i), int(j)
            neighbors = self.syn_neighbors if t == 0 else self.sup_neighbors
            neighbors[i].add(j)
            neighbors[j].add(i)
            adjacency[i, j] = adjacency[j, i] = 1.0 if t == 0 else -1.0
        degree = np.abs(adjacency).sum(axis=1)
        invsqrt = np.zeros(self.N)
        np.divide(1.0, np.sqrt(degree), out=invsqrt, where=degree > 0)
        self.normalized_adjacency_ = invsqrt[:, None] * adjacency * invsqrt[None, :]
        self.coalitions = []
        self.has_run_ = False

    def _compute_node_contribution(self, neuron_idx):
        return float(self.node_contributions[neuron_idx])

    def _member_cohesion(self, members, coalition_type):
        if len(members) < 2:
            return 0.0
        matching = self.syn_neighbors if coalition_type == 0 else self.sup_neighbors
        opposite = self.sup_neighbors if coalition_type == 0 else self.syn_neighbors
        difference = sum(int(j in matching[i]) - int(j in opposite[i])
                         for i, j in combinations(sorted(members), 2))
        return difference / (len(members) * (len(members) - 1) / 2)

    def _score_members(self, members, coalition_type):
        mean_q = self.node_contributions[sorted(members)].mean()
        cohesion = self._member_cohesion(members, coalition_type)
        return float(0.5 * (mean_q + (1 + cohesion) / 2))

    def _compute_coalition_score(self, coalition):
        coalition.score = self._score_members(coalition.members, coalition.type)
        means = self.raw[sorted(coalition.members)].mean(axis=0)
        coalition.debias_score, coalition.general_score, coalition.crosslang_score = map(float, means)
        return coalition.score

    def _candidate_neighbors(self, coalition):
        neighbors = self.syn_neighbors if coalition.type == 0 else self.sup_neighbors
        return set().union(*(neighbors[i] for i in coalition.members)) - coalition.members

    def _growth_candidate_score(self, coalition, candidate):
        if candidate not in self._candidate_neighbors(coalition):
            return -float("inf")
        return (self._score_members(coalition.members | {candidate}, coalition.type)
                - self._score_members(coalition.members, coalition.type))

    def _grow_coalition(self, coalition):
        self._compute_coalition_score(coalition)
        while True:
            candidates = sorted(self._candidate_neighbors(coalition))
            if not candidates:
                break
            gains = {i: self._growth_candidate_score(coalition, i) for i in candidates}
            best = max(candidates, key=lambda i: (gains[i], -i))
            if gains[best] <= 0:
                break
            before = coalition.score
            coalition.members.add(best)
            self._compute_coalition_score(coalition)
            coalition.growth_trace.append(dict(candidate=best, marginal_gain=gains[best],
                                                score_before=before, score_after=coalition.score))

    def _grow_all_coalitions(self):
        self.coalitions = []
        self.selected_seeds_ = {}
        for t, neighbors in enumerate((self.syn_neighbors, self.sup_neighbors)):

            seeds = [i for i in range(self.N) if neighbors[i]
                     and all(self.raw[i, 0] >= self.raw[j, 0] for j in neighbors[i])]
            self.selected_seeds_[t] = seeds
            unique = {}
            for seed in seeds:
                coalition = Coalition(-1, t, seed)
                self._grow_coalition(coalition)
                if coalition.size() == 1:
                    continue
                key = frozenset(coalition.members)
                if key in unique:
                    unique[key].seed_neurons.add(seed)
                else:
                    unique[key] = coalition
            self.coalitions.extend(unique.values())
        for cid, coalition in enumerate(self.coalitions):
            coalition.id = cid

    def _propagate_coalition_influence(self):
        self.participation_ = np.zeros((self.N, len(self.coalitions)))
        for coalition in self.coalitions:
            z = np.zeros(self.N)
            z[sorted(coalition.members)] = 1.0 / coalition.size()
            adjacency = self.normalized_adjacency_ * (1 if coalition.type == 0 else -1)
            mass = np.maximum(z + adjacency @ z, 0)

            mass = mass / mass.sum() if mass.sum() > 0 else z
            self.participation_[:, coalition.id] = mass
            coalition.member_strength = {int(i): float(mass[i]) for i in np.flatnonzero(mass)}

    def _coalition_competition(self):
        scores = np.array([c.score for c in self.coalitions])
        weighted = self.participation_ * scores[None, :]
        total = weighted.sum(axis=1, keepdims=True)
        self.competition_ = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
        self.neuron_coalition_strength = [defaultdict(float) for _ in range(self.N)]
        for i, cid in zip(*np.nonzero(self.competition_)):
            self.neuron_coalition_strength[i][int(cid)] = float(self.competition_[i, cid])

    def run(self):
        self._grow_all_coalitions()
        self._propagate_coalition_influence()
        self._coalition_competition()
        self.has_run_ = True
        return self.coalitions

    def compute_neuron_weights(self, gamma):
        if not self.has_run_:
            raise RuntimeError("Call run() first")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        influence = np.zeros((self.N, 2))
        for coalition in self.coalitions:
            influence[:, coalition.type] += self.competition_[:, coalition.id] * coalition.score
        self.influence_ = influence
        self.u_ = (rank_normalize(influence[:, 0], keep_zero=True)
                   - rank_normalize(influence[:, 1], keep_zero=True)
                   + rank_normalize(self.node_contributions)) / 3
        self.relative_weights_ = (1 + np.tanh(self.u_)) / 2
        self.weights_ = float(gamma) * self.relative_weights_
        return self.weights_.copy()


class CoalitionGraphOptimizer:

    def __init__(self, device="cpu"):
        self.device = device

    def build_graph(self, bias_df, general_df, crosslang_df,
                    synergistic_pairs, suppressive_pairs):
        frames = []
        for label, frame in zip(("bias", "general", "crosslang"), (bias_df, general_df, crosslang_df)):
            if not {"layer", "neuron", "weight"} <= set(frame.columns):
                raise ValueError(f"{label} is missing layer/neuron/weight columns")
            ids = frame[["layer", "neuron"]].to_numpy(dtype=float)
            if not np.isfinite(ids).all() or np.any(ids < 0) or np.any(ids != np.floor(ids)):
                raise ValueError(f"{label} neuron coordinates must be nonnegative integers")
            keys = list(map(tuple, ids.astype(int)))
            if len(set(keys)) != len(keys):
                raise ValueError(f"{label} contains duplicate neurons")
            values = frame["weight"].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{label} contains non-finite weights")
            frames.append(dict(zip(keys, values)))
        self.neurons = list(frames[0])
        if not self.neurons:
            raise ValueError("The shared-neuron set must not be empty")
        for feature in frames[1:]:
            if set(self.neurons) - feature.keys():
                raise ValueError("General/cross-language features are missing shared neurons; implicit zero-filling is not allowed")
        self.num_neurons = len(self.neurons)
        self.neuron_to_idx = {neuron: i for i, neuron in enumerate(self.neurons)}
        self.idx_to_neuron = dict(enumerate(self.neurons))
        self.raw_features = np.array([[feature[neuron] for feature in frames] for neuron in self.neurons])
        edges, types = [], []
        for t, pairs in enumerate((synergistic_pairs, suppressive_pairs)):
            for pair in pairs:
                coords = np.asarray(pair, dtype=float)
                if coords.shape != (4,) or not np.isfinite(coords).all() or np.any(coords != np.floor(coords)):
                    raise ValueError("Each neuron pair must contain four integers: layer_A, neuron_A, layer_B, neuron_B")
                a, b = tuple(coords[:2].astype(int)), tuple(coords[2:].astype(int))
                if a not in self.neuron_to_idx or b not in self.neuron_to_idx:
                    raise ValueError(f"A relation contains non-shared neurons: {a}, {b}")
                edges.append((self.neuron_to_idx[a], self.neuron_to_idx[b]))
                types.append(t)
        self.edge_index = np.asarray(edges, dtype=int).reshape(-1, 2).T
        self.edge_type = np.asarray(types, dtype=int)
        self.graph_source_ = "provided_edges_not_reverified"
        return self

    def build_graph_from_profiles(self, bias_df, general_df, crosslang_df,
                                  profiles, ratios, evaluate):
        self.build_graph(bias_df, general_df, crosslang_df, [], [])
        if len(profiles) != self.num_neurons:
            raise ValueError("The number of profile rows must equal the shared-neuron count")
        pairs, self.similarity_graph_, self.communities_ = candidate_pairs_from_profiles(profiles)
        self.candidate_pairs_ = pairs
        self.edge_index, self.edge_type, self.interactions_ = verify_interactions(pairs, ratios, evaluate)
        self.graph_source_ = "profiles_and_multi_ratio_verification"
        return self

    def optimize(self, gamma):
        self.coalition_growing = CoalitionGrowing(self.num_neurons, None, self.edge_index,
                                                 self.edge_type, self.raw_features, self.device)
        self.coalitions = self.coalition_growing.run()
        self.weights_ = self.coalition_growing.compute_neuron_weights(gamma)
        self.node_features = self.coalition_growing.node_features
        return self.weights_.copy()

    def get_neuron_coalitions(self, neuron_idx):
        engine = self.coalition_growing
        return {cid: dict(type="SD" if engine.coalitions[cid].type == 0 else "SU",
                          strength=strength, participation=float(engine.participation_[neuron_idx, cid]),
                          score=engine.coalitions[cid].score,
                          core_member=neuron_idx in engine.coalitions[cid].members)
                for cid, strength in engine.neuron_coalition_strength[neuron_idx].items()}


def load_data(base_path):
    base = Path(base_path)
    frames = [
        pd.read_csv(base / name)
        for name in (
            "bias_neuron_scores.csv",
            "general_neuron_scores.csv",
            "cross_language_neuron_scores.csv",
        )
    ]
    pairs = pd.read_csv(base / "stable_neuron_pairs.csv")
    required = ["layer_A", "neuron_A", "layer_B", "neuron_B", "relationship"]
    if not set(required) <= set(pairs):
        raise ValueError(f"The relation table must contain {required}")
    labels = pairs["relationship"].astype(str).str.strip().str.lower()
    if not labels.isin(["synergistic", "suppressive"]).all():
        raise ValueError("relationship must be synergistic or suppressive; unknown values cannot default to suppressive")
    syn = list(pairs.loc[labels == "synergistic", required[:4]].itertuples(index=False, name=None))
    sup = list(pairs.loc[labels == "suppressive", required[:4]].itertuples(index=False, name=None))
    return *frames, syn, sup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing the four input CSV files")
    parser.add_argument("--gamma", type=float, required=True, help="Global enhancement scaling factor")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "neuron_coalition_weights-section35.csv")
    args = parser.parse_args(argv)
    if not np.isfinite(args.gamma) or args.gamma <= 0:
        parser.error("--gamma must be finite and positive")
    optimizer = CoalitionGraphOptimizer().build_graph(*load_data(args.data_dir))
    weights = optimizer.optimize(gamma=args.gamma)
    engine = optimizer.coalition_growing
    records = []
    for i, (layer, neuron) in optimizer.idx_to_neuron.items():
        memberships = optimizer.get_neuron_coalitions(i)
        cid = max(memberships, key=lambda key: memberships[key]["strength"]) if memberships else -1
        records.append(dict(layer=int(layer), neuron=int(neuron), final_weight=float(weights[i]),
                            relative_weight=float(engine.relative_weights_[i]),
                            composite_weight=float(engine.node_contributions[i]),
                            bias_contribution=float(engine.raw[i, 0]), general_contribution=float(engine.raw[i, 1]),
                            crosslang_contribution=float(engine.raw[i, 2]),
                            synergy_influence=float(engine.influence_[i, 0]),
                            suppression_influence=float(engine.influence_[i, 1]),
                            combined_effect=float(engine.u_[i]), main_coalition=cid,
                            main_coalition_type=memberships[cid]["type"] if cid >= 0 else "N/A",
                            coalition_strength=memberships[cid]["strength"] if cid >= 0 else 0.0))
    result = pd.DataFrame(records).sort_values("final_weight", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    details = dict(gamma=args.gamma, graph_source=optimizer.graph_source_,
                   rank_normalization="average rank, (rank-1)/(N-1); zero influence stays zero",
                   node_ids=[[int(layer), int(neuron)] for layer, neuron in optimizer.neurons],
                   selected_seeds=engine.selected_seeds_, coalitions=[])
    for c in optimizer.coalitions:
        details["coalitions"].append(dict(id=c.id, type="SD" if c.type == 0 else "SU",
                                          seeds=sorted(c.seed_neurons), members=sorted(c.members),
                                          score=c.score, participation=c.member_strength,
                                          growth_trace=c.growth_trace))
    args.output.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Neurons={optimizer.num_neurons}, Unique edges={engine.edge_index.shape[1]}, Coalitions={len(engine.coalitions)}")
    print(f"Results: {args.output}; Explanation data: {args.output.with_suffix('.json')}")
    print("Input edges were loaded from existing CSV files; interaction verification across enhancement ratios was not rerun.")
    return result, optimizer


if __name__ == "__main__":
    main()
