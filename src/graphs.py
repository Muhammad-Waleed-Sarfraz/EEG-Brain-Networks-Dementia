import numpy as np
import networkx as nx


def _prep(W):
    """Weighted graph with edge length = 1 / weight (strong link = short distance)."""
    G = nx.from_numpy_array(W)
    for u, v, d in G.edges(data=True):
        d["length"] = 1.0 / max(d["weight"], 1e-6)
    return G


def _path_and_eff(G):
    """Characteristic path length and global efficiency from shortest paths on 'length'."""
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    vals = [dist[i][j] for i in G for j in G if i != j]
    return float(np.mean(vals)), float(np.mean([1.0 / v for v in vals]))


def small_world(W, n_rand=100, seed=42):
    """sigma = (C / C_rand) / (L / L_rand), plus the three ratios to weight-shuffled graphs."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(W, k=1)
    w = W[iu]
    G = _prep(W)
    L, E = _path_and_eff(G)
    C = nx.average_clustering(G, weight="weight")
    Cr, Lr, Er = [], [], []
    for _ in range(n_rand):
        Wr = np.zeros_like(W)
        Wr[iu] = rng.permutation(w)
        Wr = Wr + Wr.T
        Gr = _prep(Wr)
        l, e = _path_and_eff(Gr)
        Lr.append(l)
        Er.append(e)
        Cr.append(nx.average_clustering(Gr, weight="weight"))
    c_norm, l_norm, e_norm = C / np.mean(Cr), L / np.mean(Lr), E / np.mean(Er)
    return {"small_world": float(c_norm / l_norm), "c_norm": float(c_norm),
            "l_norm": float(l_norm), "e_norm": float(e_norm)}


def graph_metrics(W, seed=42, n_rand=100):
    """Global weighted graph measures for one 19 x 19 connectivity matrix."""
    G = _prep(W)
    L, E = _path_and_eff(G)
    C = nx.average_clustering(G, weight="weight")
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    Q = nx.community.modularity(G, comms, weight="weight")
    S = float(np.mean([d for _, d in G.degree(weight="weight")]))
    out = {"strength": S, "clustering": float(C), "path_length": L, "efficiency": E,
           "modularity": float(Q)}
    out.update(small_world(W, n_rand=n_rand, seed=seed))
    return out


def binary_metrics(W, keep=(0.1, 0.2, 0.3)):
    """Robustness check: clustering and efficiency on graphs keeping the strongest edges."""
    iu = np.triu_indices_from(W, k=1)
    out = {}
    for p in keep:
        thr = np.quantile(W[iu], 1 - p)
        A = (W >= thr).astype(int)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        out[f"bin{int(p * 100)}_clustering"] = float(nx.average_clustering(G))
        out[f"bin{int(p * 100)}_efficiency"] = float(nx.global_efficiency(G))
    return out
