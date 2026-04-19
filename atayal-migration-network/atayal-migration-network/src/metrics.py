"""
metrics.py
自定義的網絡分析指標，補充 NetworkX 未內建的函式。
"""

import networkx as nx
import numpy as np


def freeman_centralization(centrality_dict):
    """
    計算 Freeman degree centralization。

    CD = sum(C_max - C_i) / max_possible

    Parameters
    ----------
    centrality_dict : dict
        key 為節點，value 為該節點的 centrality 值。

    Returns
    -------
    float
        值域 [0, 1]。1 表示完全集中化（star），0 表示完全平等（circle）。
    """
    values = list(centrality_dict.values())
    if len(values) < 3:
        return 0.0
    max_val = max(values)
    n = len(values)
    numerator = sum(max_val - v for v in values)
    denominator = (n - 1) * (n - 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def ei_index(G, attribute_dict):
    """
    計算 E-I Index（Krackhardt & Stern, 1988）。

    EI = (External - Internal) / (External + Internal)

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    attribute_dict : dict
        key 為節點，value 為該節點的分類屬性（如縣市、亞群）。

    Returns
    -------
    float
        值域 [-1, 1]。-1 完全同質，+1 完全異質。
    """
    internal = 0
    external = 0
    for u, v in G.edges():
        if attribute_dict.get(u) == attribute_dict.get(v):
            internal += 1
        else:
            external += 1
    total = external + internal
    if total == 0:
        return 0.0
    return (external - internal) / total


def compute_all_centralities(G, weight="weight"):
    """
    一次計算所有 centrality 指標，回傳整合的 DataFrame。

    Returns
    -------
    pd.DataFrame
        index 為節點名稱，columns 為各指標。
    """
    import pandas as pd

    in_deg = dict(G.in_degree(weight=weight))
    out_deg = dict(G.out_degree(weight=weight))
    betw = nx.betweenness_centrality(G, weight=weight, normalized=True)

    # eigenvector centrality 在 directed graph 上可能不收斂，
    # 不收斂時改用 PageRank
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000, weight=weight)
    except nx.PowerIterationFailedConvergence:
        eigen = nx.pagerank(G, weight=weight)

    # clustering coefficient 轉為 undirected 計算
    G_u = G.to_undirected()
    clust = nx.clustering(G_u, weight=weight)

    df = pd.DataFrame({
        "in_degree": in_deg,
        "out_degree": out_deg,
        "betweenness": betw,
        "eigenvector": eigen,
        "clustering": clust,
    })
    df.index.name = "node"
    return df


def network_summary(G, partition=None, weight="weight"):
    """
    計算網絡層級的摘要指標。

    Parameters
    ----------
    G : nx.DiGraph
    partition : dict, optional
        community detection 的結果（node -> community_id）。
    weight : str

    Returns
    -------
    dict
    """
    import community as community_louvain

    G_u = G.to_undirected()
    clust_values = list(nx.clustering(G_u, weight=weight).values())
    betw_values = list(
        nx.betweenness_centrality(G, weight=weight, normalized=True).values()
    )

    if partition is None:
        partition = community_louvain.best_partition(G_u, weight=weight)

    modularity = community_louvain.modularity(partition, G_u, weight=weight)

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": np.mean(clust_values) if clust_values else 0,
        "modularity": modularity,
        "n_communities": len(set(partition.values())),
        "centralization_betw": freeman_centralization(
            dict(zip(G.nodes(), betw_values))
        ),
    }
