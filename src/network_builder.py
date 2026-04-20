"""
network_builder.py
建構 directed weighted network 的工具函式。
"""

import pandas as pd
import networkx as nx


def build_directed_network(df, source_col="source", target_col="target",
                           weight_col="weight"):
    """
    從 edge list DataFrame 建構 directed weighted graph。

    Parameters
    ----------
    df : pd.DataFrame
        至少包含 source, target, weight 三欄。
    source_col, target_col, weight_col : str
        欄位名稱。

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row[source_col],
            row[target_col],
            weight=row[weight_col]
        )
    return G


def build_period_networks(df, period_col="period", **kwargs):
    """
    依時間切片欄位分組，為每個時段建構獨立的 DiGraph。

    Returns
    -------
    dict[str, nx.DiGraph]
        key 為時段名稱，value 為對應的 DiGraph。
    """
    networks = {}
    for period, group in df.groupby(period_col):
        networks[period] = build_directed_network(group, **kwargs)
    return networks


def add_node_attributes(G, node_df, key_col="node"):
    """
    將節點屬性表合併至 graph 的節點屬性中。

    Parameters
    ----------
    G : nx.DiGraph
    node_df : pd.DataFrame
        必須包含 key_col 欄位，其餘欄位將作為節點屬性。
    """
    attr_dict = node_df.set_index(key_col).to_dict("index")
    nx.set_node_attributes(G, attr_dict)
    return G


def aggregate_edges(df, source_col="origin", target_col="destination",
                    count_col="count"):
    """
    將原始遷徙紀錄彙總為 edge list。
    同一組 (source, target) 的 count 加總。

    Returns
    -------
    pd.DataFrame
        columns: source, target, weight
    """
    edges = (
        df.groupby([source_col, target_col])[count_col]
        .sum()
        .reset_index()
    )
    edges.columns = ["source", "target", "weight"]
    return edges
