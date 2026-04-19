# %% [markdown]
# # 02 Network Construction
# 從 edge_list.csv 建構各時段的 directed weighted network。

# %%
import pandas as pd
import networkx as nx
import sys; sys.path.append("..")
from src.network_builder import build_directed_network, build_period_networks

# %%
# edges = pd.read_csv("../data/processed/edge_list.csv")
# nodes = pd.read_csv("../data/processed/node_attributes.csv")

# %% [markdown]
# ## 2.1 建構整體網絡
# %%
# G = build_directed_network(edges)
# print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# %% [markdown]
# ## 2.2 建構各時段網絡
# %%
# period_graphs = build_period_networks(edges, period_col="period")

# %% [markdown]
# ## 2.3 基本網絡特徵
# %%
# print("Density:", nx.density(G))
# print("Weakly connected components:", nx.number_weakly_connected_components(G))
