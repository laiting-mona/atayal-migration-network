# %% [markdown]
# # 04 Community Detection

# %%
import pandas as pd
import networkx as nx
import community as community_louvain
import sys; sys.path.append("..")
from src.metrics import ei_index, network_summary

# %% [markdown]
# ## 4.1 Louvain
# %%
# G_u = G.to_undirected()
# partition = community_louvain.best_partition(G_u, weight="weight")
# modularity = community_louvain.modularity(partition, G_u, weight="weight")

# %% [markdown]
# ## 4.2 Community 與地理屬性比對
# %%
# df_comm = pd.DataFrame({"node": list(partition.keys()), "community": list(partition.values())})
# df_comm = df_comm.merge(nodes, left_on="node", right_on="tribe_name", how="left")

# %% [markdown]
# ## 4.3 E-I Index
# %%
# county_dict = dict(zip(nodes["tribe_name"], nodes["county"]))
# ei = ei_index(G, county_dict)

# %% [markdown]
# ## 4.4 各時段摘要比較表
# %%
# summary_rows = []
# for name, g in period_graphs.items():
#     s = network_summary(g)
#     s["period"] = name
#     summary_rows.append(s)
# pd.DataFrame(summary_rows).to_csv("../outputs/tables/period_comparison.csv", index=False)
