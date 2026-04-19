# %% [markdown]
# # 03 Centrality Analysis

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys; sys.path.append("..")
from src.metrics import compute_all_centralities, freeman_centralization

# %% [markdown]
# ## 3.1 計算所有 centrality
# %%
# df_cent = compute_all_centralities(G)
# df_cent.sort_values("betweenness", ascending=False).head(10)

# %% [markdown]
# ## 3.2 Freeman Centralization
# %%
# fc = freeman_centralization(df_cent["betweenness"].to_dict())

# %% [markdown]
# ## 3.3 分布圖
# %%
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# for ax, col in zip(axes.flat, ["in_degree","out_degree","betweenness","clustering"]):
#     ax.hist(df_cent[col], bins=30, edgecolor="black", alpha=0.7)
#     ax.set_title(col)
# plt.tight_layout()
# plt.savefig("../outputs/figures/centrality_distribution.png", dpi=300)
