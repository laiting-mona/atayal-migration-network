# %% [markdown]
# # 06 Cross-Period Comparison

# %%
import pandas as pd
import matplotlib.pyplot as plt
import sys; sys.path.append("..")
from src.visualization import plot_multi_period, load_basemap

# %% [markdown]
# ## 6.1 指標趨勢圖
# %%
# df_summary = pd.read_csv("../outputs/tables/period_comparison.csv")
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# for ax, col in zip(axes.flat, ["avg_clustering","modularity","n_communities","centralization_betw"]):
#     ax.bar(df_summary["period"], df_summary[col])
#     ax.set_title(col)
# plt.tight_layout()
# plt.savefig("../outputs/figures/period_trends.png", dpi=300)

# %% [markdown]
# ## 6.2 多時段並排地圖
# %%
# basemap = load_basemap("../data/raw/gis/taiwan_counties.shp")
# plot_multi_period(basemap, period_data, save_path="../outputs/figures/multi_period_map.png")
