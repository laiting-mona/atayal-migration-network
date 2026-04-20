# %% [markdown]
# # 05 Spatial Visualization

# %%
import sys; sys.path.append("..")
from src.visualization import load_basemap, build_node_gdf, build_edge_gdf, plot_network_map, create_pyvis_network

# %% [markdown]
# ## 5.1 載入底圖
# %%
# basemap = load_basemap("../data/raw/gis/taiwan_counties.shp")

# %% [markdown]
# ## 5.2 建構空間資料
# %%
# gdf_nodes = build_node_gdf(nodes)
# node_coords = dict(zip(nodes["tribe_name"], zip(nodes["lon"], nodes["lat"])))
# gdf_edges = build_edge_gdf(edges, node_coords)

# %% [markdown]
# ## 5.3 網絡地圖
# %%
# plot_network_map(basemap, gdf_nodes, gdf_edges, save_path="../outputs/figures/network_map.png")

# %% [markdown]
# ## 5.4 pyVis 互動版
# %%
# create_pyvis_network(G, betw, "../outputs/interactive/atayal_network.html")
