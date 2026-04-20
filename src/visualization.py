"""
visualization.py
GeoPandas 空間視覺化與 pyVis 互動式網絡圖的工具函式。
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


ATAYAL_COUNTIES = [
    "新北市", "桃園市", "新竹縣", "苗栗縣", "臺中市", "南投縣", "宜蘭縣"
]


def load_basemap(shapefile_path, counties=None):
    """
    載入台灣縣市邊界 shapefile，篩選泰雅族分布區域。
    """
    taiwan = gpd.read_file(shapefile_path)
    if counties is None:
        counties = ATAYAL_COUNTIES
    return taiwan[taiwan["COUNTYNAME"].isin(counties)]


def build_node_gdf(node_df, lon_col="lon", lat_col="lat"):
    """
    將節點屬性表轉為 GeoDataFrame。
    """
    return gpd.GeoDataFrame(
        node_df,
        geometry=gpd.points_from_xy(node_df[lon_col], node_df[lat_col]),
        crs="EPSG:4326"
    )


def build_edge_gdf(edge_df, node_coords):
    """
    將邊列表轉為 GeoDataFrame（LineString）。

    Parameters
    ----------
    edge_df : pd.DataFrame
        columns: source, target, weight
    node_coords : dict
        key 為節點名稱，value 為 (lon, lat) tuple。
    """
    lines = []
    for _, row in edge_df.iterrows():
        src = node_coords.get(row["source"])
        tgt = node_coords.get(row["target"])
        if src is None or tgt is None:
            continue
        lines.append({
            "geometry": LineString([src, tgt]),
            "weight": row["weight"],
            "source": row["source"],
            "target": row["target"],
        })
    return gpd.GeoDataFrame(lines, crs="EPSG:4326")


def plot_network_map(basemap, gdf_nodes, gdf_edges,
                     size_col="betweenness", color_col="community",
                     title="Atayal Migration Network",
                     figsize=(12, 16), save_path=None):
    """
    繪製遷徙網絡地圖：底圖 + 邊 + 節點。

    Parameters
    ----------
    basemap : GeoDataFrame
        台灣縣市邊界。
    gdf_nodes : GeoDataFrame
        含 centrality 指標與 community 欄位。
    gdf_edges : GeoDataFrame
        含 weight 欄位。
    size_col : str
        用來映射節點大小的欄位名稱。
    color_col : str
        用來映射節點顏色的欄位名稱。
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    basemap.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.5)

    # 邊
    max_w = gdf_edges["weight"].max()
    if max_w > 0:
        gdf_edges.plot(
            ax=ax,
            linewidth=gdf_edges["weight"] / max_w * 3,
            color="steelblue",
            alpha=0.3
        )

    # 節點
    max_s = gdf_nodes[size_col].max()
    node_sizes = gdf_nodes[size_col] / max_s * 300 + 10 if max_s > 0 else 20

    gdf_nodes.plot(
        ax=ax,
        markersize=node_sizes,
        column=color_col,
        cmap="Set2",
        edgecolor="black",
        linewidth=0.3,
        legend=True,
        alpha=0.8
    )

    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_multi_period(basemap, period_data, size_col="betweenness",
                      color_col="community", save_path=None):
    """
    多時段並排比較圖。

    Parameters
    ----------
    period_data : dict
        key 為時段名稱，value 為 dict 含 gdf_nodes 與 gdf_edges。
    """
    n = len(period_data)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 16))
    if n == 1:
        axes = [axes]

    for ax, (period_name, data) in zip(axes, period_data.items()):
        basemap.plot(ax=ax, color="whitesmoke", edgecolor="gray", linewidth=0.5)

        gdf_e = data["gdf_edges"]
        gdf_n = data["gdf_nodes"]

        max_w = gdf_e["weight"].max()
        if max_w > 0:
            gdf_e.plot(ax=ax, linewidth=gdf_e["weight"]/max_w*3,
                       color="steelblue", alpha=0.3)

        max_s = gdf_n[size_col].max()
        sizes = gdf_n[size_col]/max_s*300+10 if max_s > 0 else 20
        gdf_n.plot(ax=ax, markersize=sizes, column=color_col,
                   cmap="Set2", edgecolor="black", linewidth=0.3, alpha=0.8)

        ax.set_title(period_name, fontsize=14)
        ax.set_axis_off()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_pyvis_network(G, centrality_dict, output_path="network.html"):
    """
    用 pyVis 建立互動式網絡視覺化。
    """
    from pyvis.network import Network

    net = Network(height="800px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="black")

    for node in G.nodes():
        c = centrality_dict.get(node, 0)
        net.add_node(
            node,
            size=c * 100 + 5,
            title=f"{node}\nCentrality: {c:.4f}"
        )

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1))

    net.show(output_path)
    return output_path
