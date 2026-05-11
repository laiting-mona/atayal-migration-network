"""
Layer A：泰雅族縣市時序遷徙網絡分析
2013H1 – 2022H2（共 20 個半年期）

資料邏輯：_Intact 檔 + EthnicityCate=2（泰雅族）
出生地(BirthPlaceCate) → 現居縣市(AdmiPrefCityCate) 作為終身遷徙存量 OD 流
PopnDynaStus=12 雖為縣際遷徙，但 Intact 檔僅有單一現居地欄位，無 T1 起源地，
因此全面採「出生地→現居地」多時點存量對比法重建時序。
"""
import csv
import os
import sys
import warnings

# force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

try:
    import community as community_louvain
    LOUVAIN_PKG = 'community'
except ImportError:
    community_louvain = None
    LOUVAIN_PKG = None

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_HALF = ROOT / 'data' / 'PopulationDynamcicsData_HalfYearPeriod'
OUT_TABLES = ROOT / 'outputs' / 'tables'
OUT_FIGS   = ROOT / 'outputs' / 'figures'
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ── 字型（無 CJK 字型時退回英文標籤）──────────────────────────────────────
for font in ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'Microsoft JhengHei',
             'PingFang TC', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = font
        break
    except Exception:
        pass
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 縣市代碼正規化 ──────────────────────────────────────────────────────────
# BirthPlaceCate 使用舊行政區代碼，統一合併至 22 縣市
BIRTH_NORM = {
    '10001': '65000',  # 新北（台北縣→新北市）
    '10003': '68000',  # 桃園（舊桃園縣→桃園市）
    '10006': '66000',  # 台中（台中縣→台中市）
    '10019': '66000',  # 台中（舊台中市→台中市）
    '10011': '67000',  # 台南（台灣縣→台南市）
    '10021': '67000',  # 台南（舊台南市→台南市）
    '10012': '64000',  # 高雄（高雄縣→高雄市）
    '64000': '64000',  # 高雄（舊高雄市→高雄市）
    '9007':  '09007',  # 連江
    '9020':  '09020',  # 金門
}
ADMI_NORM = {
    '10003': '68000',  # 桃園縣舊代碼
    '66001': '66000',  # 台中市舊直轄市
    '66002': '66000',  # 台中縣
    '67001': '67000',  # 台南市舊直轄市
    '67002': '67000',  # 台灣縣
    '64001': '64000',  # 高雄市舊直轄市
    '64002': '64000',  # 高雄縣
    '68000': '68000',  # 桃園市（已升格）
    '65000': '65000',  # 新北市
    '9007':  '09007',
    '9020':  '09020',
}

def norm_birth(c):
    c = c.strip()
    return BIRTH_NORM.get(c, c)

def norm_admi(c):
    c = c.strip()
    return ADMI_NORM.get(c, c)

# ── 縣市中文名 & 地理座標 ─────────────────────────────────────────────────
CITY_ZH = {
    '09007': '連江', '09020': '金門', '10002': '宜蘭',
    '10004': '新竹縣', '10005': '苗栗', '10007': '彰化',
    '10008': '南投', '10009': '雲林', '10010': '嘉義縣',
    '10013': '屏東', '10014': '臺東', '10015': '花蓮',
    '10016': '澎湖', '10017': '基隆', '10018': '新竹市',
    '10020': '嘉義市', '63000': '臺北', '64000': '高雄',
    '65000': '新北', '66000': '臺中', '67000': '臺南',
    '68000': '桃園',
}
CITY_EN = {
    '09007': 'Lienchiang', '09020': 'Kinmen', '10002': 'Yilan',
    '10004': 'Hsinchu Co.', '10005': 'Miaoli', '10007': 'Changhua',
    '10008': 'Nantou', '10009': 'Yunlin', '10010': 'Chiayi Co.',
    '10013': 'Pingtung', '10014': 'Taitung', '10015': 'Hualien',
    '10016': 'Penghu', '10017': 'Keelung', '10018': 'Hsinchu City',
    '10020': 'Chiayi City', '63000': 'Taipei', '64000': 'Kaohsiung',
    '65000': 'New Taipei', '66000': 'Taichung', '67000': 'Tainan',
    '68000': 'Taoyuan',
}
# 地理座標 (lon, lat)
COORD = {
    '10002': (121.75, 24.75), '10004': (121.10, 24.70), '10005': (120.85, 24.55),
    '10007': (120.55, 24.05), '10008': (120.85, 23.85), '10009': (120.50, 23.70),
    '10010': (120.55, 23.45), '10013': (120.55, 22.55), '10014': (121.10, 22.75),
    '10015': (121.50, 23.95), '10016': (119.55, 23.55), '10017': (121.74, 25.13),
    '10018': (120.95, 24.80), '10020': (120.45, 23.48), '63000': (121.55, 25.05),
    '64000': (120.30, 22.65), '65000': (121.50, 25.00), '66000': (120.65, 24.15),
    '67000': (120.20, 23.00), '68000': (121.30, 24.95),
    '09007': (119.95, 26.15), '09020': (118.30, 24.45),
}

def city_label(c):
    return CITY_ZH.get(c, CITY_EN.get(c, c))

# ── 半年期清單 ────────────────────────────────────────────────────────────
PERIODS = [
    ('2013H1','HalfYearPeriod_2013_1stHalf','TwnAbori_PopnDyna201303To201305_Intact.CSV'),
    ('2013H2','HalfYearPeriod_2013_2ndHalf','TwnAbori_PopnDyna201307To201311_Intact.CSV'),
    ('2014H1','HalfYearPeriod_2014_1stHalf','TwnAbori_PopnDyna201401To201406_Intact.CSV'),
    ('2014H2','HalfYearPeriod_2014_2ndHalf','TwnAbori_PopnDyna201408To201412_Intact.CSV'),
    ('2015H1','HalfYearPeriod_2015_1stHalf','TwnAbori_PopnDyna201502To201506_Intact.CSV'),
    ('2015H2','HalfYearPeriod_2015_2ndHalf','TwnAbori_PopnDyna201507To201512_Intact.CSV'),
    ('2016H1','HalfYearPeriod_2016_1stHalf','TwnAbori_PopnDyna201601To201606_Intact.CSV'),
    ('2016H2','HalfYearPeriod_2016_2ndHalf','TwnAbori_PopnDyna201607To201612_Intact.CSV'),
    ('2017H1','HalfYearPeriod_2017_1stHalf','TwnAbori_PopnDyna201701To201706_Intact.CSV'),
    ('2017H2','HalfYearPeriod_2017_2ndHalf','TwnAbori_PopnDyna201707To201712_Intact.CSV'),
    ('2018H1','HalfYearPeriod_2018_1stHalf','TwnAbori_PopnDyna201801To201806_Intact.CSV'),
    ('2018H2','HalfYearPeriod_2018_2ndHalf','TwnAbori_PopnDyna201807To201812_Intact.CSV'),
    ('2019H1','HalfYearPeriod_2019_1stHalf','TwnAbori_PopnDyna201901To201906_Intact.CSV'),
    ('2019H2','HalfYearPeriod_2019_2ndHalf','TwnAbori_PopnDyna201907To201912_Intact.CSV'),
    ('2020H1','HalfYearPeriod_2020_1stHalf','TwnAbori_PopnDyna202001To202006_Intact.CSV'),
    ('2020H2','HalfYearPeriod_2020_2ndHalf','TwnAbori_PopnDyna202007To202012_Intact.CSV'),
    ('2021H1','HalfYearPeriod_2021_1stHalf','TwnAbori_PopnDyna202101To202106_Intact.CSV'),
    ('2021H2','HalfYearPeriod_2021_2ndHalf','TwnAbori_PopnDyna202107To202112_Intact.CSV'),
    ('2022H1','HalfYearPeriod_2022_1stHalf','TwnAbori_PopnDyna202201To202206_Intact.CSV'),
    ('2022H2','HalfYearPeriod_2022_2ndHalf','TwnAbori_PopnDyna202207To202212_Intact.CSV'),
]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1  讀取資料、建立 OD 矩陣
# ══════════════════════════════════════════════════════════════════════════
def load_period(folder_name, csv_name):
    """載入單一時段 CSV → 回傳 (od_counter, node_pop)
    od_counter: Counter {(origin_code, dest_code): weight}  (僅跨縣市)
    node_pop:   Counter {county_code: weight}               (所有縣市人口)
    """
    path = DATA_HALF / folder_name / 'PopnDynamics_Intact' / csv_name
    od = Counter()
    pop = Counter()
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('EthnicityCate', '').strip() != '2':
                continue
            birth = norm_birth(row.get('BirthPlaceCate', '').strip())
            dest  = norm_admi(row.get('AdmiPrefCityCate', '').strip())
            w     = int(row.get('Weight', '1'))
            if not birth or birth == '0' or not dest or dest == '0':
                continue
            pop[dest] += w
            if birth != dest:
                od[(birth, dest)] += w
    return od, pop


def build_graph(od_counter, node_pop):
    """OD Counter → NetworkX DiGraph（有向加權）"""
    G = nx.DiGraph()
    # 先加所有節點（含無邊縣市）
    all_nodes = set(node_pop.keys()) | {n for pair in od_counter for n in pair}
    for n in all_nodes:
        G.add_node(n, pop=node_pop.get(n, 0), label=city_label(n))
    # 加邊
    for (o, d), w in od_counter.items():
        G.add_edge(o, d, weight=w)
    return G


# ══════════════════════════════════════════════════════════════════════════
# STEP 2  計算網絡指標
# ══════════════════════════════════════════════════════════════════════════
def freeman_centralization(cent_dict):
    """Freeman 圖中心化 = sum(max_c - c_i) / ((n-1) * max_c)"""
    vals = np.array(list(cent_dict.values()), dtype=float)
    n = len(vals)
    if n <= 1:
        return 0.0
    max_c = vals.max()
    if max_c == 0:
        return 0.0
    return float((max_c - vals).sum() / ((n - 1) * max_c))


def compute_network_metrics(G, n_null=100, rng_seed=42):
    """計算有向加權網絡的完整指標集"""
    rng = np.random.RandomState(rng_seed)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    # ── in/out degree（加權）──
    in_deg  = dict(G.in_degree(weight='weight'))
    out_deg = dict(G.out_degree(weight='weight'))

    # ── 轉無向（for Louvain, clustering, betweenness on undirected）──
    G_un = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d['weight']
        if G_un.has_edge(u, v):
            G_un[u][v]['weight'] += w
        elif G_un.has_edge(v, u):
            G_un[v][u]['weight'] += w
        else:
            G_un.add_edge(u, v, weight=w)

    # ── betweenness（使用 1/weight 作距離，有向圖）──
    G_dist = G.copy()
    for u, v, d in G_dist.edges(data=True):
        d['dist'] = 1.0 / d['weight'] if d['weight'] > 0 else 1e6
    betw = nx.betweenness_centrality(G_dist, weight='dist', normalized=True)

    # ── eigenvector centrality（有向圖，以 numpy 求解避免不收斂）──
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception:
        eig = {n: 0.0 for n in G.nodes()}

    # ── clustering coefficient（有向加權）──
    # networkx directed clustering = geometric mean of cycle fractions
    try:
        clust = nx.clustering(G, weight='weight')
    except Exception:
        clust = {n: 0.0 for n in G.nodes()}
    avg_clust = np.mean(list(clust.values())) if clust else 0.0

    # ── Freeman centralization ──
    # in-degree: 吸納集中度；out-degree: 輸出集中度
    in_deg_norm  = nx.in_degree_centrality(G)
    out_deg_norm = nx.out_degree_centrality(G)
    betw_norm    = betw  # already normalized
    eig_norm     = eig

    freeman_in   = freeman_centralization(in_deg_norm)
    freeman_out  = freeman_centralization(out_deg_norm)
    freeman_betw = freeman_centralization(betw_norm)

    # ── Louvain community detection（無向圖）──
    partition = {}
    modularity = float('nan')
    n_comm = 0
    if community_louvain is not None and G_un.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(G_un, weight='weight',
                                                          random_state=rng_seed)
            modularity = community_louvain.modularity(partition, G_un, weight='weight')
            n_comm = len(set(partition.values()))
        except Exception as e:
            print(f"    Louvain error: {e}")
    else:
        # fallback: networkx greedy modularity
        try:
            comms = list(nx.community.greedy_modularity_communities(G_un, weight='weight'))
            partition = {}
            for i, comm in enumerate(comms):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(G_un, comms, weight='weight')
            n_comm = len(comms)
        except Exception:
            pass

    # ── Null model：配置模型隨機化（保持度序列）──
    null_mods = []
    if G_un.number_of_edges() > 0:
        deg_seq = [d for _, d in G_un.degree()]
        for _ in range(n_null):
            try:
                Gn = nx.configuration_model(deg_seq, seed=int(rng.randint(10**7)))
                Gn = nx.Graph(Gn)
                Gn.remove_edges_from(nx.selfloop_edges(Gn))
                if Gn.number_of_edges() > 0 and community_louvain:
                    part_n = community_louvain.best_partition(Gn, random_state=42)
                    null_mods.append(community_louvain.modularity(part_n, Gn))
                elif Gn.number_of_edges() > 0:
                    comms_n = list(nx.community.greedy_modularity_communities(Gn))
                    null_mods.append(nx.community.modularity(Gn, comms_n))
            except Exception:
                continue

    def z_score(obs, nulls):
        if len(nulls) < 10 or np.isnan(obs):
            return float('nan')
        return float((obs - np.mean(nulls)) / (np.std(nulls) + 1e-9))

    z_mod = z_score(modularity, null_mods)

    return {
        # graph-level
        'n_nodes': n, 'n_edges': m, 'density': density,
        'avg_clustering': avg_clust,
        'modularity': modularity, 'n_communities': n_comm,
        'z_modularity': z_mod,
        'freeman_indegree': freeman_in,
        'freeman_outdegree': freeman_out,
        'freeman_betweenness': freeman_betw,
        # node-level dicts
        '_in_deg': in_deg, '_out_deg': out_deg,
        '_betw': betw, '_eig': eig, '_clust': clust,
        '_partition': partition,
        '_in_deg_norm': in_deg_norm, '_out_deg_norm': out_deg_norm,
    }


# ══════════════════════════════════════════════════════════════════════════
# STEP 3  逐時段跑分析
# ══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Layer A 泰雅族縣市時序遷徙網絡分析（2013H1–2022H2）")
print("=" * 65)

all_results = {}   # period_id → {metrics + G}
all_graphs   = {}  # period_id → G

for pid, folder, csvname in PERIODS:
    print(f"\n▶ {pid} …", end=' ', flush=True)
    od, pop = load_period(folder, csvname)
    G = build_graph(od, pop)
    total_pop    = sum(pop.values())
    total_movers = sum(od.values())
    print(f"總人口={total_pop:,}  跨縣遷徙流={total_movers:,}  "
          f"節點={G.number_of_nodes()}  邊={G.number_of_edges()}", flush=True)

    metrics = compute_network_metrics(G, n_null=100)
    metrics['period'] = pid
    metrics['total_pop'] = total_pop
    metrics['total_movers'] = total_movers
    metrics['_G'] = G
    all_results[pid] = metrics
    all_graphs[pid]  = G
    print(f"  Q={metrics['modularity']:.4f}  z={metrics['z_modularity']:+.1f}  "
          f"comms={metrics['n_communities']}  "
          f"freeman_in={metrics['freeman_indegree']:.3f}", flush=True)

print("\n✓ 所有時段計算完畢")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4  輸出 CSV 表格
# ══════════════════════════════════════════════════════════════════════════
# 4a. 各時段節點中心性表
print("\n輸出節點中心性表…")
for pid, res in all_results.items():
    G = res['_G']
    nodes = list(G.nodes())
    rows = []
    for nd in nodes:
        rows.append({
            'period': pid,
            'county_code': nd,
            'county_name': city_label(nd),
            'population': G.nodes[nd].get('pop', 0),
            'in_degree_w': res['_in_deg'].get(nd, 0),
            'out_degree_w': res['_out_deg'].get(nd, 0),
            'in_degree_norm': round(res['_in_deg_norm'].get(nd, 0), 6),
            'out_degree_norm': round(res['_out_deg_norm'].get(nd, 0), 6),
            'betweenness': round(res['_betw'].get(nd, 0), 6),
            'eigenvector': round(res['_eig'].get(nd, 0), 6),
            'clustering': round(res['_clust'].get(nd, 0), 6),
            'community': res['_partition'].get(nd, -1),
        })
    df = pd.DataFrame(rows).sort_values('in_degree_w', ascending=False)
    df.to_csv(OUT_TABLES / f'layer_a_{pid}_centrality.csv', index=False, encoding='utf-8-sig')

# 4b. 跨時段彙整表
print("輸出跨時段彙整表…")
summary_rows = []
for pid, res in all_results.items():
    summary_rows.append({
        'period': pid,
        'total_pop': res['total_pop'],
        'total_movers': res['total_movers'],
        'migration_rate': round(res['total_movers'] / res['total_pop'], 4) if res['total_pop'] else 0,
        'n_nodes': res['n_nodes'],
        'n_edges': res['n_edges'],
        'density': round(res['density'], 4),
        'avg_clustering': round(res['avg_clustering'], 4),
        'modularity_Q': round(res['modularity'], 4) if not np.isnan(res['modularity']) else None,
        'n_communities': res['n_communities'],
        'z_modularity': round(res['z_modularity'], 2) if not np.isnan(res['z_modularity']) else None,
        'freeman_indegree': round(res['freeman_indegree'], 4),
        'freeman_outdegree': round(res['freeman_outdegree'], 4),
        'freeman_betweenness': round(res['freeman_betweenness'], 4),
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_TABLES / 'layer_a_metrics_summary.csv', index=False, encoding='utf-8-sig')
print("  已存：layer_a_metrics_summary.csv")
print(df_summary[['period','total_pop','density','modularity_Q','n_communities',
                   'freeman_indegree','z_modularity']].to_string(index=False))

# 4c. 各時段 Top-5 節點快速總覽
print("\n各時段 Top-5 吸納樞紐（in-degree）：")
for pid, res in all_results.items():
    top5 = sorted(res['_in_deg'].items(), key=lambda x: -x[1])[:5]
    top5_str = '  '.join(f"{city_label(n)}({int(w):,})" for n,w in top5)
    print(f"  {pid}: {top5_str}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5  視覺化
# ══════════════════════════════════════════════════════════════════════════
PERIOD_IDS = [pid for pid, _, _ in PERIODS]
CMAP_COMM = plt.cm.Set2

def draw_network_on_map(ax, G, res, title, show_labels=True, edge_quantile=0.5):
    """在地理座標上畫有向加權網絡，節點顏色=社群，大小=in-degree"""
    pos = {n: COORD[n] for n in G.nodes() if n in COORD}
    if not pos:
        ax.set_title(title, fontsize=9); ax.axis('off'); return

    G_vis = G.subgraph(pos.keys()).copy()
    partition = res.get('_partition', {})
    comm_ids = sorted(set(partition.values())) if partition else []
    comm_colors = {c: CMAP_COMM(i / max(len(comm_ids), 1))
                   for i, c in enumerate(comm_ids)}

    # edge filtering: top (1-edge_quantile) edges
    weights = [d['weight'] for _, _, d in G_vis.edges(data=True)]
    if weights:
        threshold = np.quantile(weights, edge_quantile)
        edges_to_draw = [(u, v, d) for u, v, d in G_vis.edges(data=True)
                         if d['weight'] >= threshold]
    else:
        edges_to_draw = []

    max_w = max((d['weight'] for _, _, d in edges_to_draw), default=1)
    for u, v, d in edges_to_draw:
        if u not in pos or v not in pos:
            continue
        lw = 0.3 + 3.5 * d['weight'] / max_w
        alpha = 0.25 + 0.55 * d['weight'] / max_w
        ax.annotate(
            '', xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle='->', lw=lw, color='#555555',
                            alpha=alpha, shrinkA=7, shrinkB=7,
                            mutation_scale=10),
            zorder=2
        )

    in_deg = res['_in_deg']
    max_in = max(in_deg.values(), default=1)
    for nd in G_vis.nodes():
        if nd not in pos:
            continue
        x, y = pos[nd]
        size = 30 + 200 * in_deg.get(nd, 0) / (max_in + 1)
        c_id = partition.get(nd, -1)
        color = comm_colors.get(c_id, '#cccccc')
        ax.scatter(x, y, s=size, c=[color], edgecolors='white',
                   linewidths=0.8, zorder=3, alpha=0.9)
        if show_labels and in_deg.get(nd, 0) > max_in * 0.05:
            ax.annotate(city_label(nd), (x, y), fontsize=6.5,
                        ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points',
                        fontweight='bold', zorder=4,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white',
                                  ec='none', alpha=0.7))
    ax.set_xlim(118, 122.8); ax.set_ylim(21.8, 26.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=8.5, fontweight='bold', pad=3)
    # 簡短統計
    Q = res.get('modularity', float('nan'))
    z = res.get('z_modularity', float('nan'))
    ax.text(0.02, 0.02,
            f"Q={Q:.3f} z={z:+.1f}\ncomm={res['n_communities']}",
            transform=ax.transAxes, fontsize=6.5, va='bottom',
            bbox=dict(fc='white', ec='none', alpha=0.8))


# ────────────────────────────────────────────────────────────────────────
# Fig 1：20 時段網絡地圖（4行×5列）
# ────────────────────────────────────────────────────────────────────────
print("\n繪製 Fig 1：20 時段網絡地圖…")
fig1, axes1 = plt.subplots(4, 5, figsize=(22, 18))
fig1.patch.set_facecolor('#f5f5f5')
for ax_row in axes1:
    for ax in ax_row:
        ax.set_facecolor('#e8f4f8')

for i, pid in enumerate(PERIOD_IDS):
    ax = axes1[i // 5][i % 5]
    draw_network_on_map(ax, all_graphs[pid], all_results[pid],
                        title=pid, show_labels=(i == 0 or i == 19))

fig1.suptitle(
    '泰雅族縣市終身遷徙網絡：2013H1–2022H2 全時段（Layer A）\n'
    '節點大小=加權 in-degree（吸納量）；顏色=Louvain 社群；箭頭粗細=流量',
    fontsize=13, fontweight='bold', y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(OUT_FIGS / 'layer_a_network_grid.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  → layer_a_network_grid.png")


# ────────────────────────────────────────────────────────────────────────
# Fig 2：各時段指標時序折線圖
# ────────────────────────────────────────────────────────────────────────
print("繪製 Fig 2：指標時序圖…")
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 13))
fig2.patch.set_facecolor('#fafafa')
xs = np.arange(len(PERIOD_IDS))
xt = PERIOD_IDS
xt_short = [p if p.endswith('H1') else '' for p in xt]

def ts_plot(ax, key, label, ylabel, color='steelblue', ref_line=None):
    vals = [all_results[p][key] for p in PERIOD_IDS]
    vals_f = [v if not (isinstance(v, float) and np.isnan(v)) else None for v in vals]
    ax.plot(xs, vals, 'o-', color=color, lw=2, ms=5, alpha=0.85)
    ax.set_xticks(xs[::2]); ax.set_xticklabels(xt[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ref_line is not None:
        ax.axhline(ref_line, color='red', ls='--', alpha=0.5, lw=1)
    # annotate first and last
    for idx in [0, -1]:
        v = vals[idx]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ax.annotate(f'{v:.3f}', (xs[idx], v),
                        textcoords='offset points', xytext=(4, 4), fontsize=7.5)

def ts_plot2(ax, vals_list, title, ylabel, color='steelblue', ref_line=None):
    ax.plot(xs, vals_list, 'o-', color=color, lw=2, ms=5, alpha=0.85)
    ax.set_xticks(xs[::2]); ax.set_xticklabels(xt[::2], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ref_line is not None:
        ax.axhline(ref_line, color='red', ls='--', alpha=0.5, lw=1)
    for idx in [0, -1]:
        v = vals_list[idx]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ax.annotate(f'{v:.3f}', (xs[idx], v),
                        textcoords='offset points', xytext=(4, 4), fontsize=7.5)

ts_plot2(axes2[0,0], df_summary['density'].tolist(),
         '① 網絡密度 (Density)', 'density', '#1565c0')
ts_plot2(axes2[0,1], df_summary['modularity_Q'].tolist(),
         '② Louvain Modularity Q', 'Q', '#6a1fa2')
ts_plot2(axes2[1,0], df_summary['freeman_indegree'].tolist(),
         '③ Freeman 中心化（in-degree）', 'C_D', '#c62828')
ts_plot2(axes2[1,1], df_summary['freeman_betweenness'].tolist(),
         '④ Freeman 中心化（betweenness）', 'C_B', '#e65100')
ts_plot2(axes2[2,0], df_summary['avg_clustering'].tolist(),
         '⑤ 平均聚群係數', 'avg C', '#2e7d32')

# z-score plot
z_vals = df_summary['z_modularity'].tolist()
axes2[2,1].plot(xs, z_vals, 'o-', color='#37474f', lw=2, ms=5, alpha=0.85)
axes2[2,1].axhline(1.96, color='red', ls='--', lw=1, alpha=0.6, label='p=0.05 (z=1.96)')
axes2[2,1].axhline(-1.96, color='blue', ls='--', lw=1, alpha=0.6)
axes2[2,1].set_xticks(xs[::2]); axes2[2,1].set_xticklabels(xt[::2], rotation=45, ha='right', fontsize=8)
axes2[2,1].set_ylabel('z-score', fontsize=9)
axes2[2,1].set_title('⑥ Modularity vs Null z-score\n（±1.96 = p<0.05）', fontsize=11, fontweight='bold')
axes2[2,1].legend(fontsize=8); axes2[2,1].grid(axis='y', alpha=0.3)
axes2[2,1].spines['top'].set_visible(False); axes2[2,1].spines['right'].set_visible(False)

fig2.suptitle('泰雅族縣市遷徙網絡關鍵指標時序（2013H1–2022H2）',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig2.savefig(OUT_FIGS / 'layer_a_metrics_timeseries.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  → layer_a_metrics_timeseries.png")


# ────────────────────────────────────────────────────────────────────────
# Fig 3：中心性熱圖（縣市 × 時段）
# ────────────────────────────────────────────────────────────────────────
print("繪製 Fig 3：中心性熱圖…")
# 取所有時段都出現的主要縣市（共 22 縣市最多）
all_nodes_set = set()
for G in all_graphs.values():
    all_nodes_set.update(G.nodes())
main_nodes = sorted(all_nodes_set, key=lambda n: -sum(
    all_results[pid]['_in_deg'].get(n, 0) for pid in PERIOD_IDS
))[:20]  # top-20 by total in-degree

metrics_to_plot = [
    ('_in_deg_norm',  'In-degree\n(norm)'),
    ('_out_deg_norm', 'Out-degree\n(norm)'),
    ('_betw',         'Betweenness\nCentrality'),
    ('_eig',          'Eigenvector\nCentrality'),
    ('_clust',        'Clustering\nCoefficient'),
]

fig3, axes3 = plt.subplots(1, 5, figsize=(26, 10))
fig3.patch.set_facecolor('#fafafa')
node_labels = [city_label(n) for n in main_nodes]

for ax, (key, title) in zip(axes3, metrics_to_plot):
    mat = np.zeros((len(main_nodes), len(PERIOD_IDS)))
    for j, pid in enumerate(PERIOD_IDS):
        for i, nd in enumerate(main_nodes):
            mat[i, j] = all_results[pid][key].get(nd, 0)
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(main_nodes)))
    ax.set_yticklabels(node_labels, fontsize=8.5)
    ax.set_xticks(range(0, len(PERIOD_IDS), 2))
    ax.set_xticklabels(PERIOD_IDS[::2], rotation=45, ha='right', fontsize=7.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

fig3.suptitle('泰雅族縣市中心性熱圖（Top-20 縣市 × 20 時段）',
              fontsize=12, fontweight='bold')
plt.tight_layout()
fig3.savefig(OUT_FIGS / 'layer_a_centrality_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  → layer_a_centrality_heatmap.png")


# ────────────────────────────────────────────────────────────────────────
# Fig 4：Louvain 社群偵測──選 5 個代表時段的地圖
# ────────────────────────────────────────────────────────────────────────
print("繪製 Fig 4：社群偵測地圖（5 代表時段）…")
rep_periods = ['2013H1', '2015H1', '2017H1', '2019H1', '2022H2']
fig4, axes4 = plt.subplots(1, 5, figsize=(25, 6))
fig4.patch.set_facecolor('#f0f4f8')
for ax in axes4:
    ax.set_facecolor('#ddeeff')

for ax, pid in zip(axes4, rep_periods):
    res = all_results[pid]
    G   = all_graphs[pid]
    draw_network_on_map(ax, G, res, title=pid, show_labels=True, edge_quantile=0.4)

fig4.suptitle(
    '泰雅族縣市遷徙網絡 Louvain 社群偵測空間視覺化（5 代表時段）\n'
    '節點顏色=社群；大小=加權 in-degree；箭頭=主要遷徙流（Top 60%）',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
fig4.savefig(OUT_FIGS / 'layer_a_community_maps.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  → layer_a_community_maps.png")


# ────────────────────────────────────────────────────────────────────────
# Fig 5：各時段 net flow 排名（in - out）前後5名
# ────────────────────────────────────────────────────────────────────────
print("繪製 Fig 5：淨流量時序…")
# 選泰雅族主要縣市
key_counties = ['65000','63000','68000','10002','10004','10005','10008','10015','10014']
kc_labels    = [city_label(c) for c in key_counties]

fig5, axes5 = plt.subplots(3, 1, figsize=(16, 14))
fig5.patch.set_facecolor('#fafafa')

# (a) Net flow over time heatmap
net_mat = np.zeros((len(key_counties), len(PERIOD_IDS)))
for j, pid in enumerate(PERIOD_IDS):
    res = all_results[pid]
    for i, nd in enumerate(key_counties):
        in_w  = res['_in_deg'].get(nd, 0)
        out_w = res['_out_deg'].get(nd, 0)
        net_mat[i, j] = in_w - out_w

vmax = np.abs(net_mat).max()
im5 = axes5[0].imshow(net_mat, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
axes5[0].set_yticks(range(len(key_counties)))
axes5[0].set_yticklabels(kc_labels, fontsize=9)
axes5[0].set_xticks(range(len(PERIOD_IDS)))
axes5[0].set_xticklabels(PERIOD_IDS, rotation=45, ha='right', fontsize=7.5)
axes5[0].set_title('① 淨遷徙流（In-weight − Out-weight）\n紅=淨吸納；藍=淨輸出',
                    fontsize=11, fontweight='bold')
plt.colorbar(im5, ax=axes5[0], shrink=0.7)
for i in range(net_mat.shape[0]):
    for j in range(net_mat.shape[1]):
        v = int(net_mat[i,j])
        tc = 'white' if abs(net_mat[i,j]) > vmax*0.55 else 'black'
        axes5[0].text(j, i, f'{v:+d}', ha='center', va='center', fontsize=6, color=tc)

# (b) In-degree trend for key counties
colors5 = plt.cm.tab10(np.linspace(0, 1, len(key_counties)))
for i, (nd, lab) in enumerate(zip(key_counties, kc_labels)):
    vals = [all_results[pid]['_in_deg'].get(nd, 0) for pid in PERIOD_IDS]
    axes5[1].plot(xs, vals, 'o-', label=lab, color=colors5[i], lw=1.8, ms=4)
axes5[1].set_xticks(xs[::2]); axes5[1].set_xticklabels(PERIOD_IDS[::2], rotation=45, ha='right', fontsize=8)
axes5[1].set_ylabel('加權 in-degree（遷入存量）', fontsize=9)
axes5[1].set_title('② 主要縣市遷入存量時序', fontsize=11, fontweight='bold')
axes5[1].legend(fontsize=8, ncol=3, loc='upper left'); axes5[1].grid(alpha=0.3)
axes5[1].spines['top'].set_visible(False); axes5[1].spines['right'].set_visible(False)

# (c) Out-degree trend
for i, (nd, lab) in enumerate(zip(key_counties, kc_labels)):
    vals = [all_results[pid]['_out_deg'].get(nd, 0) for pid in PERIOD_IDS]
    axes5[2].plot(xs, vals, 's--', label=lab, color=colors5[i], lw=1.5, ms=4, alpha=0.85)
axes5[2].set_xticks(xs[::2]); axes5[2].set_xticklabels(PERIOD_IDS[::2], rotation=45, ha='right', fontsize=8)
axes5[2].set_ylabel('加權 out-degree（遷出存量）', fontsize=9)
axes5[2].set_title('③ 主要縣市遷出存量時序', fontsize=11, fontweight='bold')
axes5[2].legend(fontsize=8, ncol=3, loc='upper left'); axes5[2].grid(alpha=0.3)
axes5[2].spines['top'].set_visible(False); axes5[2].spines['right'].set_visible(False)

fig5.suptitle('泰雅族縣市遷徙網絡 淨流量與 in/out-degree 時序',
              fontsize=13, fontweight='bold')
plt.tight_layout()
fig5.savefig(OUT_FIGS / 'layer_a_netflow_timeseries.png', dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  → layer_a_netflow_timeseries.png")


# ────────────────────────────────────────────────────────────────────────
# Fig 6：中心性分布圖（以最後一期 2022H2 為範例）
# ────────────────────────────────────────────────────────────────────────
print("繪製 Fig 6：2022H2 中心性分布圖…")
pid_eg = '2022H2'
res_eg = all_results[pid_eg]
G_eg   = all_graphs[pid_eg]
nodes_eg = list(G_eg.nodes())
labels_eg = [city_label(n) for n in nodes_eg]

fig6, axes6 = plt.subplots(2, 2, figsize=(14, 11))
fig6.patch.set_facecolor('#fafafa')

metrics_6 = [
    (res_eg['_in_deg_norm'],  'In-degree Centrality (norm)', '#1565c0'),
    (res_eg['_out_deg_norm'], 'Out-degree Centrality (norm)', '#b71c1c'),
    (res_eg['_betw'],         'Betweenness Centrality',       '#2e7d32'),
    (res_eg['_eig'],          'Eigenvector Centrality',       '#6a1fa2'),
]
for ax, (cent_dict, title, color) in zip(axes6.flat, metrics_6):
    vals  = [cent_dict.get(n, 0) for n in nodes_eg]
    lbls  = labels_eg
    order = np.argsort(vals)[::-1]
    sorted_vals  = [vals[i] for i in order]
    sorted_lbls  = [lbls[i] for i in order]
    bars = ax.barh(range(len(sorted_lbls)), sorted_vals, color=color, alpha=0.8)
    ax.set_yticks(range(len(sorted_lbls)))
    ax.set_yticklabels(sorted_lbls, fontsize=9)
    ax.set_xlabel(title.split('(')[0].strip(), fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, v in zip(bars, sorted_vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=7.5)

fig6.suptitle(f'泰雅族縣市遷徙網絡中心性分布（{pid_eg}）',
              fontsize=12, fontweight='bold')
plt.tight_layout()
fig6.savefig(OUT_FIGS / 'layer_a_centrality_distribution_2022H2.png', dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  → layer_a_centrality_distribution_2022H2.png")


# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("✅ Layer A 分析完成")
print(f"   表格輸出：{OUT_TABLES}")
print(f"   圖片輸出：{OUT_FIGS}")
print("=" * 65)
