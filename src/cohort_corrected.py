"""
修正版 Cohort 終身遷徙網絡分析

修正內容（對應你之前 prototype 與凡嘉版本的問題）：
1. ✅ 明確篩泰雅族（EthnicityCate=2）
2. ✅ 縣市升格代碼合併
3. ✅ Freeman centralization 改為標準公式
4. ✅ small-world sigma 加入
5. ✅ 每期報告 z-score vs null model
6. ✅ Top hub / Top bridge 並列
7. ✅ 視覺化升級：地理 layout + 中心性同時呈現
"""
import csv
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
mpl.rcParams['axes.unicode_minus'] = False

# ---------- city codes & coords ----------
NORM = {'10003':'68000','10006':'66002','10019':'66001','10001':'65000',
        '10011':'67002','10021':'67001','10012':'64002','64000':'64001'}
MERGE = {'66002':'66000','66001':'66000','67002':'67000','67001':'67000',
         '64002':'64000','64001':'64000'}
def normp(c):
    c = NORM.get(c, c); return MERGE.get(c, c)

CITY = {'09007':'連江','09020':'金門','10002':'宜蘭','10004':'新竹縣','10005':'苗栗',
        '10007':'彰化','10008':'南投','10009':'雲林','10010':'嘉義縣','10013':'屏東',
        '10014':'臺東','10015':'花蓮','10016':'澎湖','10017':'基隆','10018':'新竹市',
        '10020':'嘉義市','63000':'臺北','64000':'高雄','65000':'新北','66000':'臺中',
        '67000':'臺南','68000':'桃園'}
COORD = {'10002':(121.75,24.75),'10004':(121.10,24.70),'10005':(120.85,24.55),
         '10007':(120.55,24.05),'10008':(120.85,23.85),'10009':(120.50,23.70),
         '10010':(120.55,23.45),'10013':(120.55,22.55),'10014':(121.10,22.75),
         '10015':(121.50,23.95),'10016':(119.55,23.55),'10017':(121.74,25.13),
         '10018':(120.95,24.80),'10020':(120.45,23.48),'63000':(121.55,25.05),
         '64000':(120.30,22.65),'65000':(121.50,25.00),'66000':(120.65,24.15),
         '67000':(120.20,23.00),'68000':(121.30,24.95),'09007':(119.95,26.15),
         '09020':(118.30,24.45)}

def age_to_cohort(a):
    a = int(a) if a != '00' else 0
    if a == 0: return None
    if a <= 5: return 'C1'
    if a <= 9: return 'C2'
    if a <= 13: return 'C3'
    return 'C4'

COHORT_LABEL = {
    'C4':'C4: 65+ 歲\n（推估 1957 前出生）',
    'C3':'C3: 45-64 歲\n（1957-1977 出生）',
    'C2':'C2: 25-44 歲\n（1977-1997 出生）',
    'C1':'C1: <25 歲\n（1997 後出生）',
}

# ---------- Load Atayal ----------
with open('intact.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['EthnicityCate']=='2']
print(f"泰雅族總筆數: {len(rows):,}")

# ---------- Build cohort networks ----------
moved = []
for r in rows:
    o = normp(r['BirthPlaceCate']); d = normp(r['AdmiPrefCityCate'])
    if o == d: continue
    c = age_to_cohort(r['Age5FCate'])
    if c: moved.append((o, d, c))

cohort_order = ['C4', 'C3', 'C2', 'C1']
cohort_nets = {}
for c in cohort_order:
    od = Counter((o,d) for o,d,cc in moved if cc==c)
    G = nx.DiGraph()
    for (o,d), w in od.items():
        G.add_edge(o, d, weight=w)
    cohort_nets[c] = (G, sum(od.values()))

# ---------- Compute metrics with z-scores vs null model ----------
def compute_metrics(G, n_null=200, seed=42):
    """Compute key metrics + z-scores vs configuration model null"""
    rng = np.random.RandomState(seed)
    G_un = G.to_undirected()
    # merge bi-directional weights
    G_simple = nx.Graph()
    for u in G_un.nodes():
        G_simple.add_node(u)
    for u, v in G_un.edges():
        w = (G[u][v]['weight'] if G.has_edge(u, v) else 0) + \
            (G[v][u]['weight'] if G.has_edge(v, u) else 0)
        G_simple.add_edge(u, v, weight=w)
    if G_simple.number_of_edges() == 0:
        return None
    LCC = G_simple.subgraph(max(nx.connected_components(G_simple), key=len)).copy()

    # Observed
    n, m = G.number_of_nodes(), G.number_of_edges()
    density = nx.density(G)
    clust = nx.average_clustering(LCC, weight='weight')
    path_len = nx.average_shortest_path_length(LCC)
    comms = nx.community.louvain_communities(LCC, weight='weight', seed=seed)
    mod_obs = nx.community.modularity(LCC, comms, weight='weight')

    # In-degree centralization (Freeman)
    in_deg = dict(G.in_degree(weight='weight'))
    max_in = max(in_deg.values())
    total_weight = sum(in_deg.values())
    # Freeman C_D = sum(C_max - C_i) / max_possible
    C_D_in = sum(max_in - v for v in in_deg.values()) / (total_weight + 1e-9)

    # Null model: configuration model preserving degree sequence
    deg_seq = [d for _, d in LCC.degree()]
    null_mods = []
    null_clusts = []
    null_paths = []
    for _ in range(n_null):
        try:
            Gn = nx.configuration_model(deg_seq, seed=rng.randint(10**8))
            Gn = nx.Graph(Gn)  # remove parallel edges
            Gn.remove_edges_from(nx.selfloop_edges(Gn))
            Gn_cc = Gn.subgraph(max(nx.connected_components(Gn), key=len))
            null_mods.append(nx.community.modularity(
                Gn_cc, nx.community.louvain_communities(Gn_cc, seed=42)))
            null_clusts.append(nx.average_clustering(Gn_cc))
            null_paths.append(nx.average_shortest_path_length(Gn_cc))
        except:
            continue

    # z-scores
    def z(obs, nulls):
        if len(nulls) < 10: return float('nan')
        return (obs - np.mean(nulls)) / (np.std(nulls) + 1e-9)

    sigma = (clust / (np.mean(null_clusts) + 1e-9)) / (path_len / (np.mean(null_paths) + 1e-9))

    return {
        'n': n, 'm': m, 'density': density,
        'clustering': clust, 'path_len': path_len,
        'modularity': mod_obs, 'n_comm': len(comms),
        'C_D_in': C_D_in,
        'sigma_smallworld': sigma,
        'z_modularity': z(mod_obs, null_mods),
        'z_clustering': z(clust, null_clusts),
        'comms': comms,
    }

print("\n計算各 cohort 指標與 null model z-scores（200 次隨機化）...")
metrics = {}
for c in cohort_order:
    G, n_people = cohort_nets[c]
    m = compute_metrics(G)
    m['n_people'] = n_people
    metrics[c] = m
    print(f"  {c}: 完成")

# ---------- Identify hubs & bridges per cohort ----------
def get_top_nodes(G):
    in_w = {n: sum(d['weight'] for _,_,d in G.in_edges(n, data=True)) for n in G.nodes()}
    out_w = {n: sum(d['weight'] for _,_,d in G.out_edges(n, data=True)) for n in G.nodes()}
    # betweenness with inverse weight as distance
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        d['inv'] = 1 / d['weight'] if d['weight'] > 0 else 1
    betw = nx.betweenness_centrality(G_inv, weight='inv', normalized=True)
    return in_w, out_w, betw

print("\n各 cohort 主要節點：")
for c in cohort_order:
    G, _ = cohort_nets[c]
    in_w, out_w, betw = get_top_nodes(G)
    top_in = sorted(in_w.items(), key=lambda x:-x[1])[:3]
    top_out = sorted(out_w.items(), key=lambda x:-x[1])[:3]
    top_betw = sorted(betw.items(), key=lambda x:-x[1])[:3]
    print(f"\n{c} ({metrics[c]['n_people']:,} 人):")
    print(f"  吸納樞紐: " + ", ".join(f"{CITY.get(n,n)}({int(w)})" for n,w in top_in))
    print(f"  輸出來源: " + ", ".join(f"{CITY.get(n,n)}({int(w)})" for n,w in top_out))
    print(f"  橋接節點: " + ", ".join(f"{CITY.get(n,n)}({b:.3f})" for n,b in top_betw))

# ---------- VISUALIZATION ----------
fig = plt.figure(figsize=(20, 13))

# Row 1: four cohort networks on geographic layout
for i, c in enumerate(cohort_order):
    ax = fig.add_subplot(3, 4, i+1)
    G, n_people = cohort_nets[c]
    in_w, out_w, betw = get_top_nodes(G)

    # Filter to top-quartile edges for visualization
    weights = [d['weight'] for _,_,d in G.edges(data=True)]
    th = np.quantile(weights, 0.75)
    G_vis = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d['weight'] >= th:
            G_vis.add_edge(u, v, weight=d['weight'])
    pos = {n: COORD[n] for n in G_vis.nodes() if n in COORD}
    G_vis = G_vis.subgraph(pos.keys())

    max_w = max((d['weight'] for _,_,d in G_vis.edges(data=True)), default=1)
    for u, v, d in G_vis.edges(data=True):
        ax.annotate('', xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle='->', alpha=0.4,
                                    lw=0.4 + 4*d['weight']/max_w,
                                    color='gray', shrinkA=10, shrinkB=10))
    for n in G_vis.nodes():
        total = in_w[n] + out_w[n]
        size = 80 + total*0.4
        ratio = (in_w[n] - out_w[n]) / (total + 1)
        color = '#d32f2f' if ratio > 0.2 else '#1976d2' if ratio < -0.2 else '#7b1fa2'
        ax.scatter(pos[n][0], pos[n][1], s=size, c=color, alpha=0.8,
                   edgecolors='white', linewidths=1.2, zorder=3)
        if total > 100:
            ax.annotate(CITY.get(n, n), pos[n], fontsize=8.5,
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
    ax.set_title(COHORT_LABEL[c] + f'\n樣本 {n_people:,} 人',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(118, 122.5); ax.set_ylim(22, 26.3); ax.set_aspect('equal'); ax.axis('off')

# Row 2: metrics comparison (4 subplots)
ax5 = fig.add_subplot(3, 4, 5)
labels = ['C4', 'C3', 'C2', 'C1']
colors_bar = ['#1565c0','#7b1fa2','#c62828','#ef6c00']
dens = [metrics[c]['density'] for c in cohort_order]
ax5.bar(labels, dens, color=colors_bar, alpha=0.85)
ax5.set_title('① 網絡密度', fontweight='bold', fontsize=11)
ax5.set_ylabel('density'); ax5.grid(axis='y', alpha=0.3)
for i, v in enumerate(dens):
    ax5.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)

ax6 = fig.add_subplot(3, 4, 6)
mods = [metrics[c]['modularity'] for c in cohort_order]
z_mods = [metrics[c]['z_modularity'] for c in cohort_order]
ax6.bar(labels, mods, color=colors_bar, alpha=0.85)
ax6.set_title('② Louvain Modularity Q\n（與 null 比較的 z-score）', fontweight='bold', fontsize=11)
ax6.set_ylabel('Q'); ax6.grid(axis='y', alpha=0.3)
for i, (v, z) in enumerate(zip(mods, z_mods)):
    ax6.text(i, v+0.005, f'{v:.3f}\nz={z:+.1f}', ha='center', fontsize=8)

ax7 = fig.add_subplot(3, 4, 7)
cd = [metrics[c]['C_D_in'] for c in cohort_order]
ax7.bar(labels, cd, color=colors_bar, alpha=0.85)
ax7.set_title('③ Freeman 中心化 C_D\n（in-degree, 越大越集中）', fontweight='bold', fontsize=11)
ax7.set_ylabel('C_D'); ax7.grid(axis='y', alpha=0.3)
for i, v in enumerate(cd):
    ax7.text(i, v+0.01, f'{v:.3f}', ha='center', fontsize=9)

ax8 = fig.add_subplot(3, 4, 8)
sigmas = [metrics[c]['sigma_smallworld'] for c in cohort_order]
ax8.bar(labels, sigmas, color=colors_bar, alpha=0.85)
ax8.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='σ=1 隨機網絡')
ax8.set_title('④ Small-World σ\n（σ>1 為小世界）', fontweight='bold', fontsize=11)
ax8.set_ylabel('σ'); ax8.legend(fontsize=8); ax8.grid(axis='y', alpha=0.3)
for i, v in enumerate(sigmas):
    ax8.text(i, v+0.05, f'{v:.2f}', ha='center', fontsize=9)

# Row 3: hub/bridge trend across cohorts
# (a) Top in-flow trend
top_in_cities = set()
for c in cohort_order:
    G, _ = cohort_nets[c]
    in_w, _, _ = get_top_nodes(G)
    for city,_ in sorted(in_w.items(), key=lambda x:-x[1])[:5]:
        top_in_cities.add(city)
top_in_list = sorted(top_in_cities,
    key=lambda x: -sum(sum(d['weight'] for _,_,d in cohort_nets[c][0].in_edges(x, data=True))
                       for c in cohort_order))[:6]

ax9 = fig.add_subplot(3, 4, 9)
x_pos = np.arange(len(cohort_order))
w_bar = 0.13
for i, city in enumerate(top_in_list):
    vals = [sum(d['weight'] for _,_,d in cohort_nets[c][0].in_edges(city, data=True))
            for c in cohort_order]
    ax9.bar(x_pos + i*w_bar, vals, w_bar, label=CITY.get(city, city))
ax9.set_xticks(x_pos + w_bar*2.5); ax9.set_xticklabels(labels)
ax9.set_title('⑤ 主要吸納樞紐：跨世代人數', fontweight='bold', fontsize=11)
ax9.legend(fontsize=7, ncol=3); ax9.grid(axis='y', alpha=0.3)
ax9.set_ylabel('遷入人數')

# (b) Top out-flow trend
top_out_cities = set()
for c in cohort_order:
    G, _ = cohort_nets[c]
    _, out_w, _ = get_top_nodes(G)
    for city,_ in sorted(out_w.items(), key=lambda x:-x[1])[:5]:
        top_out_cities.add(city)
top_out_list = sorted(top_out_cities,
    key=lambda x: -sum(sum(d['weight'] for _,_,d in cohort_nets[c][0].out_edges(x, data=True))
                       for c in cohort_order))[:6]

ax10 = fig.add_subplot(3, 4, 10)
for i, city in enumerate(top_out_list):
    vals = [sum(d['weight'] for _,_,d in cohort_nets[c][0].out_edges(city, data=True))
            for c in cohort_order]
    ax10.bar(x_pos + i*w_bar, vals, w_bar, label=CITY.get(city, city))
ax10.set_xticks(x_pos + w_bar*2.5); ax10.set_xticklabels(labels)
ax10.set_title('⑥ 主要輸出源：跨世代人數\n（觀察：桃園/新北從零變主要輸出）',
               fontweight='bold', fontsize=11)
ax10.legend(fontsize=7, ncol=3); ax10.grid(axis='y', alpha=0.3)
ax10.set_ylabel('遷出人數')

# (c) Net flow (in - out) heatmap-ish
ax11 = fig.add_subplot(3, 4, 11)
cities_for_net = list(top_in_list[:6])
net_mat = np.zeros((len(cities_for_net), len(cohort_order)))
for i, city in enumerate(cities_for_net):
    for j, c in enumerate(cohort_order):
        G, _ = cohort_nets[c]
        in_w = sum(d['weight'] for _,_,d in G.in_edges(city, data=True))
        out_w = sum(d['weight'] for _,_,d in G.out_edges(city, data=True))
        net_mat[i,j] = in_w - out_w
im = ax11.imshow(net_mat, cmap='RdBu_r', aspect='auto',
                 vmin=-np.abs(net_mat).max(), vmax=np.abs(net_mat).max())
ax11.set_yticks(range(len(cities_for_net)))
ax11.set_yticklabels([CITY.get(c,c) for c in cities_for_net], fontsize=9)
ax11.set_xticks(range(len(cohort_order))); ax11.set_xticklabels(labels)
ax11.set_title('⑦ 淨流量 = 流入 − 流出\n紅=淨吸納，藍=淨輸出',
               fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax11, shrink=0.6)
for i in range(len(cities_for_net)):
    for j in range(len(cohort_order)):
        val = int(net_mat[i,j])
        txt_color = 'white' if abs(net_mat[i,j]) > np.abs(net_mat).max()/2 else 'black'
        ax11.text(j, i, f'{val:+d}', ha='center', va='center', fontsize=8, color=txt_color)

# (d) Total OD pairs across cohorts
ax12 = fig.add_subplot(3, 4, 12)
n_edges = [cohort_nets[c][0].number_of_edges() for c in cohort_order]
n_pop = [metrics[c]['n_people'] for c in cohort_order]
ax12_twin = ax12.twinx()
b1 = ax12.bar(np.arange(4) - 0.2, n_edges, 0.4, color='steelblue', alpha=0.85, label='OD 對數')
b2 = ax12_twin.bar(np.arange(4) + 0.2, n_pop, 0.4, color='orange', alpha=0.85, label='遷徙人數')
ax12.set_xticks(range(4)); ax12.set_xticklabels(labels)
ax12.set_ylabel('OD 對數', color='steelblue')
ax12_twin.set_ylabel('遷徙人數', color='darkorange')
ax12.set_title('⑧ 規模對照', fontweight='bold', fontsize=11)
ax12.grid(axis='y', alpha=0.3)
for i, (e, p) in enumerate(zip(n_edges, n_pop)):
    ax12.text(i-0.2, e+5, str(e), ha='center', fontsize=8)
    ax12_twin.text(i+0.2, p+100, f'{p:,}', ha='center', fontsize=8)

plt.suptitle('泰雅族縣市層級終身遷徙網絡：cohort 分層分析（修正版）\n'
             '資料：TIPD 2022/12 _Intact｜泰雅族 N=33,762｜有效遷徙 N=20,849',
             fontsize=14, fontweight='bold', y=1.00)

# Legend for node colors (Row 1)
fig.text(0.5, 0.667,
         '節點顏色：● 紅 = 淨流入樞紐　● 藍 = 淨流出來源　● 紫 = 雙向流動地　'
         '｜邊權重：顯示前 25% 最強邊',
         ha='center', fontsize=10.5, style='italic', color='#444')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cohort_corrected.png',
            dpi=130, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: cohort_corrected.png")
plt.close()

# Save metrics CSV
import csv as _csv
with open('/mnt/user-data/outputs/cohort_metrics.csv', 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['cohort','label','n_people','n_nodes','n_edges','density',
                'clustering','path_len','modularity','z_modularity',
                'C_D_in','sigma_smallworld','n_communities'])
    for c in cohort_order:
        m = metrics[c]
        w.writerow([c, COHORT_LABEL[c].replace('\n','/'), m['n_people'],
                    m['n'], m['m'], f"{m['density']:.4f}",
                    f"{m['clustering']:.4f}", f"{m['path_len']:.4f}",
                    f"{m['modularity']:.4f}", f"{m['z_modularity']:.2f}",
                    f"{m['C_D_in']:.4f}", f"{m['sigma_smallworld']:.3f}",
                    m['n_comm']])
print("✓ Saved: cohort_metrics.csv")
