"""
Leiden 演算法穩健性對照：
1. 鄉鎮人口共源相似度網絡（與 Louvain 結果比對）
2. 四個 cohort 終身遷徙網絡（與 Louvain 結果比對）

比對指標：
- Adjusted Rand Index (ARI)：兩個分區的一致度，1.0=完全相同，0=隨機
- Normalized Mutual Information (NMI)
- 兩法各自的 modularity Q
- partition 圖示對照
"""
import csv
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from town_codes import TOWN_NAMES, MOUNTAIN_TOWNS, PREF_NAMES, norm_pref

mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
mpl.rcParams['axes.unicode_minus'] = False

# ======================================================
# Helper: convert NetworkX → igraph (preserving node order)
# ======================================================
def nx_to_ig(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs['name'] = nodes
    g.es['weight'] = weights
    return g, nodes

# Helper: list of community sets -> dict node -> id
def comms_to_dict(comms):
    d = {}
    for cid, c in enumerate(comms):
        for n in c:
            d[n] = cid
    return d

def partition_to_dict(part, nodes):
    """leidenalg partition → dict node→community id"""
    d = {}
    for cid, c in enumerate(part):
        for vid in c:
            d[nodes[vid]] = cid
    return d

# ======================================================
# 1. 載入泰雅族資料
# ======================================================
with open('intact.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['EthnicityCate']=='2']

# ======================================================
# 2. 重建鄉鎮共源相似度網絡（同前一份 prototype 邏輯）
# ======================================================
MIN_POP = 30
TOP_EDGE_PCT = 0.10

town_birth = defaultdict(Counter)
town_pop = Counter()
for r in rows:
    town_birth[r['AdmiTownCate']][norm_pref(r['BirthPlaceCate'])] += 1
    town_pop[r['AdmiTownCate']] += 1
townships = sorted([t for t, p in town_pop.items() if p >= MIN_POP])
all_prefs = sorted(set(p for t in townships for p in town_birth[t]))
vectors = {t: np.array([town_birth[t].get(p,0)/town_pop[t] for p in all_prefs])
           for t in townships}
def br(v1, v2): return 1 - 0.5 * np.abs(v1 - v2).sum()
n = len(townships)
S = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        S[i,j] = S[j,i] = br(vectors[townships[i]], vectors[townships[j]])
threshold = np.quantile(S[np.triu_indices(n, k=1)], 1 - TOP_EDGE_PCT)
G_sim = nx.Graph()
for t in townships: G_sim.add_node(t)
for i in range(n):
    for j in range(i+1, n):
        if S[i,j] >= threshold:
            G_sim.add_edge(townships[i], townships[j], weight=S[i,j])
G_sim = G_sim.subgraph(max(nx.connected_components(G_sim), key=len)).copy()

# Louvain
louvain_sim = nx.community.louvain_communities(G_sim, weight='weight', seed=42)
Q_louvain_sim = nx.community.modularity(G_sim, louvain_sim, weight='weight')

# Leiden
g_ig, ig_nodes = nx_to_ig(G_sim)
leiden_sim_part = la.find_partition(g_ig, la.ModularityVertexPartition,
                                    weights='weight', seed=42)
leiden_sim = [set(g_ig.vs[c]['name']) for c in leiden_sim_part]
Q_leiden_sim = nx.community.modularity(G_sim, leiden_sim, weight='weight')

# ARI / NMI
nodes_list = sorted(G_sim.nodes())
lou_d = comms_to_dict(louvain_sim)
lei_d = comms_to_dict(leiden_sim)
lou_labels = [lou_d[n] for n in nodes_list]
lei_labels = [lei_d[n] for n in nodes_list]
ari_sim = adjusted_rand_score(lou_labels, lei_labels)
nmi_sim = normalized_mutual_info_score(lou_labels, lei_labels)

print("="*70)
print("【鄉鎮共源相似度網絡】Louvain vs Leiden")
print("="*70)
print(f"  Louvain : {len(louvain_sim):>2} 社群, Q = {Q_louvain_sim:.4f}")
print(f"  Leiden  : {len(leiden_sim):>2} 社群, Q = {Q_leiden_sim:.4f}")
print(f"  ARI = {ari_sim:.4f}   NMI = {nmi_sim:.4f}")
print(f"  → {'高度一致' if ari_sim > 0.7 else '中度一致' if ari_sim > 0.4 else '一致性低'}")

# ======================================================
# 3. 建四個 cohort 縣市終身遷徙網絡，並對每個做 Louvain vs Leiden
# ======================================================
PREF_NORM_MAP = {'10003':'68000','10006':'66002','10019':'66001','10001':'65000',
                 '10011':'67002','10021':'67001','10012':'64002','64000':'64001'}
PREF_MERGE = {'66002':'66000','66001':'66000','67002':'67000','67001':'67000',
              '64002':'64000','64001':'64000'}
def normp(c):
    c = PREF_NORM_MAP.get(c, c); return PREF_MERGE.get(c, c)

def age_to_cohort(a):
    a = int(a) if a != '00' else 0
    if a == 0: return None
    if a <= 5: return 'C1'
    if a <= 9: return 'C2'
    if a <= 13: return 'C3'
    return 'C4'

moved = []
for r in rows:
    o = normp(r['BirthPlaceCate']); d = normp(r['AdmiPrefCityCate'])
    if o == d: continue
    c = age_to_cohort(r['Age5FCate'])
    if c: moved.append((o, d, c))

cohort_results = {}
print("\n" + "="*70)
print("【縣市 cohort 終身遷徙網絡】Louvain vs Leiden")
print("="*70)

# Order from oldest to youngest
cohort_order = ['C4', 'C3', 'C2', 'C1']
cohort_label = {'C4':'C4 (65+ 歲)','C3':'C3 (45-64 歲)',
                'C2':'C2 (25-44 歲)','C1':'C1 (<25 歲)'}

for c in cohort_order:
    od = Counter((o,d) for o,d,cc in moved if cc==c)
    G = nx.DiGraph()
    for (o,d), w in od.items():
        G.add_edge(o, d, weight=w)
    # For community detection, use undirected projection
    G_un = G.to_undirected()
    # Merge bi-directional weights
    for u, v in G_un.edges():
        w_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0
        w_vu = G[v][u]['weight'] if G.has_edge(v, u) else 0
        G_un[u][v]['weight'] = w_uv + w_vu

    # Largest CC
    G_un = G_un.subgraph(max(nx.connected_components(G_un), key=len)).copy()

    # Louvain
    lou = nx.community.louvain_communities(G_un, weight='weight', seed=42)
    Q_lou = nx.community.modularity(G_un, lou, weight='weight')

    # Leiden
    g_ig, ig_nodes_c = nx_to_ig(G_un)
    lei_part = la.find_partition(g_ig, la.ModularityVertexPartition,
                                 weights='weight', seed=42)
    lei = [set(g_ig.vs[ci]['name']) for ci in lei_part]
    Q_lei = nx.community.modularity(G_un, lei, weight='weight')

    nodes_c = sorted(G_un.nodes())
    lou_d = comms_to_dict(lou); lei_d = comms_to_dict(lei)
    ari = adjusted_rand_score([lou_d[n] for n in nodes_c],
                              [lei_d[n] for n in nodes_c])
    nmi = normalized_mutual_info_score([lou_d[n] for n in nodes_c],
                                       [lei_d[n] for n in nodes_c])

    cohort_results[c] = {
        'G': G_un, 'louvain': lou, 'leiden': lei,
        'Q_lou': Q_lou, 'Q_lei': Q_lei, 'ari': ari, 'nmi': nmi,
        'n_lou': len(lou), 'n_lei': len(lei),
    }

    print(f"\n{cohort_label[c]}: {G_un.number_of_nodes()} 節點, {G_un.number_of_edges()} 邊")
    print(f"  Louvain: {len(lou):>2} 社群, Q = {Q_lou:.4f}")
    print(f"  Leiden : {len(lei):>2} 社群, Q = {Q_lei:.4f}")
    print(f"  ARI = {ari:.4f}, NMI = {nmi:.4f}")

# ======================================================
# 4. 視覺化：穩健性對照圖
# ======================================================
fig = plt.figure(figsize=(16, 10))

# (A) Modularity comparison
ax1 = fig.add_subplot(2, 3, 1)
labels_all = ['鄉鎮相似度'] + [cohort_label[c] for c in cohort_order]
Q_lou_all = [Q_louvain_sim] + [cohort_results[c]['Q_lou'] for c in cohort_order]
Q_lei_all = [Q_leiden_sim] + [cohort_results[c]['Q_lei'] for c in cohort_order]
x = np.arange(len(labels_all))
w = 0.35
ax1.bar(x - w/2, Q_lou_all, w, label='Louvain', color='#1976d2', alpha=0.85)
ax1.bar(x + w/2, Q_lei_all, w, label='Leiden',  color='#c62828', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(labels_all, rotation=20, fontsize=9, ha='right')
ax1.set_ylabel('Modularity Q')
ax1.set_title('① 兩演算法的 Modularity 對照', fontweight='bold', fontsize=11)
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
for i, (l, le) in enumerate(zip(Q_lou_all, Q_lei_all)):
    ax1.text(i - w/2, l + 0.005, f'{l:.3f}', ha='center', fontsize=7)
    ax1.text(i + w/2, le + 0.005, f'{le:.3f}', ha='center', fontsize=7)

# (B) ARI / NMI
ax2 = fig.add_subplot(2, 3, 2)
ari_all = [ari_sim] + [cohort_results[c]['ari'] for c in cohort_order]
nmi_all = [nmi_sim] + [cohort_results[c]['nmi'] for c in cohort_order]
ax2.bar(x - w/2, ari_all, w, label='ARI', color='#2e7d32', alpha=0.85)
ax2.bar(x + w/2, nmi_all, w, label='NMI', color='#ef6c00', alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(labels_all, rotation=20, fontsize=9, ha='right')
ax2.set_ylabel('一致性指標')
ax2.set_title('② Louvain vs Leiden 一致性\n(1.0=完全相同分區)',
              fontweight='bold', fontsize=11)
ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='高度一致 (0.7)')
ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 1.05)
for i, (a, m) in enumerate(zip(ari_all, nmi_all)):
    ax2.text(i - w/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=7)
    ax2.text(i + w/2, m + 0.01, f'{m:.2f}', ha='center', fontsize=7)

# (C) Number of communities
ax3 = fig.add_subplot(2, 3, 3)
n_lou = [len(louvain_sim)] + [cohort_results[c]['n_lou'] for c in cohort_order]
n_lei = [len(leiden_sim)] + [cohort_results[c]['n_lei'] for c in cohort_order]
ax3.bar(x - w/2, n_lou, w, label='Louvain', color='#1976d2', alpha=0.85)
ax3.bar(x + w/2, n_lei, w, label='Leiden',  color='#c62828', alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(labels_all, rotation=20, fontsize=9, ha='right')
ax3.set_ylabel('社群數')
ax3.set_title('③ 偵測社群數對照', fontweight='bold', fontsize=11)
ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

# (D)-(G): Louvain vs Leiden partition for each cohort, side by side
# Use a single layout per cohort for visual fairness
for idx, c in enumerate(cohort_order):
    G_un = cohort_results[c]['G']
    lou = cohort_results[c]['louvain']
    lei = cohort_results[c]['leiden']
    lou_d = comms_to_dict(lou); lei_d = comms_to_dict(lei)
    pos = nx.spring_layout(G_un, seed=42, weight='weight', k=0.5)

    ax = fig.add_subplot(2, 6, 7 + idx*1) if False else None
    # We'll create custom subplots row 2 with 4 panels
    pass

# Manually layout row 2: 4 cohort panels each with sub-split for Louvain/Leiden
# Use one big panel per cohort, with colors representing Louvain (left) vs Leiden (right) overlay
gs_axes = []
for i, c in enumerate(cohort_order):
    ax = fig.add_subplot(2, 4, 5 + i)
    G_un = cohort_results[c]['G']
    lou = cohort_results[c]['louvain']
    lou_d = comms_to_dict(lou)
    pos = nx.spring_layout(G_un, seed=42, weight='weight', k=0.6)

    # Edges
    for u, v in G_un.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color='gray', alpha=0.15, lw=0.4, zorder=1)
    # Nodes colored by Louvain
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(lou), 3)))
    node_colors = [colors[lou_d[n]] for n in G_un.nodes()]
    nx.draw_networkx_nodes(G_un, pos, node_color=node_colors,
                            node_size=180, edgecolors='black',
                            linewidths=0.8, ax=ax)
    # Label top 5 by degree
    deg = dict(G_un.degree(weight='weight'))
    top5 = sorted(deg.items(), key=lambda x:-x[1])[:5]
    for n,_ in top5:
        ax.annotate(PREF_NAMES.get(n, n), pos[n], fontsize=7,
                    ha='center', va='center', fontweight='bold')
    ax.set_title(f"{cohort_label[c]}\n"
                 f"Louvain: {cohort_results[c]['n_lou']} 社群, Q={cohort_results[c]['Q_lou']:.2f} | "
                 f"ARI={cohort_results[c]['ari']:.2f}",
                 fontsize=9.5, fontweight='bold')
    ax.axis('off')

plt.suptitle('Leiden 演算法穩健性檢驗\n'
             '所有網絡 ARI≥0.7 即可宣稱 Louvain 結果穩健（不依賴特定演算法）',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/leiden_robustness_check.png',
            dpi=130, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: leiden_robustness_check.png")
plt.close()

# Summary table
print("\n" + "="*70)
print("總結：穩健性檢驗結果")
print("="*70)
print(f"{'網絡':<20} {'Lou Q':>8} {'Lei Q':>8} {'ARI':>8} {'NMI':>8} {'判定':>10}")
print(f"{'鄉鎮共源相似度':<20} {Q_louvain_sim:>8.3f} {Q_leiden_sim:>8.3f} "
      f"{ari_sim:>8.3f} {nmi_sim:>8.3f} {'穩健' if ari_sim>0.7 else '中等':>10}")
for c in cohort_order:
    r = cohort_results[c]
    print(f"{cohort_label[c]:<20} {r['Q_lou']:>8.3f} {r['Q_lei']:>8.3f} "
          f"{r['ari']:>8.3f} {r['nmi']:>8.3f} "
          f"{'穩健' if r['ari']>0.7 else '中等' if r['ari']>0.4 else '不穩健':>10}")
