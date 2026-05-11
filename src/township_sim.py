"""
Mills 2013 PNAS 方法翻譯版：
鄉鎮層級「人口共源相似度網絡」(Demographic Co-origin Similarity Network)

節點：泰雅族居住鄉鎮（人口 >= MIN_POP）
節點向量：該鄉鎮居民的「出生縣市分布」之 per-capita 比例
邊（無向加權）：兩鄉鎮之 Brainerd-Robinson 相似度
        BR_ij = 1 - 0.5 * sum(|p_ik - p_jk|)
        值域 [0, 1]，1=完全相同，0=完全不重疊
        對應 Mills 用陶器類型相似度的方法

注意：用 BR 而非 cosine — BR 對「比例分布」是經典且合理的測度
     並設定 threshold 只留 top quantile 強邊，避免「人人都像」的死網絡
"""
import csv
from collections import Counter, defaultdict
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from town_codes import TOWN_NAMES, MOUNTAIN_TOWNS, PREF_NAMES, norm_pref

mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
mpl.rcParams['axes.unicode_minus'] = False

MIN_POP = 30   # 鄉鎮最低泰雅族人口門檻
TOP_EDGE_PCT = 0.10   # 保留前 10% 最強邊

# =============== 1. 載入泰雅族資料 ===============
with open('intact.csv') as f:
    rows = [r for r in csv.DictReader(f) if r['EthnicityCate']=='2']

# =============== 2. 為每個鄉鎮，建立「居民出生縣市分布」向量 ===============
# town -> Counter(birth_prefecture -> count)
town_birth = defaultdict(Counter)
town_pop = Counter()
for r in rows:
    town = r['AdmiTownCate']
    birth = norm_pref(r['BirthPlaceCate'])
    town_birth[town][birth] += 1
    town_pop[town] += 1

# Filter by population
townships = sorted([t for t, p in town_pop.items() if p >= MIN_POP])
print(f"納入鄉鎮: {len(townships)} 個（人口 ≥ {MIN_POP}）")
print(f"涵蓋泰雅人口: {sum(town_pop[t] for t in townships):,} / {sum(town_pop.values()):,}")

# All prefectures across all townships
all_prefs = sorted(set(p for t in townships for p in town_birth[t]))
print(f"出生縣市維度: {len(all_prefs)}")

# Build per-capita vectors
vectors = {}
for t in townships:
    pop = town_pop[t]
    vec = np.array([town_birth[t].get(p, 0) / pop for p in all_prefs])
    vectors[t] = vec

# =============== 3. Brainerd-Robinson similarity matrix ===============
def br_similarity(v1, v2):
    return 1 - 0.5 * np.abs(v1 - v2).sum()

n = len(townships)
S = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        S[i,j] = S[j,i] = br_similarity(vectors[townships[i]], vectors[townships[j]])

print(f"\nBR 相似度矩陣統計：")
upper = S[np.triu_indices(n, k=1)]
print(f"  min={upper.min():.3f}  median={np.median(upper):.3f}  "
      f"mean={upper.mean():.3f}  max={upper.max():.3f}")

# =============== 4. 建網絡：留 top 10% 強邊 ===============
threshold = np.quantile(upper, 1 - TOP_EDGE_PCT)
print(f"\n邊權重 threshold (top {int(TOP_EDGE_PCT*100)}%): {threshold:.3f}")

G = nx.Graph()
for i, t in enumerate(townships):
    G.add_node(t, pop=town_pop[t], is_mountain=(t in MOUNTAIN_TOWNS))
for i in range(n):
    for j in range(i+1, n):
        if S[i,j] >= threshold:
            G.add_edge(townships[i], townships[j], weight=S[i,j])

print(f"網絡: {G.number_of_nodes()} 節點, {G.number_of_edges()} 邊")
print(f"密度: {nx.density(G):.3f}")

# 取最大連通分量
ccs = list(nx.connected_components(G))
print(f"連通分量數: {len(ccs)}, 最大分量大小: {len(max(ccs, key=len))}")
Glcc = G.subgraph(max(ccs, key=len)).copy()

# =============== 5. 中心性 + 社群偵測 ===============
betw = nx.betweenness_centrality(Glcc, weight='weight', normalized=True)
deg_w = dict(Glcc.degree(weight='weight'))
eig = nx.eigenvector_centrality_numpy(Glcc, weight='weight')

# Louvain
comms = nx.community.louvain_communities(Glcc, weight='weight', seed=42)
modularity = nx.community.modularity(Glcc, comms, weight='weight')
print(f"\nLouvain: {len(comms)} 社群, Modularity Q = {modularity:.3f}")

# Map node → community id
node_comm = {}
for cid, nodes in enumerate(comms):
    for n_ in nodes:
        node_comm[n_] = cid

# =============== 6. 報表：每個社群的代表性鄉鎮、結構特徵 ===============
print("\n" + "="*70)
print("各社群剖析")
print("="*70)
for cid in range(len(comms)):
    members = [n for n in comms[cid] if n in node_comm]
    members_sorted = sorted(members, key=lambda x: -deg_w[x])
    n_mountain = sum(1 for m in members if m in MOUNTAIN_TOWNS)
    total_pop = sum(town_pop[m] for m in members)
    print(f"\n社群 {cid+1}: {len(members)} 鄉鎮, 泰雅人口 {total_pop:,}, "
          f"山地原鄉 {n_mountain}/{len(members)}")
    print(f"  前 8 個高 degree 成員:")
    for t in members_sorted[:8]:
        name = TOWN_NAMES.get(t, t)
        flag = '🏔' if t in MOUNTAIN_TOWNS else '🏙'
        print(f"    {flag} {name} ({town_pop[t]} 人, k_w={deg_w[t]:.1f}, "
              f"betw={betw[t]:.4f})")

# =============== 7. 結構洞分析（Burt） ===============
# 對 top 30 high-betweenness 節點計算 effective size 與 constraint
print("\n" + "="*70)
print("結構洞分析（top 15 betweenness）")
print("="*70)
top_betw = sorted(betw.items(), key=lambda x: -x[1])[:15]
print(f"{'鄉鎮':<20} {'人口':>6} {'山地':>4} {'社群':>4} "
      f"{'betw':>8} {'eff_size':>10} {'constraint':>11}")
eff = nx.effective_size(Glcc, weight='weight')
con = nx.constraint(Glcc, weight='weight')
for t, b in top_betw:
    name = TOWN_NAMES.get(t, t)
    flag = '🏔' if t in MOUNTAIN_TOWNS else '🏙'
    e = eff.get(t, float('nan'))
    c = con.get(t, float('nan'))
    print(f"{name:<20} {town_pop[t]:>6} {flag:>4} {node_comm[t]+1:>4} "
          f"{b:>8.4f} {e:>10.2f} {c:>11.3f}")

# =============== 8. 視覺化：三聯圖 ===============
fig = plt.figure(figsize=(20, 9))

# (a) 網絡圖
ax1 = fig.add_subplot(1, 3, (1, 2))
pos = nx.kamada_kawai_layout(Glcc, weight='weight')

# 顏色 = 社群；形狀 = 是否山地原鄉
comm_colors = plt.cm.tab10(np.linspace(0, 1, max(len(comms), 3)))
node_colors = [comm_colors[node_comm[n]] for n in Glcc.nodes()]
node_sizes = [40 + np.sqrt(town_pop[n])*8 for n in Glcc.nodes()]

# Edges - alpha by weight
edge_weights = np.array([d['weight'] for _,_,d in Glcc.edges(data=True)])
ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)
for (u,v,d), alpha in zip(Glcc.edges(data=True), ew_norm):
    ax1.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
             color='gray', alpha=0.1 + alpha*0.4, linewidth=0.3 + alpha*1.5,
             zorder=1)
# Mountain = star; non-mountain = circle
mountain_nodes = [n for n in Glcc.nodes() if n in MOUNTAIN_TOWNS]
nonmountain_nodes = [n for n in Glcc.nodes() if n not in MOUNTAIN_TOWNS]
ax1.scatter([pos[n][0] for n in nonmountain_nodes],
            [pos[n][1] for n in nonmountain_nodes],
            s=[40+np.sqrt(town_pop[n])*8 for n in nonmountain_nodes],
            c=[comm_colors[node_comm[n]] for n in nonmountain_nodes],
            edgecolors='white', linewidths=0.7, alpha=0.85, zorder=2)
ax1.scatter([pos[n][0] for n in mountain_nodes],
            [pos[n][1] for n in mountain_nodes],
            s=[100+np.sqrt(town_pop[n])*12 for n in mountain_nodes],
            c=[comm_colors[node_comm[n]] for n in mountain_nodes],
            edgecolors='black', linewidths=1.8, marker='*', zorder=3)

# Labels: top by population + all mountains
to_label = set([n for n,_ in sorted([(n,town_pop[n]) for n in Glcc.nodes()],
                                    key=lambda x:-x[1])[:25]])
to_label.update(mountain_nodes)
for n in to_label:
    name = TOWN_NAMES.get(n, n[-4:])
    ax1.annotate(name, pos[n], fontsize=8, ha='center', va='center',
                 fontweight='bold',
                 xytext=(0, -10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

ax1.set_title('泰雅族鄉鎮「人口共源相似度網絡」\n'
              f'(節點={Glcc.number_of_nodes()}, 邊={Glcc.number_of_edges()}, '
              f'Q={modularity:.3f}, {len(comms)} 社群)\n'
              '★ 山地原鄉　● 都會 / 平地鄉鎮　顏色=Louvain 社群',
              fontsize=12, fontweight='bold')
ax1.axis('off')

# (b) Community composition
ax2 = fig.add_subplot(2, 3, 3)
comm_stats = []
for cid in range(len(comms)):
    members = list(comms[cid])
    n_mountain = sum(1 for m in members if m in MOUNTAIN_TOWNS)
    n_other = len(members) - n_mountain
    pop = sum(town_pop[m] for m in members)
    comm_stats.append((cid+1, n_mountain, n_other, pop))

cids = [s[0] for s in comm_stats]
n_mtn = [s[1] for s in comm_stats]
n_oth = [s[2] for s in comm_stats]
ax2.bar(cids, n_mtn, label='山地原鄉 ★', color='#2e7d32', alpha=0.85)
ax2.bar(cids, n_oth, bottom=n_mtn, label='都會/平地 ●', color='#c62828', alpha=0.85)
ax2.set_xlabel('社群編號'); ax2.set_ylabel('鄉鎮數')
ax2.set_title('各社群的山地 vs 都會構成', fontweight='bold', fontsize=11)
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
for i, (c, m, o, p) in enumerate(comm_stats):
    ax2.text(c, m+o+0.3, f'n={m+o}\n{p:,}人', ha='center', fontsize=8)

# (c) Top betweenness (橋接型)
ax3 = fig.add_subplot(2, 3, 6)
top12 = sorted(betw.items(), key=lambda x: -x[1])[:12]
names = [('★ ' if t in MOUNTAIN_TOWNS else '● ') + TOWN_NAMES.get(t, t[-4:]) for t,_ in top12]
values = [b for _,b in top12]
colors_bar = ['#2e7d32' if t in MOUNTAIN_TOWNS else '#c62828' for t,_ in top12]
y = np.arange(len(names))
ax3.barh(y, values, color=colors_bar, alpha=0.85)
ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=8.5)
ax3.invert_yaxis()
ax3.set_xlabel('Betweenness Centrality')
ax3.set_title('Top 12 橋接鄉鎮\n(連接不同社群的關鍵節點)', fontweight='bold', fontsize=11)
ax3.grid(axis='x', alpha=0.3)

plt.suptitle('泰雅族鄉鎮層級「人口共源相似度網絡」 — Mills (2013, PNAS) 方法翻譯版\n'
             '資料：TIPD 2022/12 _Intact｜方法：Brainerd-Robinson 相似度 + Louvain',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/township_similarity_network.png',
            dpi=130, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: township_similarity_network.png")
plt.close()

# =============== 9. 匯出可重用 edge list（給後續分析用） ===============
import csv as _csv
with open('/mnt/user-data/outputs/township_similarity_edgelist.csv', 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['source_code','source_name','target_code','target_name',
                'similarity','source_pop','target_pop',
                'source_is_mountain','target_is_mountain'])
    for u,v,d in Glcc.edges(data=True):
        w.writerow([u, TOWN_NAMES.get(u,u), v, TOWN_NAMES.get(v,v),
                    f"{d['weight']:.4f}", town_pop[u], town_pop[v],
                    int(u in MOUNTAIN_TOWNS), int(v in MOUNTAIN_TOWNS)])
print("✓ Saved: township_similarity_edgelist.csv")
