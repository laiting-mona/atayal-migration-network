"""
Layer A Township v1 — 泰雅族鄉鎮市區層級三方法分析
═══════════════════════════════════════════════════════════
方法 A : 縣市→鄉鎮 二模式有向網絡（出生縣市 → 現居鄉鎮）
         每期計算：入度多樣性（Shannon Entropy）、Gini 集中度
         追蹤：山地原鄉的主要「引力」來源縣市變化

方法 B : 鄉鎮 BR 相似度時序
         Brainerd-Robinson similarity（出生縣市向量）
         20期 × top 邊的演變、Louvain 社群穩定性

方法 C : 山地原鄉人口存量時序
         8 個核心山地原鄉的半年期存量折線圖
         (復興區、尖石鄉、五峰鄉、大同鄉、南澳鄉、和平區、烏來區、仁愛鄉)
═══════════════════════════════════════════════════════════
"""
import csv, sys, io, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import community as community_louvain
except ImportError:
    community_louvain = None

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_HALF = ROOT / 'data' / 'PopulationDynamcicsData_HalfYearPeriod'
OUT_T     = ROOT / 'outputs' / 'tables'
OUT_F     = ROOT / 'outputs' / 'figures'
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

# ── font ───────────────────────────────────────────────────────────────────
for f in ['Noto Sans CJK TC','Noto Sans CJK JP','Microsoft JhengHei','Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = f; break
    except: pass
matplotlib.rcParams['axes.unicode_minus'] = False

# ── county code normalisation ──────────────────────────────────────────────
BIRTH_NORM = {'10001':'65000','10003':'68000','10006':'66000','10019':'66000',
              '10011':'67000','10021':'67000','10012':'64000','64000':'64000',
              '9007':'09007','9020':'09020'}
ADMI_PREF_NORM = {'10003':'68000','66001':'66000','66002':'66000','67001':'67000',
                  '67002':'67000','64001':'64000','64002':'64000'}

def nb(c): c=c.strip(); return BIRTH_NORM.get(c,c)
def np_(c): c=c.strip(); return ADMI_PREF_NORM.get(c,c)

# ── township code normalisation ────────────────────────────────────────────
# Taoyuan was upgraded 2014-12, changing township prefix 10003 → 68000
def norm_town(tc):
    tc = tc.strip()
    if tc.startswith('10003') and len(tc)==7:
        return '68000' + tc[5:]
    return tc

# ── county names ───────────────────────────────────────────────────────────
CITY = {'09007':'連江','09020':'金門','10002':'宜蘭','10004':'新竹縣','10005':'苗栗',
        '10007':'彰化','10008':'南投','10009':'雲林','10010':'嘉義縣','10013':'屏東',
        '10014':'臺東','10015':'花蓮','10016':'澎湖','10017':'基隆','10018':'新竹市',
        '10020':'嘉義市','63000':'臺北','64000':'高雄','65000':'新北','66000':'臺中',
        '67000':'臺南','68000':'桃園'}

# ── mountain indigenous township names (identified via SandiStatusCate=1) ──
# Taoyuan 68000: 復興區(13), Hsinchu 10004: 尖石鄉(12)/五峰鄉(13)
# Miaoli 10005: 泰安鄉(18), Taichung 66000: 和平區(29)
# New Taipei 65000: 烏來區(29), Nantou 10008: 仁愛鄉(13)
# Yilan 10002: 大同鄉(12)/南澳鄉(11), Hualien 10015: 秀林鄉(11)/萬榮鄉(12)/卓溪鄉(13)
MOUNTAIN_TOWN_NAMES = {
    '6800013': '復興區',    # Taoyuan
    '1000412': '尖石鄉',    # Hsinchu County
    '1000413': '五峰鄉',    # Hsinchu County
    '1000518': '泰安鄉',    # Miaoli
    '1000212': '大同鄉',    # Yilan
    '1000211': '南澳鄉',    # Yilan
    '6600029': '和平區',    # Taichung
    '6500029': '烏來區',    # New Taipei
    '1000813': '仁愛鄉',    # Nantou (Atayal/Seediq)
    '1001511': '秀林鄉',    # Hualien
    '1001512': '萬榮鄉',    # Hualien
    '1001513': '卓溪鄉',    # Hualien
    '1000812': '信義鄉',    # Nantou small
    '6400038': '茂林區',    # Kaohsiung (small)
}
# Core 8 for Method C time-series (the main Atayal mountain homelands)
CORE_MOUNTAINS = ['6800013','1000412','1000413','1000212','1000211',
                  '6600029','6500029','1000813']

def town_label(tc):
    if tc in MOUNTAIN_TOWN_NAMES:
        return MOUNTAIN_TOWN_NAMES[tc]
    pref = tc[:5] if len(tc)>=5 else tc
    return CITY.get(pref, tc) + f'(T{tc[5:]})'

# ── periods ────────────────────────────────────────────────────────────────
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
PIDS = [p for p,_,_ in PERIODS]
MIN_POP_TOWN = 30  # 鄉鎮最低泰雅族人口門檻（方法B用）

# ═══════════════════════════════════════════════════════════════════════════
# 資料讀取
# ═══════════════════════════════════════════════════════════════════════════
def load_township(folder, csvname):
    """
    Returns:
      town_pop  : Counter {town_code: total_Atayal_pop}
      birth_town: Counter {(birth_county, town_code): count}  ← Method A
      town_birth: dict {town_code: Counter{birth_county: count}} ← Method B
      sandi_cnt : Counter {town_code: sandi_count}
    """
    path = DATA_HALF / folder / 'PopnDynamics_Intact' / csvname
    town_pop  = Counter()
    birth_town= Counter()
    town_birth= defaultdict(Counter)
    sandi_cnt = Counter()
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('EthnicityCate','').strip() != '2': continue
            tc  = norm_town(row.get('AdmiTownCate','').strip())
            src = nb(row.get('BirthPlaceCate','').strip())
            ss  = row.get('SandiStatusCate','').strip()
            if not tc or tc=='0' or not src or src=='0': continue
            w = int(row.get('Weight','1'))
            town_pop[tc]    += w
            birth_town[(src,tc)] += w
            town_birth[tc][src]  += w
            if ss == '1': sandi_cnt[tc] += w
    return town_pop, birth_town, town_birth, sandi_cnt

print("="*65)
print("Layer A Township v1 — 三方法鄉鎮分析")
print("="*65)

# Main data load loop
all_tp, all_bt, all_tb, all_sc = {}, {}, {}, {}
for pid, folder, csvname in PERIODS:
    print(f"  [{pid}]", end=' ', flush=True)
    tp, bt, tb, sc = load_township(folder, csvname)
    all_tp[pid] = tp
    all_bt[pid] = bt
    all_tb[pid] = tb
    all_sc[pid] = sc
    # Identify sandi townships
    mt = sorted([t for t in tp if sc.get(t,0)/tp.get(t,1) >= 0.95 and tp[t]>=20], key=lambda x:-tp[x])
    print(f"總人口={sum(tp.values()):,}  鄉鎮={len(tp)}  山地原鄉={len(mt)}", flush=True)

print("\n✓ 資料載入完畢")

# ═══════════════════════════════════════════════════════════════════════════
# 方法 A：縣市→鄉鎮 二模式有向網絡
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("方法 A：縣市→鄉鎮二模式有向網絡")
print("─"*65)

def shannon_entropy(v):
    v = np.array(v, dtype=float)
    v = v[v>0]
    if len(v)==0: return 0.0
    p = v/v.sum()
    return float(-np.sum(p*np.log(p)))

def gini(v):
    v = np.array(sorted(v), dtype=float)
    n = len(v); s = v.sum()
    if s==0 or n<=1: return 0.0
    return float((2*np.sum(v*np.arange(1,n+1)) - (n+1)*s) / (n*s))

method_a_rows = []

for pid, folder, csvname in PERIODS:
    tp  = all_tp[pid]
    bt  = all_bt[pid]
    sc  = all_sc[pid]

    # Focus on townships with ≥ MIN_POP_TOWN Atayal
    valid_towns = {t for t,p in tp.items() if p >= MIN_POP_TOWN}

    for tc in valid_towns:
        # Which birth counties feed this township?
        sources = {bc: cnt for (bc,t),cnt in bt.items() if t==tc}
        total   = tp[tc]
        is_mtn  = sc.get(tc,0)/total >= 0.95
        name    = town_label(tc)
        county  = tc[:5]

        ent  = shannon_entropy(list(sources.values()))
        gin  = gini(list(sources.values()))
        n_src= len(sources)
        top_src = sorted(sources.items(), key=lambda x:-x[1])[:3]

        method_a_rows.append({
            'period': pid,
            'town_code': tc,
            'town_name': name,
            'county_pref': county,
            'county_name': CITY.get(county, county),
            'is_mountain': int(is_mtn),
            'total_pop': total,
            'n_source_counties': n_src,
            'diversity_entropy': round(ent, 4),
            'gini_concentration': round(gin, 4),
            'top1_src_county': CITY.get(top_src[0][0], top_src[0][0]) if top_src else '',
            'top1_src_cnt': top_src[0][1] if top_src else 0,
            'top2_src_county': CITY.get(top_src[1][0], top_src[1][0]) if len(top_src)>1 else '',
            'top2_src_cnt': top_src[1][1] if len(top_src)>1 else 0,
            'top3_src_county': CITY.get(top_src[2][0], top_src[2][0]) if len(top_src)>2 else '',
            'top3_src_cnt': top_src[2][1] if len(top_src)>2 else 0,
        })

df_a = pd.DataFrame(method_a_rows)
df_a.to_csv(OUT_T/'layer_a_town_method_a.csv', index=False, encoding='utf-8-sig')
print(f"  → layer_a_town_method_a.csv ({len(df_a)} 列)")

# Summary: mountain townships diversity over time
mtn_div = df_a[df_a.is_mountain==1].groupby('period')['diversity_entropy'].mean().reset_index()
all_div = df_a.groupby('period')['diversity_entropy'].mean().reset_index()
print("\n  山地原鄉平均入度多樣性 (Shannon Entropy)：")
print(mtn_div.set_index('period').rename(columns={'diversity_entropy':'mtn_mean_entropy'}).to_string())

# ═══════════════════════════════════════════════════════════════════════════
# 方法 B：鄉鎮 BR 相似度時序
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("方法 B：鄉鎮 BR 相似度時序")
print("─"*65)

def br_similarity(v1, v2):
    return 1 - 0.5 * np.abs(v1 - v2).sum()

method_b_summary = []

for pid, folder, csvname in PERIODS:
    tp   = all_tp[pid]
    tb   = all_tb[pid]

    towns = sorted([t for t,p in tp.items() if p >= MIN_POP_TOWN])
    all_prefs = sorted(set(p for t in towns for p in tb.get(t,{}).keys()))
    if len(towns) < 3 or not all_prefs:
        method_b_summary.append({'period':pid,'n_towns':len(towns),'n_edges':0,
                                   'modularity_Q':None,'n_communities':0,'br_median':None})
        continue

    # Build per-capita birth vectors
    vecs = {}
    for t in towns:
        pop = tp[t]
        vec = np.array([tb[t].get(p,0)/pop for p in all_prefs])
        vecs[t] = vec

    # BR similarity matrix → graph (top 15% edges)
    n_t = len(towns)
    S = np.zeros((n_t, n_t))
    for i in range(n_t):
        for j in range(i+1, n_t):
            S[i,j] = S[j,i] = br_similarity(vecs[towns[i]], vecs[towns[j]])

    upper = S[np.triu_indices(n_t, k=1)]
    threshold = np.quantile(upper, 0.85)  # top 15%

    G = nx.Graph()
    for i,t in enumerate(towns): G.add_node(t, pop=tp[t])
    for i in range(n_t):
        for j in range(i+1, n_t):
            if S[i,j] >= threshold:
                G.add_edge(towns[i], towns[j], weight=S[i,j])

    # Louvain
    mod, n_comm = float('nan'), 0
    if G.number_of_edges() > 0:
        try:
            if community_louvain:
                part = community_louvain.best_partition(G, weight='weight', random_state=42)
                mod  = community_louvain.modularity(part, G, weight='weight')
            else:
                comms = list(nx.community.greedy_modularity_communities(G, weight='weight'))
                mod   = nx.community.modularity(G, comms, weight='weight')
                part  = {}
                for i,c in enumerate(comms):
                    for nd in c: part[nd]=i
            n_comm = len(set(part.values()))
        except: pass

    method_b_summary.append({
        'period': pid,
        'n_towns': n_t,
        'n_edges': G.number_of_edges(),
        'modularity_Q': round(mod,4) if not (isinstance(mod,float) and np.isnan(mod)) else None,
        'n_communities': n_comm,
        'br_median': round(float(np.median(upper)),4),
        'br_mean': round(float(np.mean(upper)),4),
    })
    q_str = f"{mod:.4f}" if not np.isnan(mod) else "nan"
    print(f"  [{pid}] n_towns={n_t}  edges={G.number_of_edges()}  "
          f"Q={q_str}  comms={n_comm}  BR_med={np.median(upper):.3f}")

df_b = pd.DataFrame(method_b_summary)
df_b.to_csv(OUT_T/'layer_a_town_method_b.csv', index=False, encoding='utf-8-sig')
print(f"\n  → layer_a_town_method_b.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 方法 C：山地原鄉人口存量時序
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("方法 C：山地原鄉人口存量時序")
print("─"*65)

# For each period, get population of each mountain township
mtn_pop_ts = {tc: [] for tc in CORE_MOUNTAINS}
for pid,_,_ in PERIODS:
    tp = all_tp[pid]
    for tc in CORE_MOUNTAINS:
        mtn_pop_ts[tc].append(tp.get(tc, 0))

# Also compute net change
mtn_pop_df = pd.DataFrame(mtn_pop_ts, index=PIDS)
mtn_pop_df.index.name = 'period'
mtn_pop_df.columns = [MOUNTAIN_TOWN_NAMES.get(c,c) for c in CORE_MOUNTAINS]
print(mtn_pop_df.to_string())

mtn_pop_df.to_csv(OUT_T/'layer_a_town_method_c.csv', encoding='utf-8-sig')
print(f"\n  → layer_a_town_method_c.csv")

# Show net change 2013H1 → 2022H2
print("\n  2013H1 → 2022H2 人口變化：")
for tc in CORE_MOUNTAINS:
    p0 = mtn_pop_ts[tc][0]
    p1 = mtn_pop_ts[tc][-1]
    name = MOUNTAIN_TOWN_NAMES.get(tc, tc)
    delta = p1 - p0
    pct = delta/p0*100 if p0>0 else float('nan')
    print(f"    {name:6s}: {p0:,} → {p1:,}  ({delta:+,}, {pct:+.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 視覺化
# ═══════════════════════════════════════════════════════════════════════════
xs = np.arange(len(PIDS))
CMAP_LINE = plt.cm.tab10

# ── Fig T1：方法A — 山地原鄉入度多樣性 vs 都市鄉鎮 時序 ─────────────────
print("\n繪圖 Fig T1…", end=' ', flush=True)
fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor('#fafafa')

mtn_e  = df_a[df_a.is_mountain==1].groupby('period')['diversity_entropy'].mean().reindex(PIDS)
urb_e  = df_a[df_a.is_mountain==0].groupby('period')['diversity_entropy'].mean().reindex(PIDS)
mtn_g  = df_a[df_a.is_mountain==1].groupby('period')['gini_concentration'].mean().reindex(PIDS)
urb_g  = df_a[df_a.is_mountain==0].groupby('period')['gini_concentration'].mean().reindex(PIDS)

ax1b = ax1.twinx()
ax1.plot(xs, mtn_e.values, 'o-', color='#2e7d32', lw=2, ms=5, label='山地原鄉 Entropy')
ax1.plot(xs, urb_e.values, 's--', color='#c62828', lw=1.5, ms=4, label='都市鄉鎮 Entropy')
ax1b.plot(xs, mtn_g.values, '^-.', color='#1565c0', lw=1.5, ms=4, label='山地原鄉 Gini (右軸)')
ax1b.plot(xs, urb_g.values, 'v:', color='#f57c00', lw=1.5, ms=4, label='都市鄉鎮 Gini (右軸)')

ax1.set_xticks(xs[::2]); ax1.set_xticklabels(PIDS[::2], rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Shannon Entropy（出生縣市多樣性）', fontsize=9)
ax1b.set_ylabel('Gini（集中度，越高=越單一來源）', fontsize=9)
ax1.set_title('方法 A：鄉鎮入度多樣性時序\n山地原鄉 vs 都市鄉鎮（Shannon Entropy & Gini）',
              fontsize=12, fontweight='bold')
lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbls1+lbls2, fontsize=8.5, loc='lower right')
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1b.spines['top'].set_visible(False)
plt.tight_layout()
fig1.savefig(OUT_F/'layer_a_town_fig1_diversity.png', dpi=150, bbox_inches='tight')
plt.close(fig1); print("layer_a_town_fig1_diversity.png")

# ── Fig T2：方法A — 核心山地原鄉的主要來源縣市（2022H2 snapshot）──────────
print("繪圖 Fig T2…", end=' ', flush=True)
pid_snap = '2022H2'
bt_snap = all_bt[pid_snap]
tp_snap = all_tp[pid_snap]
sc_snap = all_sc[pid_snap]

# Bipartite graph: county → mountain township
B = nx.DiGraph()
mtn_all = [t for t in tp_snap if sc_snap.get(t,0)/tp_snap.get(t,1) >= 0.95 and tp_snap[t]>=30]
for tc in mtn_all:
    B.add_node(tc, bipartite=1, name=town_label(tc), pop=tp_snap[tc])
for src in CITY:
    total_out = sum(cnt for (bc,t),cnt in bt_snap.items() if bc==src and t in mtn_all)
    if total_out > 0:
        B.add_node(src, bipartite=0, name=CITY[src])
for (bc,tc), cnt in bt_snap.items():
    if tc in mtn_all and bc in B.nodes():
        B.add_edge(bc, tc, weight=cnt)

# Layout: counties on left, mountain townships on right
county_nodes = [n for n,d in B.nodes(data=True) if d.get('bipartite')==0 and n in B]
town_nodes   = sorted([n for n,d in B.nodes(data=True) if d.get('bipartite')==1 and n in B],
                       key=lambda x: -B.in_degree(x, weight='weight'))

n_c = len(county_nodes); n_t2 = len(town_nodes)
pos = {}
for i, n in enumerate(county_nodes):
    pos[n] = (0, (n_c-1)/2 - i)
for i, n in enumerate(town_nodes):
    pos[n] = (2, (n_t2-1)/2 - i)

fig2, ax2 = plt.subplots(figsize=(14, max(8, n_t2*0.7)))
fig2.patch.set_facecolor('#f5f8f5')

# Draw edges (width by weight)
all_wts = [d['weight'] for _,_,d in B.edges(data=True)]
max_w = max(all_wts, default=1)
for u,v,d in B.edges(data=True):
    w = d['weight']
    lw = 0.3 + 4.0 * w/max_w; alp = 0.15 + 0.7 * w/max_w
    ax2.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
             color='#4a90d9', alpha=alp, lw=lw, zorder=1)

# Draw nodes
for n in county_nodes:
    x,y = pos[n]
    ax2.scatter(x, y, s=120, c='#e53935', edgecolors='white', lw=1, zorder=3)
    ax2.annotate(B.nodes[n]['name'], (x,y), fontsize=8.5, ha='right', va='center',
                 xytext=(-6,0), textcoords='offset points', fontweight='bold')
for n in town_nodes:
    x,y = pos[n]
    pop_n = tp_snap.get(n, 0)
    s = 60 + pop_n*0.18
    ax2.scatter(x, y, s=s, c='#2e7d32', edgecolors='white', lw=1.2, zorder=3, marker='*')
    in_w = B.in_degree(n, weight='weight')
    ax2.annotate(f"{B.nodes[n]['name']} ({pop_n}人)", (x,y), fontsize=8.5,
                 ha='left', va='center', xytext=(6,0), textcoords='offset points', fontweight='bold')

ax2.set_xlim(-1.2, 3.5); ax2.axis('off')
ax2.set_title(f'方法 A：出生縣市 → 山地原鄉 二模式網絡（{pid_snap}）\n'
              '綠★=山地原鄉（大小=Atayal人口），紅●=出生縣市，線寬=人流量',
              fontsize=11, fontweight='bold')
plt.tight_layout()
fig2.savefig(OUT_F/'layer_a_town_fig2_bipartite.png', dpi=150, bbox_inches='tight')
plt.close(fig2); print("layer_a_town_fig2_bipartite.png")

# ── Fig T3：方法A — 復興區歷年主要來源縣市（sankey-style bar stacked）──────
print("繪圖 Fig T3…", end=' ', flush=True)
focus_towns = ['6800013','1000412','1000212','6500029']  # 復興/尖石/大同/烏來
fig3, axs3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.patch.set_facecolor('#fafafa')
top_sources_all = sorted(CITY.keys(), key=lambda x: -sum(
    all_bt[pid].get((x, t), 0) for pid,_,_ in PERIODS for t in focus_towns))[:8]

for ax3, tc in zip(axs3.flat, focus_towns):
    stacked = {}
    for src in top_sources_all:
        stacked[src] = [all_bt[pid].get((src, tc), 0) for pid,_,_ in PERIODS]

    bottom = np.zeros(len(PIDS))
    colors3 = [CMAP_LINE(i/len(top_sources_all)) for i in range(len(top_sources_all))]
    for i, src in enumerate(top_sources_all):
        vals = np.array(stacked[src], dtype=float)
        ax3.bar(xs, vals, bottom=bottom, color=colors3[i], alpha=0.85,
                label=CITY.get(src, src), width=0.8)
        bottom += vals

    ax3.set_xticks(xs[::4]); ax3.set_xticklabels(PIDS[::4], rotation=45, ha='right', fontsize=8)
    ax3.set_title(f'{town_label(tc)} 各期主要來源縣市', fontsize=10, fontweight='bold')
    ax3.set_ylabel('人口數', fontsize=8.5)
    ax3.legend(fontsize=7.5, loc='upper left', ncol=2)
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)

fig3.suptitle('方法 A：四大山地原鄉的出生縣市來源組成（2013H1–2022H2）',
              fontsize=12, fontweight='bold')
plt.tight_layout()
fig3.savefig(OUT_F/'layer_a_town_fig3_source_stacked.png', dpi=150, bbox_inches='tight')
plt.close(fig3); print("layer_a_town_fig3_source_stacked.png")

# ── Fig T4：方法B — BR相似度網絡 Modularity 時序 ─────────────────────────
print("繪圖 Fig T4…", end=' ', flush=True)
fig4, ax4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.patch.set_facecolor('#fafafa')

q_vals = df_b['modularity_Q'].tolist()
n_town_vals = df_b['n_towns'].tolist()
br_med_vals = df_b['br_median'].tolist()
n_comm_vals = df_b['n_communities'].tolist()

ax4[0].plot(xs, q_vals, 'o-', color='#6a1fa2', lw=2, ms=5)
ax4[0].set_xticks(xs[::2]); ax4[0].set_xticklabels(PIDS[::2], rotation=45, ha='right', fontsize=8)
ax4[0].set_ylabel('Louvain Modularity Q', fontsize=9)
ax4[0].set_title('方法 B：BR 相似度網絡 Modularity 時序', fontsize=11, fontweight='bold')
ax4[0].grid(axis='y', alpha=0.3)
ax4b = ax4[0].twinx()
ax4b.plot(xs, n_comm_vals, 's--', color='#e65100', lw=1.5, ms=4, label='# communities (右軸)')
ax4b.set_ylabel('社群數', fontsize=8.5, color='#e65100')
ax4b.legend(fontsize=8)

ax4[1].plot(xs, br_med_vals, 'o-', color='#00695c', lw=2, ms=5, label='BR median')
ax4[1].plot(xs, n_town_vals, 's--', color='#37474f', lw=1.5, ms=4, label='# townships (右軸)')
ax4[1].set_xticks(xs[::2]); ax4[1].set_xticklabels(PIDS[::2], rotation=45, ha='right', fontsize=8)
ax4[1].set_ylabel('BR Similarity Median', fontsize=9)
ax4[1].set_title('方法 B：BR 中位數 & 納入鄉鎮數', fontsize=11, fontweight='bold')
ax4[1].legend(fontsize=8.5)
ax4[1].grid(axis='y', alpha=0.3)

for ax in ax4:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig4.savefig(OUT_F/'layer_a_town_fig4_br_timeseries.png', dpi=150, bbox_inches='tight')
plt.close(fig4); print("layer_a_town_fig4_br_timeseries.png")

# ── Fig T5：方法C — 山地原鄉人口存量時序 ────────────────────────────────
print("繪圖 Fig T5…", end=' ', flush=True)
fig5, ax5 = plt.subplots(figsize=(14, 6))
fig5.patch.set_facecolor('#fafafa')

colors5 = [CMAP_LINE(i/len(CORE_MOUNTAINS)) for i in range(len(CORE_MOUNTAINS))]
for i, tc in enumerate(CORE_MOUNTAINS):
    name = MOUNTAIN_TOWN_NAMES.get(tc, tc)
    vals = mtn_pop_ts[tc]
    ax5.plot(xs, vals, 'o-', color=colors5[i], lw=2.2, ms=5, label=name, alpha=0.87)
    # Annotate endpoint
    ax5.annotate(f"{name}\n{vals[-1]:,}", (xs[-1], vals[-1]),
                 textcoords='offset points', xytext=(5,0), fontsize=7.5, color=colors5[i],
                 va='center')

ax5.set_xticks(xs); ax5.set_xticklabels(PIDS, rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('泰雅族人口（人）', fontsize=9)
ax5.set_title('方法 C：核心山地原鄉泰雅族人口存量時序（2013H1–2022H2）\n'
              '上升=都市流入持續增加（lifelong-stock 記帳法）',
              fontsize=11, fontweight='bold')
ax5.legend(fontsize=8.5, loc='upper left', ncol=2)
ax5.grid(axis='y', alpha=0.3)
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
plt.tight_layout()
fig5.savefig(OUT_F/'layer_a_town_fig5_stock_timeseries.png', dpi=150, bbox_inches='tight')
plt.close(fig5); print("layer_a_town_fig5_stock_timeseries.png")

# ── Fig T6：方法C — 2013H1→2022H2 各山地原鄉成長率橫向比較 ─────────────
print("繪圖 Fig T6…", end=' ', flush=True)
fig6, ax6 = plt.subplots(figsize=(10, 6))
fig6.patch.set_facecolor('#fafafa')

chg_list = []
for tc in CORE_MOUNTAINS:
    p0 = mtn_pop_ts[tc][0]; p1 = mtn_pop_ts[tc][-1]
    name = MOUNTAIN_TOWN_NAMES.get(tc, tc)
    chg_list.append((name, p1-p0, (p1-p0)/p0*100 if p0>0 else 0))
chg_list.sort(key=lambda x: x[1])  # ascending

names6 = [c[0] for c in chg_list]
deltas6 = [c[1] for c in chg_list]
pcts6   = [c[2] for c in chg_list]
bar_cols = ['#c62828' if d < 0 else '#2e7d32' for d in deltas6]
y6 = np.arange(len(names6))

ax6b = ax6.twinx()
ax6.barh(y6, deltas6, color=bar_cols, alpha=0.85, height=0.5, label='絕對人口增量（左軸）')
ax6b.plot(y6, pcts6, 'D-', color='#1565c0', lw=2, ms=7, label='增長率 % (右軸)')

ax6.set_yticks(y6); ax6.set_yticklabels(names6, fontsize=10)
ax6.set_xlabel('人口淨增（2013H1→2022H2）', fontsize=9)
ax6b.set_ylabel('增長率 %', fontsize=9, color='#1565c0')
ax6.set_title('方法 C：山地原鄉泰雅族人口 2013→2022 增長比較',
              fontsize=12, fontweight='bold')
lines6a, lbls6a = ax6.get_legend_handles_labels()
lines6b, lbls6b = ax6b.get_legend_handles_labels()
ax6.legend(lines6a+lines6b, lbls6a+lbls6b, fontsize=9, loc='lower right')
ax6.axvline(0, color='black', lw=0.8, alpha=0.5)
ax6.grid(axis='x', alpha=0.3)
ax6.spines['top'].set_visible(False)
plt.tight_layout()
fig6.savefig(OUT_F/'layer_a_town_fig6_growth.png', dpi=150, bbox_inches='tight')
plt.close(fig6); print("layer_a_town_fig6_growth.png")

print("\n" + "="*65)
print("✅  Layer A Township 三方法完成")
print(f"    表格 → {OUT_T}")
print(f"    圖片 → {OUT_F}")
print("="*65)
