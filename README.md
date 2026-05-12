# atayal-migration-network

以社會網絡分析探索泰雅族當代遷徙網絡的結構特徵。

NTNU Social Network Analysis, Dr. Chun-Hsiang Chan, 2026 Spring.

## Overview

本研究結合 TICD 與 TIPsMigDynamics 資料集，
以有向加權圖建構泰雅族的遷徙網絡，
計算 centrality、clustering、community detection 等指標，
並以 GeoPandas 進行空間視覺化，
探討泰雅族當代人口流動中的核心節點、社群結構與網絡拓撲特徵。

## Research Questions

1. 哪些部落或地點在遷徙流中扮演 hub 或 bridge 角色？
2. 遷徙網絡是否形成穩定的社群群集？是否對應地理區位？
3. 整體網絡是否呈現中心化、群聚化或核心邊陲結構？

## Data Sources

| 資料集 | 說明 | 連結 |
|--------|------|------|
| TICD | 部落人口、座標、族群比例 | [OSF](https://osf.io/esw67/) |
| TIPsMigDynamics | 點對點遷徙動態 | [政大學術集成](https://ah.lib.nccu.edu.tw/item?item_id=171103) 、[OSF](https://osf.io/6rpz9/)|
| GIS 點位 | 部落空間定位 | [depositar](https://data.depositar.io/dataset/rf03-1) |

原始資料因授權限制未上傳，請依上方連結自行下載，放入 `data/raw/` 對應子資料夾。

## Project Structure

```
data/
  raw/                原始下載檔案（不上傳至 GitHub）
  processed/          清理後的中間資料
notebooks/            分析流程（按編號順序執行）
  01_data_cleaning.ipynb
  02_network_construction.ipynb
  03_centrality_analysis.ipynb
  04_community_detection.ipynb
  05_spatial_visualization.ipynb
  06_cross_period_comparison.ipynb
src/                  可重複使用的 Python 函式
outputs/
  figures/            匯出圖檔
  tables/             匯出表格
  interactive/        pyVis 互動式 HTML
docs/
  proposal/           proposal 文件
  references/         文獻筆記
  slides/             簡報檔案
reports/              期末報告
```

## How to Reproduce

```bash
git clone https://github.com/YOUR_USERNAME/atayal-migration-network.git
cd atayal-migration-network
pip install -r requirements.txt
# 下載原始資料至 data/raw/（見 data/README.md）
# 依序執行 notebooks/01 至 06
```

## Requirements

- Python 3.10+
- pandas, networkx, geopandas, matplotlib, python-louvain, pyvis, shapely

完整套件清單見 `requirements.txt`。

## Team

| 姓名 | 學號 | 負責 |
|------|------|------|
| 賴亭穎 | NTU B12303147 | 文獻整合、空間視覺化 |
| 黃凡嘉 | NTU B11701165 | 資料處理、網絡分析 |

## References

見 [docs/proposal/](docs/proposal/) 中的完整參考文獻列表。

核心文獻：
- 廖守臣 (1984). 泰雅族的文化：部落遷徙與拓展.
- 藤井志津枝 (1997). 理蕃：日本治理台灣的計策.
- Granovetter, M. S. (1973). The strength of weak ties.
- Mills, B. J. et al. (2018). Evaluating Chaco migration scenarios using dynamic SNA.
- Wasserman, S. & Faust, K. (1994). Social Network Analysis: Methods and Applications.

## License

MIT License. 資料集各有其授權條款，請參閱原始來源。
