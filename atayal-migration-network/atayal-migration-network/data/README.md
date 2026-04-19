# Data Sources

原始資料因授權限制或檔案大小未上傳至 GitHub。
請依以下說明自行下載，放入對應子資料夾。

## data/raw/timd/

**TIPsMigDynamics：台灣原住民族各族群遷徙動態**

- 來源：政大學術集成
- 連結：https://ah.lib.nccu.edu.tw/item?item_id=171103
- 說明：根據人口動態資料建立的點對點遷徙動態，涵蓋初級、回流與連續遷徙。
  點位為村里人口重心座標。
- 下載後放入 `data/raw/timd/`

## data/raw/ticd/

**TICD：Taiwan Indigenous Communities Open Data**

- 來源：OSF
- 連結：https://osf.io/esw67/
- 說明：提供各部落的總人口數、性比率、教育程度、年齡結構、
  婚姻狀態、各族群人口比例、部落人口重心座標。
- 主要欄位：tribe_name, county, population, lat, lon, atayal_ratio ...
- 下載後放入 `data/raw/ticd/`

## data/raw/gis/

**中研院地理資訊科學研究中心：原住民部落遷徙與分布**

- 來源：depositar
- 連結：https://data.depositar.io/dataset/rf03-1
- 說明：GIS 點位資料，提供原住民部落的空間定位。

**台灣縣市行政區邊界 shapefile**

- 來源：政府資料開放平台
- 連結：https://data.gov.tw/dataset/7442
- 說明：用作 GeoPandas 底圖。下載「直轄市、縣市界線」shapefile。

## data/processed/

清理後的中間資料，由 `notebooks/01_data_cleaning.ipynb` 生成。

| 檔案 | 說明 |
|------|------|
| edge_list.csv | 彙總後的遷徙邊列表 (source, target, weight, period) |
| node_attributes.csv | 節點屬性表 (node, lat, lon, county, subgroup, population) |
| name_mapping.csv | 地名標準化對照表 (raw_name, standardized_name) |
