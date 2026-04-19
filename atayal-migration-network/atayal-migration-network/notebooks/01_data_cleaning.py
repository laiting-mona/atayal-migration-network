# %% [markdown]
# # 01 Data Cleaning
# 載入 TIMD 與 TICD 原始資料，清理地名、篩選泰雅族相關遷徙流，
# 輸出標準化的 edge_list.csv 與 node_attributes.csv。

# %%
import pandas as pd
import sys; sys.path.append("..")
from src.network_builder import aggregate_edges

# %% [markdown]
# ## 1.1 載入 TIMD 遷徙資料
# %%
# df_timd = pd.read_csv("../data/raw/timd/YOUR_FILE.csv")
# df_timd.head()

# %% [markdown]
# ## 1.2 載入 TICD 部落資料
# %%
# df_ticd = pd.read_csv("../data/raw/ticd/YOUR_FILE.csv")

# %% [markdown]
# ## 1.3 地名標準化
# %%
# name_map = pd.read_csv("../data/processed/name_mapping.csv")
# df_timd["origin"] = df_timd["origin"].replace(dict(zip(name_map["raw_name"], name_map["standardized_name"])))

# %% [markdown]
# ## 1.4 篩選泰雅族相關遷徙流
# %%
# atayal_counties = ["新北市","桃園市","新竹縣","苗栗縣","臺中市","南投縣","宜蘭縣"]
# df_atayal = df_timd[df_timd["origin_county"].isin(atayal_counties) | df_timd["dest_county"].isin(atayal_counties)]

# %% [markdown]
# ## 1.5 彙總為 edge list 並輸出
# %%
# edges = aggregate_edges(df_atayal, "origin", "destination", "count")
# edges.to_csv("../data/processed/edge_list.csv", index=False)

# %% [markdown]
# ## 1.6 整理節點屬性表並輸出
# %%
# node_attrs = df_ticd[["tribe_name","lat","lon","county","population"]]
# node_attrs.to_csv("../data/processed/node_attributes.csv", index=False)
