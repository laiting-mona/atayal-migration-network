# 工作概要

**\- Data preprocessing and Resource reading**  
 A. 下載 TIPD 和 TICD 資料集、處理百年地圖，探索資料結構  
	1\) 確認哪些欄位可以直接轉換為 edge list  
	2\) 處理地點名稱的不一致問題  
	3\) 嘗試將 TIPD 的村里層級地點映射回 TICD 的部落層級  
 B. 完成研究背景與動機的完整文字稿  
	精讀廖守臣、藤井志津枝、Granovetter、Mills et al. 等文獻  
\-\> 確認資料的可用性與限制，共同決定最終的節點定義和時間切片方法？

**\- model**  
A. 用 NetworkX 建構各時段的 directed weighted network  
	計算 in-degree、out-degree、betweenness centrality、eigenvector centrality、clustering coefficient、Freeman centralization  
B. 跑 Louvain community detection，計算 modularity  
	把所有指標整理成表格，每個時段一張  
C. 開始用 GeoPandas 做第一版的空間視覺化  
協助檢查分析結果是否合理（如 in-degree 最高的節點是不是確實對應到已知的都會區或重要聚落）  
D. 撰寫研究目的、研究方法  
\-\> 檢視所有分析結果，共同討論哪些發現值得在報告中重點呈現，哪些結果需要進一步分析

**\- visualition**  
A. 完成最終版的視覺化圖表  
	各時段的網絡圖、centrality 分布圖、community detection 結果疊套在台灣地形圖上的空間視覺化  
	撰寫分析結果  
B. 撰寫討論與結論，將分析結果連結回研究問題和理論框架  
	例如 community detection 的結果是否對應到傳統的 mulaxen galang 範圍  
	clustering coefficient 的數值是否支持泰雅族低合作性的假說  
整合全文，確保各章節之間的邏輯銜接通順

**\- report**  
交叉審閱對方負責的parts，互相提出修改建議  
製作簡報

# 分工與期程

**4/18 \- 4/24**  
黃：  
下載 TIPD 和 TICD 資料集、處理百年地圖  
用 NetworkX 建構各時段的 directed weighted network  
賴：  
查找並閱讀文獻  
完成研究背景與動機的完整文字稿

**4/25 \- 5/10**  
黃：  
跑 Louvain community detection，計算 modularity  
賴：  
開始用 GeoPandas 做第一版的空間視覺化

**5/15 \- 5/22**  
黃：  
撰寫研究目的、研究方法  
賴：  
完成最終版的視覺化圖表  
待議：  
撰寫討論與結論

**5/23 \- 5/29**  
待議：  
對邏輯、完成簡報