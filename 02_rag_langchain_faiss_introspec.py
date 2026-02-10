import os
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 準備資料
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

# 2. 分段
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = [Document(page_content=t) for t in text_splitter.split_text(raw_text)]

# 3. 建立向量資料庫 (明確指定模型)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.from_documents(docs, embeddings)

# --- 實驗開始 ---

query = "我入職半年，可以領多少旅遊補助？"

# A. 計算 User Prompt 的 Embedding 並顯示維度
query_vector = embeddings.embed_query(query)
print(f"=== [User Prompt 資訊] ===")
print(f"Prompt: {query}")
print(f"向量維度: {len(query_vector)}")
print(f"向量前兩維: {query_vector[:2]} ... 末兩維: {query_vector[-2:]}\n")

# B. 進行相似度搜尋 (取得結果與距離 Score)
# similarity_search_with_score 回傳的是 (doc, L2_distance). k=2代表找"兩個"最相近的.
# 注意：FAISS 預設返回的是 L2 距離（歐氏距離的平方），數值越小代表越相似.
# L2 距離 < 0.5：通常是極度相關（幾乎是原句）。
# L2 距離 0.5 ~ 0.8：語意相關，但可能有語氣轉換。
# L2 距離 > 1.0：基本上就是「沒關聯」或是「負樣本」。
results_with_score = vector_db.similarity_search_with_score(query, k=2)

print(f"=== [檢索結果 (Top 2)] ===")
for i, (doc, score) in enumerate(results_with_score):
    # 取得該 doc 在 FAISS 內部的 index
    # 在 LangChain 的 FAISS 實作中，doc 的 metadata 或內部的 docstore_id 可與 index 對應
    # 這裡我們直接從 FAISS 索引中提取該向量來顯示
    doc_id = vector_db.index_to_docstore_id[i] # 簡易獲取方式
    
    # 獲取該 chunk 的原始向量 (從底層 FAISS 指標重建)
    # 註：FAISS index 儲存的順序可能與搜尋結果順序不同，搜尋結果是按 score 排序
    # 我們需要透過檢索到的內容去比對向量
    doc_vector = vector_db.index.reconstruct(i) 

    print(f"排名 {i+1}:")
    print(f"  - 信心分數 (L2 距離): {score:.4f}")
    print(f"  - 文字片段 (前後各3字): 「{doc.page_content[:3]} ... {doc.page_content[-3:]}」")
    print(f"  - FAISS Index: {i}")
    print(f"  - Embedding 前兩維: {doc_vector[:2]} ... 末兩維: {doc_vector[-2:]}")
    print("-" * 30)