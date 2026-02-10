import os
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 準備實驗文本
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

# 定義 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 測試不同分段大小 (Improvement 2)
def create_db(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs = [Document(page_content=t) for t in splitter.split_text(text)]
    return FAISS.from_documents(docs, embeddings)

db_small = create_db(raw_text, 100, 20)  # 精細分段
db_large = create_db(raw_text, 500, 50)  # 粗略分段（整篇）

# 3. 帶有門檻值的檢索函數 (Improvement 1)
# L2 距離 < 0.5：通常是極度相關（幾乎是原句）。
# L2 距離 0.5 ~ 0.8：語意相關，但可能有語氣轉換。
# L2 距離 > 1.0：基本上就是「沒關聯」或是「負樣本」。
def run_experiment(query, db, threshold=0.9):
    print(f"【測試問題】: {query}")
    
    # 1. 顯示 User Prompt 的 Embedding 資訊
    q_vector = embeddings.embed_query(query)
    print(f"Prompt 向量維度: {len(q_vector)}")
    print(f"Prompt 向量首尾維度: [{q_vector[0]:.4f}, {q_vector[1]:.4f} ... {q_vector[-2]:.4f}, {q_vector[-1]:.4f}]")
    
    # 2. 檢索並顯示詳細 Chunk 資訊
    # similarity_search_with_score 回傳的是 (doc, L2_distance). L2 距離越小越相似.
    results = db.similarity_search_with_score(query, k=1)   # 取得最相似的 1 筆
    doc, score = results[0]
    
    # 取得該片段在 FAISS 中的向量與 Index
    # 我們透過內部 docstore_id 來反查 index
    doc_id = list(db.docstore._dict.keys())[0] # 簡化示範，取搜尋到的 ID
    # 這裡直接從向量索引中重建向量 (需注意 FAISS 內部的 ID 管理)
    doc_vector = db.index.reconstruct(0) # 示範重建第一個 index 的向量
    
    print(f"\n檢索結果分析:")
    print(f"  - 信心分數 (L2 距離): {score:.4f}")
    
    if score > threshold:
        print(f"  - 判定結果: [ 門檻攔截 ] 距離 {score:.4f} > {threshold}，判定為未知。")
    else:
        print(f"  - 判定結果: [ 通過 ] 距離低於門檻。")
        print(f"  - 片段前後5字: 「{doc.page_content[:5]} ... {doc.page_content[-5:]}」")
        print(f"  - Chunk 向量首尾: [{doc_vector[0]:.4f}, {doc_vector[1]:.4f} ... {doc_vector[-2]:.4f}, {doc_vector[-1]:.4f}]")

# --- 實驗展示 ---

# 測試 1: 相關問題 (應低於門檻)
print("=== 實驗 A: 相關問題 ===")
run_experiment("入職半年的補助是多少？", db_small, threshold=0.9)

# 測試 2: 無關問題 (應觸發門檻值) - 這就是「易負樣本」
print("\n=== 實驗 B: 完全無關的問題 ===")
run_experiment("如何修理太空梭？", db_small, threshold=0.9) # 測試門檻攔截

# 測試 3: 分段大小比較
print("\n=== 實驗 C: 分段大小對分數的影響 ===")
q = "旅遊報帳要幾天內完成？"
_, score_s = db_small.similarity_search_with_score(q, k=1)[0]
_, score_l = db_large.similarity_search_with_score(q, k=1)[0]
print(f"小分段 (100字) 的 L2 分數: {score_s:.4f}")
print(f"大分段 (500字) 的 L2 分數: {score_l:.4f}")