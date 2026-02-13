import os
import numpy as np

# Langchain refs : 
#   https://reference.langchain.com/python/langchain/
#   https://docs.langchain.com/oss/python/integrations/providers/overview
#   https://reference.langchain.com/python/integrations/
#   https://docs.langchain.com/oss/python/integrations/vectorstores
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
chunks = text_splitter.split_text(raw_text)

# 將 page(chunk)對應的向量在 FAISS 裡的 index 儲存在 metadata裡，方便後續使用vector_db.index.reconstruct()取回該向量。
# 由於vector_db.similarity_search_with_score()會傳回document物件, 裡頭會有page(或叫做chunk)的內容, 不用擔心拿不到原始chunk內容, 
# 所以我們下面這程式碼中, 建立metadata時, 重點在存入FAISS的 index (int) 而不是docstore裡對應該向量的chunk的ID(或是index), 
# 但是其實他們兩者的ID(index)是相同的, 因為我們的FAISS與docstore都是從零開始建立。
docs = [
    # 在 Python 中，只要是在 括號 []、花括號 {} 或圓括號 () 內部，你都可以隨意換行，不需要加任何特殊符號（如 \）。
    # 所以下面這一行可以變成: (相同縮排)
    #    Document(page_content=t, metadata={"original_index": i}) 
    #    for i, t in enumerate(chunks)
    # 甚至是:
    #    Document(
    #        page_content=text, 
    #        metadata={
    #            "original_index": i,
    #            ...
    #        }
    #    ) 
    #    for i, text in enumerate(raw_chunks)
    Document(page_content=t, metadata={"original_index": i}) for i, t in enumerate(chunks)
]

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

# 這邊的 FAISS 是 Langchain 封裝的版本，內部會維護一個 docstore 來對應原始文字與 FAISS 的向量索引。
# FAISS 本身只存向量座標，不存原始文字。因此 LangChain 在內部維護了一個 docstore（通常是個字典）來對應文字內容。
# 
# vector_db.index_to_docstore_id：
#   一個映射表（Mapping），Key 是 FAISS 內部的整數索引（0, 1, 2...），Value 是 docstore 裡chunk的 UUID（唯一識別碼,不是index）。
# vector_db.index.reconstruct(i) :
#   vector_db.index 是底層的 FAISS 索引物件，reconstruct 方法會根據FAISS內部索引位置 i 來將儲存在壓縮
#   或優化後的索引裡的向量座標「還原」出來。
#   回傳值：一個 Numpy Array，維度與當初使用的 Embedding 模型一致（如 1536 維）。  
# vector_db.docstore._dict.keys()：
#   這是直接存取 docstore 所有的 ID 列表。
#
# 方法/屬性	                 需要的參數 (Input)	      給出的結果 (Output)	  作用層次
# index_to_docstore_id[n]	FAISS 物理編號 (int)	UUID (string)	       中間層 (Mapping)
# index.reconstruct(n)	    FAISS 物理編號 (int)	原始向量 (list/array)	底層 (C++)
# docstore.search(uuid)	    UUID (string)	       Document 物件	      內容層 (Storage)

print(f"=== [檢索結果 (Top 2)] ===")
for i, (doc, score) in enumerate(results_with_score):
    # doc: Document 物件包含原文（page_content）與元數據（metadata）.
    
    # 注意：這裡的 i 是搜尋結果的排名 (0, 1)，並非資料庫的原始 index.
    #      嚴謹做法應透過 docstore_id 取得正確的原始向量

    # 從 Metadata 拿到我們埋進去的「docstore index」
    real_index = doc.metadata["original_index"]

    # 獲取該 chunk 的原始向量 (從底層 FAISS 指標重建)
    # 註：FAISS index 儲存的順序可能與搜尋結果順序不同，搜尋結果是按 score 排序
    # 我們需要透過檢索到的內容去比對向量
    doc_vector = vector_db.index.reconstruct(real_index) 

    print(f"排名 {i+1}:")
    print(f"  - 信心分數 (L2 距離): {score:.4f}")
    print(f"  - 文字片段 (前後各3字): 「{doc.page_content[:3]} ... {doc.page_content[-3:]}」")
    print(f"  - FAISS Index: {i}")
    print(f"  - Embedding 前兩維: {doc_vector[:2]} ... 末兩維: {doc_vector[-2:]}")
    print("-" * 30)