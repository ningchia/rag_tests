# Need install matplotlib and scikit-learn more.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. 準備資料與建立向量庫 (沿用 original_index 邏輯)
#    這邊先不用RecursiveCharacterTextSplitter, 直接手工切段, 因為我們的重點是觀察向量分布, 不需要額外的分段邏輯干擾.
raw_chunks = [
    "年資滿一年之員工，可享有新台幣 30,000 元之旅遊補助。",
    "旅遊地點不限國內外，但須於出發前兩週提交申請單。",
    "報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。",
    "若年資不滿一年，補助金額按比例計算（每月 2,500 元）。"
]

# 一樣, 建立metadata時, 希望存入FAISS的 index (int) 以便使用vector_db.index.reconstruct(i)重建向量, 
# 而不是docstore裡對應該向量的chunk的ID(或是index), 
# 但是其實他們兩者的ID(index)是相同的, 因為我們的FAISS與docstore都是從零開始建立。
docs = [
    Document(page_content=t, metadata={"original_index": i}) 
    for i, t in enumerate(raw_chunks)
]
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.from_documents(docs, embeddings_model)

# 2. 準備要觀察的測試問題 (Query)
queries = ["我入職半年可以領多少？", "報帳需要發票嗎？", "今天天氣真好"]
query_vectors = [embeddings_model.embed_query(q) for q in queries]

# 3. 提取所有 Chunk 的原始向量 (利用 reconstruct)
chunk_vectors = [vector_db.index.reconstruct(i) for i in range(len(raw_chunks))]

# 4. 合併所有向量進行 PCA 降維 (1536維 -> 2維)
#    這裡的 + 號並不是 Numpy 的向量相加，而是 Python 原生 List 的「拼接（Concatenate）」。
#    如果你想更明確地表達「合併矩陣」的意圖，通常會用 np.vstack（Vertical Stack）：
#    正規寫法 : all_vectors = np.vstack([chunk_vectors, query_vectors])
all_vectors = np.array(chunk_vectors + query_vectors)   # 簡化寫法. 應該用 np.vstack 會更清楚.
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(all_vectors)

# 5. 製作「前後3字」的標籤清單
def make_snippet(text):
    if len(text) <= 6:
        return text
    return f"{text[:3]}...{text[-3:]}"

# 產生所有標籤(Chunk + Query) as 原字串的前後各3個字, 中間用...接起
plot_labels = [make_snippet(t) for t in raw_chunks] + [make_snippet(q) for q in queries]

# 6. 繪圖
plt.figure(figsize=(12, 8))

# 解決中文顯示問題
# 有可能 Ubuntu/WSL 裡面根本沒裝中文字體. 使用 apt 安裝 fonts-noto-cjk 或 fonts-wqy-microhei 就可以了.
# WSL 有時候雖然安裝了字體，但 Matplotlib 的字體快取（Cache）沒更新，會導致它說找不到。
# 如果安裝了還是不行，請執行 import shutil; shutil.rmtree(matplotlib.get_cachedir()). 通常是不用的.
# 定義字體優先順序清單:
font_list = [
    'Microsoft YaHei',      # Windows 11
    'Noto Sans CJK TC',     # Ubuntu / Linux (常見於 Google 系列)
    'WenQuanYi Micro Hei',  # Ubuntu / Linux (經典開源黑體)
    'Arial Unicode MS',     # Mac / 通用
    'SimHei',               # 萬用黑體 (許多 Linux 伺服器會裝)
    'sans-serif'            # 最後的防線
]
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False      # 解決座標軸負號顯示為方塊的問題 (這行通常也要加)

# 畫出 DB Chunks
plt.scatter(reduced_vectors[:4, 0], reduced_vectors[:4, 1], c='blue', label='DB Chunks', s=150, alpha=0.6)
# 畫出 Queries
plt.scatter(reduced_vectors[4:, 0], reduced_vectors[4:, 1], c='red', label='Queries', marker='x', s=150)

# 標註文字
for i, label in enumerate(plot_labels):
    plt.annotate(
        label, 
        (reduced_vectors[i, 0], reduced_vectors[i, 1]),
        xytext=(5, 5), 
        textcoords='offset points',
        fontsize=9
    )

plt.title("Semantic Space Snippet Visualization")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()