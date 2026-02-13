# Need install pandas, plotly.
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. 準備資料與 Embedding
raw_chunks = [
    "凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。",
    "旅遊地點不限國內外，但須於出發前兩週提交申請單。",
    "報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。",
    "若年資不滿一年，補助金額按比例計算（每月 2,500 元）。"
]

# 使用我們討論過的「正規」Metadata 注入
docs = [
    Document(page_content=t, metadata={"original_index": i}) 
    for i, t in enumerate(raw_chunks)
]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.from_documents(docs, embeddings_model)

# 2. 測試問題與向量提取
queries = ["入職半年可以領多少？", "報帳發票規定", "今天天氣不錯"]
query_vectors = [embeddings_model.embed_query(q) for q in queries]
chunk_vectors = [vector_db.index.reconstruct(i) for i in range(len(raw_chunks))]

# 3. PCA 降維 (1536 -> 2)
all_vectors = np.vstack([chunk_vectors, query_vectors])
pca = PCA(n_components=2)
coords = pca.fit_transform(all_vectors)

# 4. 整理成 DataFrame (Plotly 繪圖最方便的格式)
df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "內容": raw_chunks + queries,
    "類型": ["資料庫 (Chunk)"] * len(raw_chunks) + ["測試問題 (Query)"] * len(queries),
    "編號": [f"Index {i}" for i in range(len(raw_chunks))] + ["Query"] * len(queries)
})

# 5. 繪製互動圖表
fig = px.scatter(
    df, x="x", y="y", 
    color="類型", 
    text="編號",           # 在點旁邊顯示編號
    hover_data=["內容"],    # 滑鼠移上去顯示完整內容
    title="RAG 語意空間互動地圖 (PCA 降維)",
    template="plotly_white"
)

# 調整點的大小與標籤位置
fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))
fig.show()