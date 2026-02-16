import os
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans # 引入 KMeans 用於找中心點
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import LocalFileStore
from langchain_core.load import loads as langchain_loads

# ==========================================
# 📊 視覺化設定區 (Config)
# ==========================================
# 選項: 
# "scatter" - 乾淨的點陣圖，標註中心點
# "spider"  - 帶有歸屬連線，強化 Voronoi Cell 視覺感
#VIS_MODE = "scatter"
VIS_MODE = "spider"

# 分群數量 (模擬 Voronoi 區域數)
N_CLUSTERS = 5
# ==========================================

# 0. 定義儲存路徑
VECTOR_DB_PATH = "./faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"
hash_key_name = "doc_hash_id"

def main():
    # 1. 載入資料
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists(VECTOR_DB_PATH):
        print("請先執行匯入程式產生向量庫。")
        return

    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    store = LocalFileStore(BYTE_STORE_PATH)

    # 2. 提取所有向量
    all_doc_ids = list(vectorstore.index_to_docstore_id.values())
    vectors = []
    metadata_list = []

    for doc_id in all_doc_ids:
        child_doc = vectorstore.docstore.search(doc_id)
        if child_doc.page_content == "init": continue
        
        vec = vectorstore.index.reconstruct(all_doc_ids.index(doc_id))
        vectors.append(vec)
        
        # 溯源 Parent 資訊
        p_hash = child_doc.metadata.get(hash_key_name)
        source_file = "未知"
        if p_hash:
            p_bytes = store.mget([p_hash])[0]
            if p_bytes:
                p_doc = langchain_loads(p_bytes.decode('utf-8'))
                source_file = os.path.basename(p_doc.metadata.get("source", "未知"))

        metadata_list.append({
            "Content": child_doc.page_content[:40],
            "Type": "Data Node",
            "Source": source_file
        })

    vectors_np = np.array(vectors)

    # 3. 計算中心點 (模擬 Voronoi Cells 的核心)
    # 假設我們將資料分為 N_CLUSTERS (5) 個區域 (n_clusters 可依資料量調整)
    n_clusters = min(N_CLUSTERS, len(vectors_np)) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vectors_np)
    centroids = kmeans.cluster_centers_

    # 4. 合併數據進行 PCA 降維 (包含數據點與中心點)
    total_vectors = np.vstack([vectors_np, centroids])
    pca = PCA(n_components=2)
    total_2d = pca.fit_transform(total_vectors)
    # 前 len(vectors_np) 個是原始數據點，後面的是中心點
    data_2d = total_2d[:len(vectors_np)]
    centroids_2d = total_2d[len(vectors_np):]

    # 5. 根據 Config 進行繪圖
    if VIS_MODE == "scatter":
        draw_scatter(data_2d, centroids_2d, cluster_labels, metadata_list, n_clusters)
    else:
        draw_spider(data_2d, centroids_2d, cluster_labels, metadata_list, n_clusters)

def draw_scatter(data_2d, centroids_2d, labels, meta, n_clusters):
    """模式 1: 傳統散佈圖 (使用 plotly.express)"""
    df_data = pd.DataFrame(data_2d, columns=['x', 'y'])
    df_data['Content'] = [m['Content'] for m in meta]
    df_data['Source'] = [m['Source'] for m in meta]
    df_data['Type'] = "Database Vector"
    df_data['Cluster'] = [f"Cell {l}" for l in labels]      # 標註屬於哪個區域 (Cell 0, Cell 1, ...)

    df_centroids = pd.DataFrame(centroids_2d, columns=['x', 'y'])
    df_centroids['Content'] = [f"Centroid {i}" for i in range(n_clusters)]
    df_centroids['Source'] = "N/A"
    df_centroids['Type'] = "Centroid"
    df_centroids['Cluster'] = [f"Cell {i}" for i in range(n_clusters)]

    df_final = pd.concat([df_data, df_centroids])
    fig = px.scatter(
        df_final, x='x', y='y', 
        color='Cluster', symbol='Type',    # 根據「群組」上色 , 根據「類型」使用不同符號 (數據點 vs 中心點)
        hover_data=['Content', 'Source'], 
        title="RAG 向量空間：中心點散佈圖"
    )
    fig.update_traces(
        # 「Traces」在 Plotly 中代表圖表上的每一組資料. 可以繞過全局設定，去修改特定的 trace。
        # size 是指 marker 的尺寸. 
        # 這裡是針對「Voronoi Centroid (地標)」這個類型的 trace 進行修改. (修改marker的屬性)
        # ex. marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), # 設定標記的大小、邊框等屬性
        marker=dict(size=15), 
        # 這裡的 selector 是根據我們在 DataFrame 中設定的 Type 欄位來選擇要修改的 trace.
        selector=dict(name='Centroid'))
    fig.show()

def draw_spider(data_2d, centroids_2d, labels, meta, n_clusters):
    """模式 2: 帶連線的蜘蛛圖 (使用 plotly.graph_objects)"""
    fig = go.Figure()

    # 1. 畫連線 (Spider Lines) - 使用 None 斷開技巧以提升效能
    line_x, line_y = [], []
    for i in range(len(data_2d)):
        c_idx = labels[i]       # 找到該資料點所屬的群集 (Cell) 索引
        # 每個資料點連線到它的中心點 (Voronoi Cell 的核心), 用None斷開以提升繪圖效能
        line_x.extend([data_2d[i, 0], centroids_2d[c_idx, 0], None])
        line_y.extend([data_2d[i, 1], centroids_2d[c_idx, 1], None])

    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.2)', width=1),
        hoverinfo='none', name='歸屬連線', showlegend=False
    ))

    # 2. 畫資料點 (Data Nodes)
    for i in range(n_clusters):
        # 這是一行 NumPy 的進階索引語法。
        # 原理：labels 存的是每個點的群組編號（如 [0, 1, 0, 2...]）。當 i 為 0 時，mask 會變成一個布林陣列
        # （如 [True, False, True, False...]）。
        # 目的：讓後面的 data_2d[mask, 0] 只抓出屬於「第 i 群」的座標。
        mask = (labels == i)

        # 在 Plotly 的底層邏輯中，每一組 add_trace 就像是在畫布上疊加透明投影片。
        # 我們為每個分群都建立一個獨立的 Scatter 物件，這樣才能針對不同分群進行個別控制。(按右邊的圖例可以開啟/關閉這一群的顯示)
        fig.add_trace(go.Scatter(
            x=data_2d[mask, 0], y=data_2d[mask, 1], # 取出該群組所有點的 X 座標（PCA 第一主成分）與 Y 座標（PCA 第二主成分）。
            mode='markers',                         # 這一層只要畫「點」，不要把點連起來。
            name=f'Cell {i}',                       # 顯示在圖表右側的圖例文字。
            marker=dict(size=8, opacity=0.8),       # 設定 20% 的透明度，這樣當多個點重疊時，顏色會變深，方便觀察資料密度。
            # 從 meta 列表（包含來源檔案與內容預覽）中，"挑選出"屬於目前這群的資料。
            text=[f"來源: {m['Source']}<br>內容: {m['Content']}" for j, m in enumerate(meta) if labels[j] == i],
            hoverinfo='text'                        # 不要顯示 X, Y 座標數值，只顯示我們自定義的 text 內容。
        ))

    # 3. 畫中心點 (Centroids)
    fig.add_trace(go.Scatter(
        x=centroids_2d[:, 0], y=centroids_2d[:, 1], mode='markers',
        marker=dict(size=18, symbol='x', line=dict(width=2, color='black')),
        name='Voronoi 地標', hoverinfo='name'
    ))

    fig.update_layout(title="RAG 向量空間：蜘蛛連線圖 (Spider Plot)",
                      xaxis_title="PCA 1", yaxis_title="PCA 2",
                      template="plotly_white", hovermode="closest")
    fig.show()

if __name__ == "__main__":
    main()
