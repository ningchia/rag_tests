# 測試問題 : 銀行在調高持卡人信用額度後會由何動作？

import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore
from langchain_core.load import loads as langchain_loads

# 0. 定義儲存路徑 (需與之前的程式一致)
VECTOR_DB_PATH = "./faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"
hash_key_name = "doc_hash_id"

def main():
    # 1. 初始化組件與載入現有資料庫
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if not os.path.exists(VECTOR_DB_PATH):
        print("錯誤：找不到向量庫路徑。請先執行匯入程式。")
        return

    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    store = LocalFileStore(BYTE_STORE_PATH)

    # 建立 Retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=hash_key_name,
    )

    # 可以告訴檢索器：每次搜尋時，請先找回前 6 個最相關的子向量 (預設通常是 4)
    # retriever.search_kwargs = {"k": 6}

    # 2. 設定 RAG Chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    template = """請根據以下提供的上下文回答問題。如果你不知道答案，就說你不知道，不要編造答案。

上下文：
{context}

問題：
{question}

回答："""
    prompt = ChatPromptTemplate.from_template(template)

    # 定義一個自定義函式來處理檢索結果並印出資訊
    def inspect_and_format_docs(docs):
        print("\n" + "-"*30)
        print(f" 🔍 [檢索到 {len(docs)} 筆相關區塊]")
        
        formatted_contents = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            # 格式化內容預覽：前3字...後3字
            preview = f"{content[:3]}...{content[-3:]}" if len(content) > 6 else content
            # 取得 Metadata 資訊
            source = doc.metadata.get("source", "未知來源")
            chunk_idx = doc.metadata.get("chunk_index", "N/A")
            
            print(f" {i+1}. 來源: {os.path.basename(source)} (區塊 {chunk_idx}) | 預覽: {preview}")
            formatted_contents.append(content)
            
        print("-"*30 + "\n")
        return "\n\n".join(formatted_contents)
        
    # 修改 RAG Chain，將 format_docs 改成我們的 inspect_and_format_docs
    rag_chain = (
        {
            "context": retriever | inspect_and_format_docs, # 這裡會先印出資訊再傳給 LLM
            # 使用 .as_retriever(search_kwargs={"k": 6}) 動態指定數量
            # "context": retriever.vectorstore.as_retriever(search_kwargs={"k": 6}) | inspect_and_format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 3. 多行輸入問答迴圈
    print("\n" + "="*50)
    print("歡迎使用 RAG 問答系統 (輸入空白行結束輸入並提交)")
    print("="*50)

    while True:
        print("\n請輸入您的問題 (直接按 Enter 結束輸入):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        query = "\n".join(lines)
        if not query.strip():
            print("程式結束。")
            break

        print("\n[檢索中並產生回答...]")
        # 取得回答
        response = rag_chain.invoke(query)
        print(f"\nAI 回答：\n{response}")

        # 4. 視覺化部分：準備向量數據
        print("\n[正在產生向量空間視覺化圖表...]")
        
        # 獲取所有子向量
        all_doc_ids = list(vectorstore.index_to_docstore_id.values())
        vectors = []
        metadata_list = []

        for doc_id in all_doc_ids:
            # 從 FAISS 的 docstore 取得 child doc
            child_doc = vectorstore.docstore.search(doc_id)
            if child_doc.page_content == "init": continue
            
            # A. 取得子向量與基本資訊
            vec = vectorstore.index.reconstruct(all_doc_ids.index(doc_id))
            vectors.append(vec)

            # B. 取得關聯的 Parent 資訊
            p_hash = child_doc.metadata.get(hash_key_name)
            parent_info = "無關聯"
            source_file = "未知"
            
            if p_hash:
                parent_bytes = store.mget([p_hash])[0]
                if parent_bytes:
                    # 使用之前學會的 langchain_loads
                    p_doc = langchain_loads(parent_bytes.decode('utf-8'))
                    p_content = p_doc.page_content.strip()
                    parent_info = f"{p_content[:3]}...{p_content[-3:]}" if len(p_content) > 6 else p_content
                    source_file = os.path.basename(p_doc.metadata.get("source", "未知"))

            metadata_list.append({
                "Child_Content": child_doc.page_content[:40] + "...",
                "Type": "Database Vector",
                "Source_File": source_file,
                "Parent_Preview": parent_info
            })

        # 加入當前 Query 的向量
        query_vec = embeddings.embed_query(query)
        vectors.append(query_vec)
        metadata_list.append({
            "Child_Content": query[:40] + "...",
            "Type": "Your Query",
            "Source_File": "N/A",
            "Parent_Preview": "N/A"
        })

        # PCA 降維
        vectors_np = np.array(vectors)
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors_np)

        # 建立 DataFrame 並繪圖
        df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
        # 將 metadata 列表轉成 DataFrame 欄位
        for key in metadata_list[0].keys():
            df[key] = [m[key] for m in metadata_list]

        fig = px.scatter(
            df, x='x', y='y', color='Type', 
            # 關鍵：在 hover_data 中加入所有想顯示的資訊
            hover_data={
                'x': False, 'y': False, # 隱藏座標數值
                'Type': True,
                'Source_File': True,
                'Child_Content': True,
                'Parent_Preview': True
            },
            title="RAG 向量空間視覺化 (帶溯源資訊)",
            template="plotly_white"
        )
        
        # 標註 Query 點的大小以便識別
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        fig.show()

if __name__ == "__main__":
    main()