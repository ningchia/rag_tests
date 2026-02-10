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
from langchain_core.prompts import ChatPromptTemplate

# 1. 準備實驗文本 (沿用你的設定)
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # 作為 Re-ranker

# 2. 建立資料庫函數
def create_db(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    docs = [Document(page_content=t) for t in splitter.split_text(text)]
    return FAISS.from_documents(docs, embeddings)

db_small = create_db(raw_text, 100, 20)

# 3. 簡易 Re-ranker 邏輯：判斷該片段是否真的能回答問題
def llm_re_rank(query, context):
    prompt = ChatPromptTemplate.from_template("""
    你是一個嚴格的審查員。請判斷下方的【參考資料】是否真的包含回答【問題】所需的資訊。
    若有，請回答 'YES'，若沒有或關聯度極低，請回答 'NO'。
    
    問題：{query}
    參考資料：{context}
    答案（僅輸出 YES/NO）：""")
    
    # 使用 LCEL (LangChain Expression Language) 語法 (prompt | llm) 建立鍊 (chain).
    # Ref: https://blog.langchain.com/langchain-expression-language/
    #
    # 跟linux的pipeline類似, 將前一個組件的輸出，直接當作下一個組件的輸入.
    # 這種寫法自動支援了 串流輸出 (Streaming)、非同步執行 (Async) 以及 重試機制 (Retry)。
    # 另外也支援串接性 (Composability)。
    # 例如： chain = prompt | llm | StrOutputParser() 這代表：生成 Prompt -> 丟給 LLM -> 將 LLM 的複雜物件直接轉成乾淨的「純字串」。
    # prompt : 一個 ChatPromptTemplate 物件。接收一個dict（例如 {"query": "...", "context": "..."}），
    #          並輸出一條格式化好的完整字串（或稱為 PromptValue）。
    # |      : 它像是一條傳送帶，自動把 prompt 產出的文字傳給下一站。
    # llm    : 一個 ChatOpenAI 物件。它接收來自傳送帶的文字，呼叫 OpenAI API，並輸出 AI 的回覆物件。
    
    # 宣告一個 「處理管線」。當執行 chain.invoke() 時，LangChain 就會啟動這個管線，自動完成文字填充與模型呼叫的工作。
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    return response.content.strip().upper()

# 4. 增強版實驗函數
def run_advanced_experiment(query, db, threshold=0.9):
    print(f"\n" + "="*60)
    print(f"【測試問題】: {query}")
    
    # 顯示 User Prompt Embedding 資訊
    q_vector = embeddings.embed_query(query)
    print(f"Prompt 向量首尾: [{q_vector[0]:.4f}, {q_vector[1]:.4f} ... {q_vector[-2]:.4f}, {q_vector[-1]:.4f}]")
    
    # 第一階段：向量檢索 (Recall)
    results = db.similarity_search_with_score(query, k=1)
    doc, score = results[0]
    
    # 取得 FAISS 內部向量資訊
    doc_vector = db.index.reconstruct(0) 
    
    print(f"\n--- 第一階段：向量檢索分析 ---")
    print(f"  - 信心分數 (L2 距離): {score:.4f}")
    
    # 初步門檻判斷
    if score > threshold:
        print(f"  - [門檻攔截] 距離 {score:.4f} > {threshold}，判定為未知。")
        return

    print(f"  - [通過門檻] 片段前後5字: 「{doc.page_content[:5]} ... {doc.page_content[-5:]}」")
    print(f"  - Chunk 向量首尾: [{doc_vector[0]:.4f}, {doc_vector[1]:.4f} ... {doc_vector[-2]:.4f}, {doc_vector[-1]:.4f}]")

    # 第二階段：Re-rank (LLM 審查)
    print(f"\n--- 第二階段：LLM Re-rank 審查 ---")
    re_rank_decision = llm_re_rank(query, doc.page_content)
    print(f"  - LLM 審查結果: {re_rank_decision}")
    
    if "YES" in re_rank_decision:
        print(">> 最終判定：此資料有效，可進行回答。")
    else:
        print(">> 最終判定：雖然向量接近，但 LLM 認為內容無關。我不知道。")

# --- 實驗展示 ---
run_advanced_experiment("入職半年的補助是多少？", db_small, threshold=0.9)
run_advanced_experiment("如何修理太空梭？", db_small, threshold=0.9)

