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

# 1. 準備實驗文本 (沿用你的版本)
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

# 初始化模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 建立資料庫
def create_db(text, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_text(raw_text)
    docs = [
        Document(page_content=t, metadata={"original_index": i}) for i, t in enumerate(chunks)
    ]
    return FAISS.from_documents(docs, embeddings)

db_small = create_db(raw_text, 100, 20)

# 3. 新增：向 LLM 詢問的函數
def ask_llm_with_context(query, context):
    # 定義 RAG 專用的 Prompt 模板
    template = """你是一個專業的人事助理。請根據下方提供的【參考資料】回答使用者的【問題】。
    如果參考資料中沒有相關資訊，請誠實回答你不知道，不要胡編亂造。

    【參考資料】：
    {context}

    【問題】：
    {query}

    你的回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
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

    # 執行並回傳結果
    response = chain.invoke({"query": query, "context": context})
    return response.content

# 4. 修改後的實驗函數
def run_experiment_v2(query, db, threshold=0.9):
    print(f"\n" + "="*60)
    print(f"【測試問題】: {query}")
    
    # 向量檢索
    results = db.similarity_search_with_score(query, k=1)
    doc, score = results[0]
    
    print(f"檢索分析 - 信心分數 (L2 距離): {score:.4f}")
    
    # 門檻檢查
    if score > threshold:
        print(f"判定結果: [ 門檻攔截 ] 距離 {score:.4f} > {threshold}，判定為未知。")
        print("AI 回答: 很抱歉，我在公司政策中找不到相關資訊。")
    else:
        print(f"判定結果: [ 通過 ] 準備送往 LLM 生成回答...")
        
        # 呼叫 LLM 進行問答
        final_answer = ask_llm_with_context(query, doc.page_content)
        
        print(f"\n【LLM 回答內容】:\n{final_answer}")

# --- 測試展示 ---
run_experiment_v2("入職半年的補助是多少？", db_small, threshold=0.9)
run_experiment_v2("公司的咖啡機怎麼用？", db_small, threshold=0.9)

# 進階用法：在 LLM 回答後，根據回答的長度或內容，再決定是否要進行「二次擴充提問」。
#
# 在 RAG 實務中，可以用這個技巧來實現更多, 例如：
#   Self-Correction (自我修正)：如果 LLM 回答說「我不知道」，你可以判斷這個字眼，然後自動更換另一個檢索器（或是擴大搜尋範圍）再試一次。
#   安全性檢查：判斷回答中是否包含敏感字眼，如果有，則自動替換為罐頭回覆。
#   格式轉換：如果偵測到使用者需要圖表，自動將文字轉為 Markdown 表格格式。
'''
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 2. 定義兩個不同的 Prompt
# 初始回答 Prompt
initial_prompt = ChatPromptTemplate.from_template("請簡短回答這個問題：{question}")

# 擴充指令 Prompt
expansion_prompt = ChatPromptTemplate.from_template(
    "你剛才的回答太簡短了：'{initial_answer}'\n"
    "請針對問題 '{question}' 提供更詳細、更具溫度的解釋。"
)

# 定義一個檢查 input_data (前一級的輸出dict) 的函式, 用 RunnableLambda 包裝後，可以在 LCEL 管線中使用它來做條件判斷和流程控制。
def check_length_and_expand(input_data):
    # 這裡接收的是上一個步驟傳來的字典
    question = input_data["question"]
    initial_answer = input_data["initial_answer"]
    
    if len(initial_answer) < 20:
        print(f"--- [系統偵測：回答僅 {len(initial_answer)} 字，觸發擴充邏輯] ---")
        # 如果太短，串接擴充 Prompt 並再次呼叫 LLM
        expansion_chain = expansion_prompt | llm | StrOutputParser()
        return expansion_chain.invoke({"question": question, "initial_answer": initial_answer})
    else:
        print("--- [系統偵測：回答長度足夠] ---")
        return initial_answer

# 4. 構建 LCEL 總管線
# Step A: 取得初始回答.這一級是一個"字典結構"的管線 
#           { 
#               "key1": runnable1
#               "key2": runnable2
#           } ，
#         LCEL 會"並行"執行這些任務，並將結果組合成一個字典傳給下一步。
# Step B: 將 (問題 + 初始回答) 傳給 RunnableLambda 做判斷

full_chain = (
    {   # 初始input 是一個字典結構的管線. 
        # 這裡我們用一個dict來同時保存原始問題和初始回答，讓後續的 RunnableLambda 可以根據這些資訊做判斷。
        # RunnablePassthrough() : 當我們需要把原始的 question 一直傳遞到後面的步驟時(傳到後面的 RunnableLambda)，
        #                         我們用它來保留原始輸入，不被中間的 LLM 呼叫給蓋掉。
        # 在這個例子中，這讓 check_length_and_expand 函數能同時拿到「問題」與「初步答案」
        "question": RunnablePassthrough(),          
        "initial_answer": initial_prompt | llm | StrOutputParser()
    }
    | RunnableLambda(check_length_and_expand)
)

result1 = full_chain.invoke("你是誰？")
''' 
