# 假設性問題檢索 (Hypothetical Question Indexing):
# 讓 LLM 為原始文本生成 3 個「可能會被問到的問題」存入向量庫，這能讓搜尋的精確度再翻倍。
# 這是因為, 
#   原始內容：  通常是陳述句，充滿細節。
#   假設性問題：我們預先讓 LLM 針對內容生成「可能會被問到的 3 個問題」。當使用者的問題進來時，它是跟這些「問題」做向量比對。
#              因為句型結構和語意意圖高度相似，搜尋的精確度會大幅提升。

import uuid
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import InMemoryByteStore

# for 視覺化
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

# 1. 準備原始長文本 (Parent Documents)
parent_docs = [
    Document(page_content="2024年公司旅遊補助政策：年資滿一年可領3萬元，滿半年按比例發放，需於10天內附發票報帳。"),
    Document(page_content="公司辦公室規範：每日9點前需打卡，咖啡機使用後請清理，會議室需提前一週預約。")
]

# 2. 初始化組件
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 建立向量庫 (存小向量)
vectorstore = FAISS.from_documents([Document(page_content="init")], embeddings)

# 建立parent文件庫 (存完整大內容)
store = InMemoryByteStore()
uuid_key_name = "abstract_and_doc_uuid"

# 初始化MultiVector檢索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,    # 向量庫物件 (FAISS + dosctore, for 檢索用)
    byte_store=store,           # parent文件庫物件 (InMemoryByteStore, for 存取完整內容用)
    id_key=uuid_key_name,       # 指定MultiVectorRetriever要用Document物件的metadata裡的哪個key來取得用來關聯向量與完整內容的uuid
)

# 擴展之前的迴圈，讓每一份原始文件都擁有一組「摘要」和三筆「問題」作為檢索的對象。

# 3-1. 使用 LLM 生成「摘要」作為檢索用的 Child Vectors
summary_chain = (
    ChatPromptTemplate.from_template("請幫這段文字寫一個極簡短的摘要,並用\"摘要:\"開頭：\n\n{doc}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 3-2. 定義生成「假設性問題」的 Chain
# 我們要求 LLM 回傳 JSON 格式以便處理
question_prompt = ChatPromptTemplate.from_template(
    "請針對以下內容，生成 3 個使用者可能會問的簡短問題。\n"
    "請只回傳 Python 列表格式，例如: ['問題1', '問題2', '問題3']\n\n"
    "內容：{doc}"
)
question_chain = question_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

# 4. 將資料存入：同時產生摘要並關聯
for doc in parent_docs:
    # 為每一筆 parent_doc 與 abstract 產生唯一的 UUID，作為它們在 InMemoryByteStore 與 docstore 中的 key，用來關聯 FAISS 中的摘要向量。
    unique_uuid = str(uuid.uuid4())
    
    # --- 策略 A: 摘要向量 ---
    print(f"正在為內容生成摘要...")
    summary = summary_chain.invoke({"doc": doc.page_content})
    summary_doc = Document(page_content=summary, metadata={uuid_key_name: unique_uuid})

    # --- 策略 B: 假設性問題向量 ---
    print(f"正在為內容生成假設性問題...")
    questions_raw = question_chain.invoke({"doc": doc.page_content})
    # 簡單處理字串轉列表 (實際開發建議用 JsonOutputParser)
    questions = eval(questions_raw) 
    
    # 將每個問題都做成一個 Document，並共享同一個 UUID，以便它們在向量庫中與原始內容關聯。
    question_docs = [
        Document(page_content=q, metadata={uuid_key_name: unique_uuid}) 
        for q in questions
    ]
    
    # 小向量與摘要用 MultiVectorRetriever.vectorstore.add_documents 加進 FAISS 與 docstore，
    # 原始大內容進入 MultiVectorRetrieverdocstore.mset 加進 ByteStore。
    retriever.vectorstore.add_documents([summary_doc])  # 加入摘要向量.
    retriever.vectorstore.add_documents(question_docs)  # 加入假設性問題向量.
    retriever.docstore.mset([(unique_uuid, doc)])       # 以"UUID, 原始內容" 作為 InMemoryByteStore 的一筆紀錄

# --- 測試搜尋 ---
query = "我剛來公司半年，旅遊有錢拿嗎？"
# 雖然搜尋的是摘要，但回傳的是「完整原始內容」
retrieved_docs = retriever.invoke(query)

print(f"【問題】: {query}")
print(f"【檢索到的完整內容】: {retrieved_docs[0].page_content}")

# =========== 視覺化所有向量點 ===============

# 1. 提取所有「小向量」與相關資訊
# 我們需要遍歷 FAISS 裡面所有的 index
vector_db = retriever.vectorstore
num_vectors = vector_db.index.ntotal
all_vectors = []
metadata_list = []

for i in range(num_vectors):
    # 取得物理向量
    vec = vector_db.index.reconstruct(i)
    
    # 取得該向量在 FAISS 內部 docstore 的 UUID
    faiss_uuid = vector_db.index_to_docstore_id[i]
    
    # 取得小內容 (摘要或問題)
    child_doc = vector_db.docstore.search(faiss_uuid)
    
    # 跳過初始化時的 "init" 點，保持圖表乾淨
    if child_doc.page_content == "init":
        continue
        
    # 取得關聯的 Parent UUID (你定義的 uuid_key_name)
    parent_uuid = child_doc.metadata.get(uuid_key_name)
    
    # 從 ByteStore 取得原始大內容 (Parent)
    parent_doc = retriever.docstore.mget([parent_uuid])[0]
    parent_content = parent_doc.page_content if parent_doc else "無關聯內容"

    all_vectors.append(vec)
    metadata_list.append({
        "內容": child_doc.page_content,
        "類型": "摘要" if "摘要" in child_doc.page_content else "假設性問題", # 簡單判斷，也可依 metadata
        "來源原文": parent_content,
        "標籤": f"Child {i}"
    })

# 2. 加入 Query 向量
query_vec = embeddings.embed_query(query)
all_vectors.append(query_vec)
metadata_list.append({
    "內容": query,
    "類型": "查詢 (Query)",
    "來源原文": "N/A",
    "標籤": "Query"
})

# 3. PCA 降維
pca = PCA(n_components=2)
coords = pca.fit_transform(np.array(all_vectors))

# 4. 建立 DataFrame
df = pd.DataFrame(metadata_list)
df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# 5. Plotly 繪圖
fig = px.scatter(
    df, x="x", y="y",
    color="類型",
    symbol="類型",
    text="標籤",
    hover_data=["內容", "來源原文"], # 滑鼠移上去能看到小標籤與它對應的原文
    title="Multi-Vector RAG 語意映射圖 (摘要 + 假設問題 + 查詢)",
    template="plotly_white",
    height=700
)

# 優化視覺效果
fig.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
fig.show()