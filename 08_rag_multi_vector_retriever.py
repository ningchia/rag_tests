# 多向量檢索用來解決這個矛盾:
#   檢索時：向量最好是「摘要」或「關鍵字」，這樣比較容易跟使用者的簡短問題匹配。
#   回答時：LLM 需要「完整且詳細的內容」才能回答正確。
# 所以我們需要讓一個「大片段（Parent Document）」可以擁有多個「小向量（Child Vectors）」，例如：
#   摘要向量：把長文縮編成一句話。
#   假設性問題向量：預測使用者可能會問什麼問題。

import uuid
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import InMemoryByteStore

# 1. 準備原始長文本 (Parent Documents)
parent_docs = [
    Document(page_content="2024年公司旅遊補助政策：年資滿一年可領3萬元，滿半年按比例發放，需於10天內附發票報帳。"),
    Document(page_content="公司辦公室規範：每日9點前需打卡，咖啡機使用後請清理，會議室需提前一週預約。")
]

# 2. 初始化組件
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 複習一下執行 FAISS.from_documents() 時發生的事:
#   Step 1: 產生 UUID. LangChain 為每個 Document 片段自動生成一個隨機 UUID（如果你沒給的話）。
#   Step 2: 處理向量 (FAISS 內部). LangChain 計算 page_content 的 Embedding。然後將這些向量丟進 FAISS 引擎。
#           FAISS 引擎內部並不存 UUID。它只會按照存入順序給一個整數編號（0, 1, 2...）。
#   Step 3: 建立映射表 (The Glue). LangChain 內部維護了一個字典叫做 index_to_docstore_id。
#           它記錄著：{0: "uuid_A", 1: "uuid_B"}。
#   Step 4: 存入 Docstore. 將 {"uuid_A": Document物件} 存入 docstore。
# 
# 當呼叫 similarity_search_with_score() 時：
#   FAISS 引擎說： 「跟query最像的是第 N 號向量。」
#   LangChain 查表： 翻開 index_to_docstore_id，看到第 N 號對應的 UUID 是 "uuid_N"。
#   Docstore 抓取： 叫用 docstore.search(uuid) 來根據 "uuid_N" 去 docstore 把那個包裹了原始文字和元數據的 Document 物件撈出來。

# 但是在MultiVectorRetriever的情境下，我們不希望FAISS的docstore存原始大內容，
# 因為我們要把原始大內容存到InMemoryByteStore裡
#
# InMemoryByteStore 與 FAISS 的內建 docstore 是不同的.
#   docstore: 當我們把「摘要」存進 FAISS 時，FAISS 內部依然有一個 docstore。但這時它存的是摘要內容。 => 為了檢索精確用.
#   InMemoryByteStore: 這是一個外部儲存空間，是用來存「原始完整文本（Parent Document）」的. => 真正要餵給 AI 的原始大檔案.
#
# MultiVector檢索運作流程：
#   使用者提問 -> 與 FAISS 裡的「摘要向量」比對。
#   找到最像的摘要後，從該Document物件的metadata裡, 指定key的值來獲取其關聯的 UUID。這個key的名稱會在初始化MultiVectorRetriever時指定。
#   拿著這個 UUID，去 InMemoryByteStore 把對應的「原始完整文本」取出來。

# 建立向量庫 (存小向量)
# 由於 FAISS 本身在呼叫 add_documents 或 add_texts 之前，必須先被「建立」起來 (先知道向量維度)，
# 所以我們先放一個 dummy document 進去，讓它知道向量維度是什麼。建立後才能用它來初始化MultiVector檢索器.
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

# 3. 使用 LLM 生成「摘要」作為檢索用的 Child Vectors
chain = (
    ChatPromptTemplate.from_template("請幫這段文字寫一個極簡短的摘要：\n\n{doc}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 4. 將資料存入：同時產生摘要並關聯
for doc in parent_docs:
    # 為每一筆 parent_doc 與 abstract 產生唯一的 UUID，作為它們在 InMemoryByteStore 與 docstore 中的 key，用來關聯 FAISS 中的摘要向量。
    # 之前我們並沒有這麼做，因為FAISS.from_documents() 會自動產生 uuid。
    # 但現在我們要自己產生，因為需要一個跟 docstore 裡標記 Document物件相同的 UUID , 來關聯在 InMemoryByteStore 裡的 parent 文件,
    # 讓 MultiVectorRetriever 找到相近向量時, 改到 InMemoryByteStore 裡, 用相同的UUID 來找到對應的完整內容。
    # 這樣就解決了「檢索用向量是摘要，但回答需要完整內容」的矛盾。
    # MultiVectorRetriever 會到Document物件的metadata裡, 用我們指定的 key (uuid_key_name) 來取得這個 UUID, 然後用這個 UUID 
    # 去 InMemoryByteStore 裡找對應的完整內容。
    unique_uuid = str(uuid.uuid4())
    # 生成摘要
    summary = chain.invoke({"doc": doc.page_content})
    summary_doc = Document(page_content=summary, metadata={uuid_key_name: unique_uuid})
    
    # 注意是透過 MultiVectorRetriever 來操作, 而非直接操作 FAISS (vector_db) 或 InMemoryByteStore.
    #   小向量與摘要用 MultiVectorRetriever.vectorstore.add_documents 加進 FAISS 與 docstore，
    #   原始大內容進入 MultiVectorRetrieverdocstore.mset 加進 ByteStore。
    retriever.vectorstore.add_documents([summary_doc])  # UUID 先前已被加在document物件的metadata裡了, 所以這裡不需要再特別傳入了.
    retriever.docstore.mset([(unique_uuid, doc)])       # 以"UUID, 原始內容" 作為 InMemoryByteStore 的一筆紀錄

# --- 測試搜尋 ---
query = "我剛來公司半年，旅遊有錢拿嗎？"
# 雖然搜尋的是摘要，但回傳的是「完整原始內容」
retrieved_docs = retriever.invoke(query)

print(f"【問題】: {query}")
print(f"【檢索到的完整內容】: {retrieved_docs[0].page_content}")