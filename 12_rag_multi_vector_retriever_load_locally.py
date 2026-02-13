from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore

# 1. 初始化路徑與模型
VECTOR_DB_PATH = "faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 載入 FAISS 向量庫
# 注意：allow_dangerous_deserialization 必須為 True 以讀取本地 pkl
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. 載入磁碟型 ByteStore
store = LocalFileStore(BYTE_STORE_PATH)

# 4. 重建檢索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key="abstract_and_doc_uuid",
)

# 5. 測試新問題
query = "會議室預約"
results = retriever.invoke(query)

print(f"【問題】: {query}")
if results:
    print(f"【檢索結果】: {results[0].page_content}")