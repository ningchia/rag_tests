import uuid
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiVectorRetriever
# 關鍵：改用 LocalFileStore
# from langchain_classic.storage import InMemoryByteStore
from langchain_classic.storage import LocalFileStore

# 0. 定義儲存路徑
VECTOR_DB_PATH = "./faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"

# 1. 準備原始長文本
parent_docs = [
    Document(page_content="2024年公司旅遊補助政策：年資滿一年可領3萬元，滿半年按比例發放，需於10天內附發票報帳。"),
    Document(page_content="公司辦公室規範：每日9點前需打卡，咖啡機使用後請清理，會議室需提前一週預約。")
]

# 2. 初始化組件
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents([Document(page_content="init")], embeddings)

# 關鍵：初始化磁碟型 ByteStore
if not os.path.exists(BYTE_STORE_PATH):
    os.makedirs(BYTE_STORE_PATH)
store = LocalFileStore(BYTE_STORE_PATH)

uuid_key_name = "abstract_and_doc_uuid"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=uuid_key_name,
)

# 3. 定義 Chain (摘要與問題生成)
summary_chain = (
    ChatPromptTemplate.from_template("請幫這段文字寫一個極簡短的摘要,並用\"摘要:\"開頭：\n\n{doc}")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 當要求 LLM 回傳 Python 列表時，它往往會順手加上 Markdown 的程式碼區塊標記（如 ```python ... ```）
# 或是前言後記。這會導致 Python 的 eval() 函數解析失敗.
# -------------------------------------------------------------
# 方法 :  改用 JsonOutputParser. 我們不再求 LLM 給 Python List，而是要求它給 JSON Array。
# LangChain 的 JsonOutputParser 會自動幫忙處理掉那些煩人的 Markdown 標籤。
from langchain_core.output_parsers import JsonOutputParser

# 1. 初始化 Parser
parser = JsonOutputParser()

# 2. 修改 Prompt 與 Chain
question_prompt = ChatPromptTemplate.from_template(
    "請針對以下內容，生成 3 個使用者可能會問的簡短問題。\n"
    "{format_instructions}\n"
    "內容：{doc}"
)

# 將 Parser 的指令注入 Prompt，這會告訴 LLM 必須回傳 JSON
question_chain = (
    question_prompt 
    | ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    | parser
)
# -------------------------------------------------------------

# 4. 處理與存入
for doc in parent_docs:
    unique_uuid = str(uuid.uuid4())
    
    # 策略 A: 摘要
    print(f"正在生成摘要...")
    summary = summary_chain.invoke({"doc": doc.page_content})
    summary_doc = Document(page_content=summary, metadata={uuid_key_name: unique_uuid})

    # 策略 B: 假設性問題
    print(f"正在生成假設性問題...")
    # ------------------------------------------------------------------
    # 用JsonOutputParser確保輸出是乾淨的, 直接就是python的字串list，不需要 eval.
    # 由parser.get_format_instructions()提供給LLM格式指令.
    questions = question_chain.invoke({
        "doc": doc.page_content,
        "format_instructions": parser.get_format_instructions()
    })
    # questions 現在直接就是 ['問題1', '問題2', '問題3']. 
    # 回傳的就是真正的 list 了，不需要 eval() 來解析了。這樣就不會有 Markdown 標記干擾了。
    # ------------------------------------------------------------------

    question_docs = [Document(page_content=q, metadata={uuid_key_name: unique_uuid}) for q in questions]
    
    # 存入向量與原始內容
    retriever.vectorstore.add_documents([summary_doc] + question_docs)
    retriever.docstore.mset([(unique_uuid, doc)])

# 5. 持久化儲存 (關鍵步驟)
print(f"\n--- 正在持久化儲存至硬碟 ---")
# 儲存 FAISS 向量與 ID 映射
retriever.vectorstore.save_local(VECTOR_DB_PATH)
print(f"FAISS 已儲存至: {VECTOR_DB_PATH}")
print(f"原始文件已存於: {BYTE_STORE_PATH}")