import uuid
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 改用 JsonOutputParser 的版本
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

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

# 當要求 LLM 回傳 Python 列表時，它往往會順手加上 Markdown 的程式碼區塊標記（如 ```python ... ```）
# 或是前言後記。這會導致 Python 的 eval() 函數解析失敗.
# 改用 LangChain 的 JsonOutputParser 可以自動幫忙處理掉那些煩人的 Markdown 標籤。
#
# 另外, 我們也會在後續的迴圈中, 讓 LLM 以 JSON 方式回傳「摘要與關鍵字」的結構化資料, 這樣我們就能個別提取與組合了。

# 初始化 Json Output Parser
json_parser = JsonOutputParser()

# 讓關鍵字也成為一種child doc.
# 兩種策略： 
#   附加在摘要中 : 將關鍵字與摘要文字結合在一起（例如：摘要內容... [關鍵字：A, B, C]）後轉換成一個向量。
#     優點：關鍵字能獲得摘要的上下文保護，向量表現較穩定。
#     缺點：如果關鍵字權重被淹沒在長句子中，純關鍵字的搜尋效果會稍打折扣。
#   獨立成一個 Child Doc : 將整個「關鍵字列表」串成一條字串做成一個獨立的向量點。
#     優點：對於「精確術語」的匹配能力極強。當使用者只輸入專有名詞時，這個點會非常靠近 Query。
#     疑慮：沒有前後文.但因為它是從同一份 Parent Doc 產生的，其向量空間位置仍會落在該主題附近。
# 我們在之後的迴圈中, 兩者都要.

# 準備 summary and keyword 的 prompt 與 chain
# 先要 LLM 以 JSON 方式summary與keywords. 之後我們好方便個別提取與組合
summary_prompt = ChatPromptTemplate.from_template(
    "請閱讀以下內容，並回傳一個 JSON 物件，包含 'summary' (極簡短摘要) 與 'keywords' (3-5個關鍵字列表)。\n"
    "{format_instructions}\n"
    "內容：{doc}"
)

summary_chain = (
    summary_prompt 
    | ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    | json_parser
)

# 準備 questions 的 prompt 與 chain
question_prompt = ChatPromptTemplate.from_template(
    "請針對以下內容，生成 3 個使用者可能會問的簡短問題。\n"
    "{format_instructions}\n"
    "內容：{doc}"
)

question_chain = (
    question_prompt 
    | ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    | json_parser
)
# -------------------------------------------------------------

# 處理與存入
for doc in parent_docs:
    unique_uuid = str(uuid.uuid4())
    
    # ------------------------------------------------------------------
    # 策略 A: 摘要與關鍵字
    print(f"正在生成 JSON 結構化摘要與關鍵字...")
    # 生成 JSON 結構化的摘要與關鍵字
    # 由json_parser.get_format_instructions()提供給 LLM 如何輸出 JSON 格式的指令.
    summary_keyword_result = summary_chain.invoke({
        "doc": doc.page_content,
        "format_instructions": json_parser.get_format_instructions()
    })
    
    # 個別提取內容
    summary_text = summary_keyword_result.get("summary", "")
    keywords_list = summary_keyword_result.get("keywords", [])
    
    # 建立 純粹的「摘要」向量點做標籤匹配
    summary_doc = Document(page_content=summary_text, metadata={uuid_key_name: unique_uuid})

    # 建立 純粹的「關鍵字列表」向量點，加強標籤匹配
    keywords_doc = Document(page_content=f"關鍵字標籤: {', '.join(keywords_list)}", metadata={uuid_key_name: unique_uuid})

    # 用「摘要+關鍵字」的內容 建立增強型向量點
    combined_text = f"摘要: {summary_text} \n關鍵字: {', '.join(keywords_list)}"
    combined_doc = Document(page_content=combined_text, metadata={uuid_key_name: unique_uuid})

    # ------------------------------------------------------------------
    # 策略 B: 假設性問題
    print(f"正在生成假設性問題...")
    # 用JsonOutputParser確保輸出是乾淨的, 直接就是python的字串list，不需要 eval.
    questions = question_chain.invoke({
        "doc": doc.page_content,
        "format_instructions": json_parser.get_format_instructions()
    })
    # questions 現在直接就是 ['問題1', '問題2', '問題3']. 
    # 回傳的就是真正的 list 了，不需要 eval() 來解析了。這樣就不會有 Markdown 標記干擾了。
    # ------------------------------------------------------------------
    question_docs = [Document(page_content=q, metadata={uuid_key_name: unique_uuid}) for q in questions]
    
    # 存入向量與原始內容
    retriever.vectorstore.add_documents([summary_doc, combined_doc, keywords_doc] + question_docs)
    retriever.docstore.mset([(unique_uuid, doc)])

# 5. 持久化儲存 (關鍵步驟)
print(f"\n--- 正在持久化儲存至硬碟 ---")
# 儲存 FAISS 向量與 ID 映射
retriever.vectorstore.save_local(VECTOR_DB_PATH)
print(f"FAISS 已儲存至: {VECTOR_DB_PATH}")
print(f"原始文件已存於: {BYTE_STORE_PATH}")
