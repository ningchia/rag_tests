# parent doc改成從命令列參數輸入文字檔, 可以輸入多個文字檔, 
# 另外為了避免重複，我們就在 parent_docs 的 metadata 中加入一個 source 檔名 以及 hash，
# 在處理前先檢查 LocalFileStore 中是否已經存在該 hash，若存在則跳過生成步驟.

# 執行方式 : 可同時匯入多個檔案
#   python3 13_rag_add_doc_incrementally.py file1.txt file2.txt

# 改用 hash 作為vector與parent doc 的關聯鍵
# import uuid
import hashlib
import argparse

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

# 引入切割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 0. 定義儲存路徑
VECTOR_DB_PATH = "./faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"

# 改用 hash 作為vector與parent doc 的關聯鍵 (因為檔名可能重複  ，但內容相同的檔案我們希望只存一份)
hash_key_name = "doc_hash_id" 

def get_content_hash(content: str) -> str:
    """計算文本內容的 SHA-256 雜湊值作為唯一識別碼"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def main():
    # 1. 設定命令列參數
    parser = argparse.ArgumentParser(description="增量同步文字檔至 Multi-Vector RAG 向量庫")
    parser.add_argument("files", nargs="+", help="要匯入的一個或多個文字檔路徑")
    args = parser.parse_args()

    # 2. 初始化組件
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 檢查並載入現有向量庫
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(os.path.join(VECTOR_DB_PATH, "index.faiss")):
        print(f"--- 載入現有向量庫: {VECTOR_DB_PATH} ---")
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("--- 建立全新向量庫 ---")
        vectorstore = FAISS.from_documents([Document(page_content="init")], embeddings)

    # 初始化磁碟型 ByteStore
    if not os.path.exists(BYTE_STORE_PATH):
        os.makedirs(BYTE_STORE_PATH)
    store = LocalFileStore(BYTE_STORE_PATH)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=hash_key_name,
    )

    # 初始化切割器：建議 chunk_size 在 800~1500 之間
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # 3. 定義 LLM 指令
    json_parser = JsonOutputParser()
    
    summary_chain = (
        ChatPromptTemplate.from_template("摘要此內容並列出3-5個關鍵字，以JSON格式回傳 ('summary', 'keywords')：\n{format_instructions}\n內容：{doc}")
        | ChatOpenAI(model="gpt-4o-mini", temperature=0) | json_parser
    )

    question_chain = (
        ChatPromptTemplate.from_template("針對內容生成 3 個問題，以JSON列表格式回傳：\n{format_instructions}\n內容：{doc}")
        | ChatOpenAI(model="gpt-4o-mini", temperature=0) | json_parser
    )

    # 4. 逐一處理輸入檔案
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"跳過：檔案不存在 {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            if not raw_text.strip():
                continue

        # 1. 將檔案切分成多個 Chunks
        chunks = text_splitter.split_text(raw_text)
        print(f"檔案 {file_path} 已切分為 {len(chunks)} 個區塊")

        for i, chunk_content in enumerate(chunks):
            # 2. 以 Chunk 內容計算 Hash (確保重複內容不重複處理)
            chunk_hash = get_content_hash(chunk_content)
            
            # 檢查 Chunk 是否已存在
            if store.mget([chunk_hash])[0] is not None:
                print(f"  - Chunk {i} 已存在，跳過")
                continue

            print(f"  - 正在處理 Chunk {i}...")
            
            # 建立 Parent Document
            parent_doc = Document(
                page_content=chunk_content, 
                metadata={"source": file_path, "chunk_index": i, "hash": chunk_hash}
            )

            # 生成子向量內容
            try:
                # 在處理法律條文時，LLM 往往會想要讓輸出變得更「結構化」（例如加上 {"條款": "...", "問題": "..."}），
                # 導致解析後的資料型態與 Document 物件不相容。
                # 透過增加 isinstance(q, dict) 的檢查，程式就能適應不同 LLM 模型或不同隨機性下的回傳差異。

                # 為 chunk 產生 summary 
                s_res = summary_chain.invoke({"doc": chunk_content, "format_instructions": json_parser.get_format_instructions()})

                # 預防 LLM 回傳結構不一致
                # summary = s_res.get("summary", "")
                # keywords = s_res.get("keywords", [])
                summary = s_res.get("summary") if isinstance(s_res, dict) else str(s_res)
                keywords = s_res.get("keywords", []) if isinstance(s_res, dict) else []

                # 加入 summary, summary+keywords,與 keywords 等 child vectors. 
                child_docs = [
                    Document(page_content=summary, metadata={hash_key_name: chunk_hash}),
                    Document(page_content=f"摘要: {summary} \n關鍵字: {', '.join(keywords)}", metadata={hash_key_name: chunk_hash}),
                    Document(page_content=f"關鍵字標籤: {', '.join(keywords)}", metadata={hash_key_name: chunk_hash})
                ]

                # 生成假設性問題 (加入強健的型別處理)
                try:
                    q_res = question_chain.invoke({"doc": chunk_content, "format_instructions": json_parser.get_format_instructions()})
                    
                    # 確保 q_res 是列表且內容為字串
                    if isinstance(q_res, list):
                        for q in q_res:
                            # 關鍵修正：確保 q_text 是字串
                            q_text = q.get("question") if isinstance(q, dict) else str(q)
                            child_docs.append(Document(page_content=q_text, metadata={hash_key_name: chunk_hash}))
                except Exception as e:
                        print(f"  ! Chunk {i} 處理失敗: {e}")
                # 存入
                retriever.vectorstore.add_documents(child_docs)
                # LocalFileStore 是一個底層的 ByteStore，它在硬碟上儲存的是原始的二進位數據（bytes）。
                # 當使用 retriever.docstore.mset 存入 Document 物件時，LangChain 在後台自動使用了 Python 的 
                # pickle 模組將物件序列化（Serialize）成 bytes 存入硬碟。
                # 因此，當我們從 LocalFileStore 讀取數據時，讀取到的就是這些被序列化成 bytes 的數據。
                # 當直接從 docstore.mget 取出資料時，拿到的會是 「被封裝過的 bytes」，而不是原本的 Document 物件。
                # 因此，我們需要使用 pickle 模組將這些 bytes 反序列化（Deserialize）回原本的 Document 物件，
                # 才能存取其中的 page_content 和 metadata。
                # 例如 : parent_bytes = store.mget([p_hash])[0]
                #       parent_doc = pickle.loads(parent_bytes)
                #
                # 不過在較新的版本或特定配置下，如果你存入的是 Document 物件，它可能會自動幫你轉成 JSON 字串儲存，
                # 而不是用 Python 傳統的 pickle 序列化。這種情況下就不需要 pickle 了，要改用 json 模組來解析，
                # 並將解析出來的字典重新包裝回 Document 物件。否則, 會看到下列錯誤:
                #   .... _pickle.UnpicklingError: invalid load key, '{'.
                #
                # 目前的版本似乎就是使用JSON儲存的，所以我們在讀取時會拿到一個 JSON 字串，而不是 bytes 的 pickle。
                # 這時候我們就要用 json.loads 來解析，然後再轉成 Document 物件。
                # json.loads() 後的 data 內容長這樣:
                # {'lc': 1, 'type': 'constructor', 'id': ['langchain', 'schema', 'document', 'Document'], 
                # 'kwargs': {'metadata': {'source': 'card_apply.txt', 'chunk_index': 2, 'hash': 'e5748f...'}, 
                # 'page_content': '原始內容...', 'type': 'Document'}}
                retriever.docstore.mset([(chunk_hash, parent_doc)])
                
            except Exception as e:
                print(f"  ! Chunk {i} 處理失敗: {e}")

    retriever.vectorstore.save_local(VECTOR_DB_PATH)
    print("\n增量更新完成！")

if __name__ == "__main__":
    main()