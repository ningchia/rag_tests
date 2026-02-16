import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
# 為了讀取 LocalFileStore 裡的 Document 物件，我們需要用到 LangChain 的 load 工具來處理它的序列化格式
from langchain_core.load import loads as langchain_loads

# import pickle  # 用於序列化和反序列化 Python 對象 (但這一版不需要了)
import json     # 改用JSON來處理 LocalFileStore 的序列化問題

# 0. 定義儲存路徑 (需與 13_rag_add_doc_incrementally.py 一致)
VECTOR_DB_PATH = "./faiss_index_save"
BYTE_STORE_PATH = "./parent_doc_storage_save"
hash_key_name = "doc_hash_id" 

def main():
    # 1. 初始化組件並載入現有資料庫
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if not (os.path.exists(VECTOR_DB_PATH) and os.path.exists(BYTE_STORE_PATH)):
        print("錯誤：找不到向量庫或文件庫路徑。")
        return

    print(f"--- 正在讀取資料庫資訊 ---")
    
    # 載入向量庫 (Child Vectors)
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 載入文件庫 (Parent Chunks)
    store = LocalFileStore(BYTE_STORE_PATH)

    # 2. 統計 Child Vectors
    # FAISS 內部的 index.ntotal 代表所有向量點的總數
    total_child_vectors = vectorstore.index.ntotal
    
    # 3. 遍歷向量庫獲取所有唯一的 Parent Hash ID
    # 我們遍歷 docstore 中所有的 Document 來找出不重複的 hash
    all_parent_hashes = set()
    for doc_id in vectorstore.index_to_docstore_id.values():    # 找出在 index_to_docstore_id 裡存在的所有 child doc 的 id
        child_doc = vectorstore.docstore.search(doc_id)         # 找到對應的 child doc
        # 跳過系統初始化的 init 點
        if child_doc.page_content == "init":
            continue
            
        parent_hash = child_doc.metadata.get(hash_key_name)     # 從child doc的metadata裡的 hash_key_name 找出對應的 parent doc 的 hash
        if parent_hash:
            all_parent_hashes.add(parent_hash)

    # 4. 印出統計結果
    print(f"\n[資料庫概況]")
    # 扣除掉 init 點後的實際子向量數
    print(f"● 總子向量數 (Child Vectors): {total_child_vectors - 1}")
    print(f"● 總原始區塊數 (Parent Chunks): {len(all_parent_hashes)}")
    print("-" * 50)

    # 5. 詳細列出每一個 Parent Chunk 的資訊
    # 字串:<數字 => 代表在 f-string 中的對齊方式和寬度設定. <：代表 「左對齊 (Left-align)」。30 或 20：代表 「佔用的總字元寬度」。
    print(f"{'來源檔案':<30} | {'內容預覽 (前後3字)':<20}")
    print("-" * 50)
    
    for p_hash in all_parent_hashes:
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

        # 從 ByteStore 取得原始數據 (這是 bytes, 被serialized成 bytes 的 Document 物件)
        parent_bytes = store.mget([p_hash])[0]
                
        if parent_bytes:
            try:
                # ----------------------------------------------------------
                # OPT-1 : 嘗試用 json 解析
                # ----------------------------------------------------------
                """
                # 將 bytes 轉成字串後解析成字典
                data = json.loads(parent_bytes.decode('utf-8'))

                # print(f"原始 JSON 數據: {data}")  # 調試用，看看拿到的 JSON 結構是什麼樣子
                # json.loads() 後的 data 內容長這樣:
                # {'lc': 1, 'type': 'constructor', 'id': ['langchain', 'schema', 'document', 'Document'], 
                # 'kwargs': {'metadata': {'source': 'card_apply.txt', 'chunk_index': 2, 'hash': 'e5748f...'}, 
                # 'page_content': '原始內容...', 'type': 'Document'}}
                
                # 從 LangChain 的 Serialized 格式中提取內容
                # 注意：內容與 metadata 都在 kwargs 裡面
                kwargs = data.get("kwargs", {})

                # 3. 從 JSON 字典重建 Document 物件
                # 根據 LangChain 的標準 JSON 結構，通常包含 'page_content' 和 'metadata'
                parent_doc = Document(
                    page_content=kwargs.get("page_content") or "",
                    metadata=kwargs.get("metadata", {})
                )
                """
                # ----------------------------------------------------------
                # OPT-2 : 使用 LangChain 內建的 langchain_loads 來自動的把 JSON 直接轉成 Document 物件 
                # ----------------------------------------------------------
                # 建議不要手動去拆解 JSON 的層級，改成使用 LangChain 內建的 load 工具來處理這種格式.
                # 不用手動取 kwargs，因為 langchain_loads 會自動幫你把 JSON 直接轉成 Document 物件。
                # 需要 from langchain_core.load import loads as langchain_loads
                parent_doc = langchain_loads(parent_bytes.decode('utf-8'))

                # ----------------------------------------------------------
                # print(f"成功解析為 Document 物件: {parent_doc}")  # 調試用，看看重建的 Document 是什麼樣子
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 備案：如果還是失敗，才嘗試用 pickle (預防萬一有舊資料)
                import pickle
                parent_doc = pickle.loads(parent_bytes)

            # 現在 parent_doc 是 Document 物件了
            content = parent_doc.page_content.strip()

            # 格式化內容預覽：前3字...後3字
            if len(content) > 6:
                preview = f"{content[:3]}...{content[-3:]}"
            else:
                preview = content
            
            # 從 metadata 取得來源檔名
            source = parent_doc.metadata.get("source", "未知來源")
            chunk_idx = parent_doc.metadata.get("chunk_index", 0)
            
            print(f"{os.path.basename(source)} (區塊 {chunk_idx})".ljust(30) + f" | {preview}")

if __name__ == "__main__":
    main()