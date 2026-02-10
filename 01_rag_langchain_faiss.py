# 需要安裝之package : pip install langchain langchain-openai langchain-community faiss-cpu
# 使用 OpenAI 的 Embedding 服務，會需要設定環境變數 OPENAI_API_KEY，或是直接在程式碼中指定 API Key (不建議，因為安全性問題)
#   export OPENAI_API_KEY="your_api_key_here" (Linux)
#   setx OPENAI_API_KEY "your_api_key_here" (Windows)
import os

# Langchain refs : 
#   https://reference.langchain.com/python/langchain/
#   https://docs.langchain.com/oss/python/integrations/providers/overview
#   https://reference.langchain.com/python/integrations/
#   https://docs.langchain.com/oss/python/integrations/vectorstores
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangChain v1.0+ 改了結構.
# from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
# 1. 準備原始資料 (可以是 PDF 或長文本)
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

# 2. 資料分段 (Chunking)
# 我們設定每段 100 字，重疊 20 字以保留上下文 (RecursiveCharacterTextSplitter會優先找段落換行、句號)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
)
chunks = text_splitter.split_text(raw_text)
# 將每一段轉成numpy的array格式，並包裝成Document物件，準備送入向量資料庫
docs = [Document(page_content=t) for t in chunks]

# 3. 向量化與建立資料庫 (Embedding & Vector Store)
# 這裡會將文字轉換成向量座標(1536 維)，並存入 FAISS 資料庫中
# FAISS is an in-memory vector store, suitable for small datasets. 
# For larger datasets, consider using a persistent vector store like Pinecone or Weaviate.
# Embedding 使用 OpenAI的text-embedding-3-small, 維度為 1536.
# ref: https://reference.langchain.com/python/integrations/langchain_openai/OpenAIEmbeddings/#langchain_openai.OpenAIEmbeddings
#      https://platform.openai.com/docs/guides/embeddings#how-to-get-embeddings
#
# 此外, OpenAI 做了一個非常聰明的處理：他們將所有的輸出向量都進行了 「歸一化 (Normalization)」。
# 這意味著所有產生的向量長度（模長）都等於 1。
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 需要 OpenAI API Key. text-embedding-3-small 維度為 1536
# 可以強制縮減維度 (比如說是1024或512)，不過會犧牲一些精確度.
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512) 

# "逐doc"叫用OpenAI的Embedding服務 (OpenAIEmbeddings) 來將每一段作embedding後加入FAISS DB (vector_db).
vector_db = FAISS.from_documents(docs, embeddings)      
# 可以使用 distance_strategy=DistanceStrategy.COSINE 來指定FAISS使用cosine距離而非默認的L2距離。
# vector_db = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)      
# 這裡不需要是因為OpenAI的Embedding服務已經將所有向量進行了歸一化處理，所以使用L2距離和Cosine距離在這種情況下是等價/正相關的。
# 所以, 即便 FAISS 預設是用 L2 距離來搜尋，它排出來的「前 k 名」最相關片段，與使用餘弦相似度排出來的結果會是完全一致的。
# 如果未來改用開源的 Embedding 模型（例如從 HuggingFace 下載的模型），有些模型可能沒有預先做歸一化。
# 在這種情況下，FAISS 預設的 L2 距離就不再等同於 Cosine 距離，可能會導致搜尋結果的差異。

# 4. 建立檢索器 (Retriever)
# 這是 RAG 的靈魂：它負責去資料庫「撈」最相關的片段
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # 撈出最相關的 2 段

# 5. 結合 LLM 進行問答 (Generation)
# 先使用最便宜的gpt-4o-mini實驗就好.
# API key 用environment variable引入, 不直接寫在程式碼中.
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# 這邊使用 RetrievalQA 這個現成的 chain 組件，來把檢索器和生成模型串在一起。
# 將來建議改用 LCEL (LangChain Expression Language) 來自己寫一個更靈活的 RAG 流程。
# ex. chain = prompt | llm | StrOutputParser() 這代表：生成 Prompt -> 丟給 LLM -> 將 LLM 的複雜物件直接轉成乾淨的「純字串」。
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 將撈到的資料全部塞進 prompt
    retriever=retriever
)

# 6. 測試提問
# RetrievalQA 把字串 query 交給 retriever。 
# retriever（本質上是 vector_db 的一個介面）發現輸入的是純文字，它會自動呼叫當初存放在內部的 embeddings.embed_query(query)。
# 此時 OpenAI 的 API 會被第二次叫用（將提問轉為向量座標）
# 之後FAISS 拿著這串新產生的座標，去跟資料庫裡的 docs 座標算距離，找出最接近的 k 個片段。
query = "我入職半年，可以領多少旅遊補助？"
response = qa_chain.invoke(query)

print(f"問題: {query}")
print(f"AI 回答: {response['result']}")