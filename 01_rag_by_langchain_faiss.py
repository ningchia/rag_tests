import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# 1. 準備原始資料 (可以是 PDF 或長文本)
raw_text = """
2024年公司旅遊政策：
1. 凡年資滿一年之員工，可享有每年新台幣 30,000 元之旅遊補助。
2. 旅遊地點不限國內外，但須於出發前兩週提交申請單。
3. 報銷時須提供正式發票，並於回國後 10 天內完成報帳流程。
4. 若年資不滿一年，補助金額按比例計算（每月 2,500 元）。
"""

# 2. 資料分段 (Chunking)
# 我們設定每段 100 字，重疊 20 字以保留上下文
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20
)
chunks = text_splitter.split_text(raw_text)
docs = [Document(page_content=t) for t in chunks]

# 3. 向量化與建立資料庫 (Embedding & Vector Store)
# 這裡會將文字轉換成向量座標，並存入 FAISS 資料庫中
embeddings = OpenAIEmbeddings() # 需要 OpenAI API Key
vector_db = FAISS.from_documents(docs, embeddings)

# 4. 建立檢索器 (Retriever)
# 這是 RAG 的靈魂：它負責去資料庫「撈」最相關的片段
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # 撈出最相關的 2 段

# 5. 結合 LLM 進行問答 (Generation)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 將撈到的資料全部塞進 prompt
    retriever=retriever
)

# 6. 測試提問
query = "我入職半年，可以領多少旅遊補助？"
response = qa_chain.invoke(query)

print(f"問題: {query}")
print(f"AI 回答: {response['result']}")