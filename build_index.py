import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 1. Load documents
docs = []
for file in os.listdir("data"):
    path = os.path.join("data", file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif file.endswith(".txt"):
        loader = TextLoader(path)
    else:
        continue
    docs.extend(loader.load())

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(docs)

# 3. Use HuggingFaceEmbeddings (LangChain-compatible)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 4. Create FAISS index
db = FAISS.from_documents(docs, embedding)

# 5. Save index
db.save_local("vector_store")
print("✅ Index built and saved in vector_store/")
