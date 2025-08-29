from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 1️⃣ Load embedding model and FAISS index
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# 2️⃣ Ask user for a question
query = input("Ask a question: ")

# 3️⃣ Retrieve relevant documents
docs = retriever.get_relevant_documents(query)

# 4️⃣ Clean retrieved documents
clean_docs = []
for d in docs:
    text = d.page_content.replace("<EOS>", "").replace("<pad>", "").strip()
    # Remove LaTeX-like references or multiple spaces
    text = " ".join([w for w in text.split() if not w.startswith('[') and len(w) > 0])
    clean_docs.append(text)

# Combine top-k chunks
full_text = " ".join(clean_docs)

# 5️⃣ Load summarization model
summarizer = pipeline("summarization", model="t5-small", device=0)  # device=0 for GPU, remove if CPU

# 6️⃣ Generate clean summary
summary = summarizer(full_text, max_length=250, min_length=80, do_sample=False)

# 7️⃣ Print the final polished summary
print("\n📌 Clean Summary:\n")
print(summary[0]['summary_text'])
