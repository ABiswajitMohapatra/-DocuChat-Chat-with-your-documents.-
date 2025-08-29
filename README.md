An interactive RAG (Retrieval-Augmented Generation) application built with LangChain, FAISS, HuggingFace Transformers, and Streamlit.
Upload or connect your documents, ask questions in natural language, and get summarized, chat-style answers like ChatGPT.
🚀 Features

Document Q&A: Retrieve answers directly from your vectorized documents.

Two Response Modes:

⚡ Fast → short, quick answers using a lightweight model.

🧠 Quality → more detailed, well-structured answers using a larger summarization model.

Chat-like Interface: History preserved across multiple turns (just like WhatsApp or ChatGPT).

Clean Summaries: Uses HuggingFace pipeline to generate complete sentences, not raw chunks.

Attractive UI: Built with Streamlit’s st.chat_message for modern bubble-style conversation.

🛠️ Tech Stack

Streamlit
 → Web interface

LangChain
 → Retriever & orchestration

FAISS
 → Vector store for embeddings

HuggingFace Transformers
 → Summarization models

Sentence-Transformers
 → Embeddings (all-mpnet-base-v2)

📂 Project Structure
📦 RAG-Chat
 ┣ 📜 app.py          # Streamlit app (main entry point)
 ┣ 📜 query.py        # Script for quick testing in terminal
 ┣ 📜 vector_store/   # FAISS index (prebuilt from docs)
 ┣ 📜 requirements.txt
 ┗ 📜 README.md

⚙️ Installation & Setup
1. Clone the repo
git clone https://github.com/your-username/rag-chat.git
cd rag-chat

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt

4. Ensure FAISS store exists

If you don’t already have a vector_store/ folder, build it using your documents:

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Example: build from a local text file
with open("your_doc.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
docs = [Document(page_content=chunk) for chunk in chunks]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(docs, embedding)
db.save_local("vector_store")

5. Run the app
streamlit run app.py

🎯 Usage

Open the app in your browser (http://localhost:8501).

Type a question into the chat input.

Choose Fast (quick summaries) or Quality (detailed summaries) from the sidebar.

Get conversational answers based directly on your documents.

📌 Example

User:

What is the conclusion of this paper?

Assistant:

The paper concludes that while the law can never be perfect, its application must strive for fairness and justice. Recent legislative changes show both the challenges and opportunities in improving accessibility and consistency across governments.
