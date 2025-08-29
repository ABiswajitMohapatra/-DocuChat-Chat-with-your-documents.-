import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")
st.title("💬 Chat with your Documents")

# -------------------
# Sidebar: Mode Toggle
# -------------------
mode = st.sidebar.radio(
    "⚡ Choose response style:",
    ["Fast (short & quick)", "Quality (detailed & slower)"],
    index=0
)

# -------------------
# Load retriever + models once
# -------------------
@st.cache_resource
def load_models():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local("vector_store", embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # Load two summarizers (fast + quality)
    fast_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    quality_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    return retriever, fast_summarizer, quality_summarizer

retriever, fast_summarizer, quality_summarizer = load_models()

# -------------------
# Chat History
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask a question about your document...")

if query:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve docs (less for speed mode)
    docs = retriever.get_relevant_documents(query)
    top_k = 3 if mode.startswith("Fast") else 5
    clean_docs = [d.page_content.replace("<EOS>", "").replace("<pad>", "").strip() for d in docs[:top_k]]
    final_text = " ".join(clean_docs)

    # Pick summarizer
    summarizer = fast_summarizer if mode.startswith("Fast") else quality_summarizer

    # Build prompt for cleaner answers
    prompt = f"Answer the question clearly and completely.\n\nQuestion: {query}\n\nContext: {final_text}"
    summary = summarizer(prompt, max_length=200, min_length=60, do_sample=False)
    answer = summary[0]['summary_text']

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------
# Render Conversation
# -------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
