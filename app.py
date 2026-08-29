"""
MediRAG - Medical Knowledge & Document Assistant
Streamlit Web Interface (Clean Native Theme)
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import extract_text_from_pdf, chunk_documents
from src.vector_store import MedicalVectorStore
from src.rag import MediRAGPipeline

load_dotenv()

st.set_page_config(
    page_title="MediRAG | Medical Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS strictly for the alert banner
st.markdown("""
<style>
    .medical-disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #856404;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_store():
    return MedicalVectorStore()


store = get_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.title("🩺 MediRAG Settings")
    st.divider()

    st.subheader("🤖 Model Configuration")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=["gemini", "ollama"],
        index=0,
        help="Use Gemini via API or run locally with Ollama."
    )

    if llm_provider == "gemini":
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash"],
            index=0
        )
    else:
        model_name = st.text_input("Ollama Model Name", value="llama3")

    st.divider()

    st.subheader("📚 Document Management")
    uploaded_file = st.file_uploader(
        "Upload a Medical PDF",
        type=["pdf"],
        help="Upload authoritative clinical guidelines, research papers, or factsheets."
    )

    if uploaded_file is not None:
        save_path = os.path.join("documents", uploaded_file.name)
        os.makedirs("documents", exist_ok=True)

        if uploaded_file.name not in st.session_state.ingested_files:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.get_buffer() if hasattr(uploaded_file, "get_buffer") else uploaded_file.read())

            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                pages = extract_text_from_pdf(save_path)
                chunks = chunk_documents(pages)
                store.add_documents(chunks)
                st.session_state.ingested_files.add(uploaded_file.name)

            st.success(f"Indexed {len(chunks)} chunks from `{uploaded_file.name}`!")

    if st.button("Re-index Sample Documents"):
        sample_path = os.path.join("documents", "sample_hypertension.pdf")
        if os.path.exists(sample_path):
            with st.spinner("Indexing sample documents..."):
                pages = extract_text_from_pdf(sample_path)
                chunks = chunk_documents(pages)
                store.add_documents(chunks)
                st.session_state.ingested_files.add("sample_hypertension.pdf")
            st.success("Sample documents indexed!")


# ---------------- MAIN CHAT PANEL ----------------
st.title("MediRAG — Medical Document Assistant")

st.markdown("""
<div class="medical-disclaimer">
    ⚠️ <strong>Educational Disclaimer:</strong> MediRAG is an AI-powered document exploration tool for educational and research purposes only. 
    It is not a clinical diagnostic tool and does not provide medical advice or replace professional consultations.
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Context & Sources"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page_number']}) — *Similarity: {src['score']}*")
                    st.caption(src["text"])
                    st.divider()

# User Input
if user_query := st.chat_input("Ask a question about your ingested medical documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    pipeline = MediRAGPipeline(
        vector_store=store, 
        llm_provider=llm_provider,
        model_name=model_name
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources & generating grounded response..."):
            result = pipeline.answer_query(user_query)
            st.markdown(result["answer"])

            if result["sources"]:
                with st.expander("🔍 View Retrieved Context & Sources"):
                    for idx, src in enumerate(result["sources"], start=1):
                        st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page_number']}) — *Similarity: {src['score']}*")
                        st.caption(src["text"])
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })