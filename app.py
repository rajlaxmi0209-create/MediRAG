"""
MediRAG - Executive Clinical Intelligence & Document Exploration Assistant
Streamlit Web Interface with Fluid Keyframe Transitions, Dynamic Theming, Live Telemetry & Audit Export
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import extract_text_from_pdf, chunk_documents
from src.vector_store import MedicalVectorStore
from src.rag import MediRAGPipeline

load_dotenv()

st.set_page_config(
    page_title="MediRAG | Clinical Intelligence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- THEME INITIALIZATION ----------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

is_dark = (st.session_state.theme_mode == "Dark")

if is_dark:
    THEME = {
        "bg_app": "#0a0d14",
        "bg_panel": "#0f1420",
        "bg_card": "#141b2a",
        "bg_card_secondary": "#1a2337",
        "bg_widget": "#162032",
        "border": "#212d45",
        "border_subtle": "#182235",
        "text_primary": "#f1f5f9",
        "text_secondary": "#94a3b8",
        "accent": "#a3d139",
        "accent_hover": "#8ebe2d",
        "accent_glow": "rgba(163, 209, 57, 0.2)",
        "btn_text": "#0a0d14",
        "warning_bg": "rgba(245, 158, 11, 0.08)",
        "warning_border": "rgba(245, 158, 11, 0.25)",
        "warning_text": "#fbbf24"
    }
else:
    THEME = {
        "bg_app": "#f8fafc",
        "bg_panel": "#ffffff",
        "bg_card": "#ffffff",
        "bg_card_secondary": "#f1f5f9",
        "bg_widget": "#ffffff",
        "border": "#cbd5e1",
        "border_subtle": "#e2e8f0",
        "text_primary": "#0f172a",
        "text_secondary": "#475569",
        "accent": "#65a30d",
        "accent_hover": "#4d7c0f",
        "accent_glow": "rgba(101, 163, 13, 0.15)",
        "btn_text": "#ffffff",
        "warning_bg": "#fffbeb",
        "warning_border": "#fef3c7",
        "warning_text": "#b45309"
    }

# ---------------- DYNAMIC CSS WITH FLUID KEYFRAME TRANSITIONS ----------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Smooth page-load keyframes */
    @keyframes smoothFadeIn {{
        0% {{
            opacity: 0;
            transform: translateY(6px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulseGlow {{
        0%, 100% {{
            box-shadow: 0 0 15px {THEME['accent_glow']};
        }}
        50% {{
            box-shadow: 0 0 25px {THEME['accent_glow']};
        }}
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {THEME['bg_app']} !important;
        color: {THEME['text_primary']} !important;
    }}

    /* Global smooth color transitions on theme switch */
    *, *::before, *::after {{
        transition: background-color 0.25s ease, border-color 0.25s ease, color 0.2s ease, box-shadow 0.2s ease, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Apply fadeIn transition to main block content */
    .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 5rem !important;
        max-width: 1200px !important;
        animation: smoothFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"], 
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: {THEME['bg_panel']} !important;
        border-right: 1px solid {THEME['border']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {THEME['text_primary']} !important;
    }}

    /* Buttons with smooth spring micro-interaction */
    .stButton > button, 
    div[data-testid="stDownloadButton"] > button,
    section[data-testid="stFileUploaderDropzone"] button,
    button[kind="secondary"] {{
        background-color: {THEME['accent']} !important;
        color: {THEME['btn_text']} !important;
        border: none !important;
        border-radius: 9px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.52rem 1.1rem !important;
        box-shadow: 0 2px 8px {THEME['accent_glow']} !important;
        cursor: pointer !important;
    }}
    .stButton > button:hover, 
    div[data-testid="stDownloadButton"] > button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: {THEME['accent_hover']} !important;
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 18px {THEME['accent_glow']} !important;
    }}
    .stButton > button:active, 
    div[data-testid="stDownloadButton"] > button:active {{
        transform: translateY(0) scale(0.98) !important;
    }}

    /* File Dropzone */
    section[data-testid="stFileUploaderDropzone"] {{
        background-color: {THEME['bg_card']} !important;
        border: 1.5px dashed {THEME['border']} !important;
        border-radius: 12px !important;
    }}
    section[data-testid="stFileUploaderDropzone"] * {{
        color: {THEME['text_primary']} !important;
    }}
    section[data-testid="stFileUploaderDropzone"] small {{
        color: {THEME['text_secondary']} !important;
    }}

    /* Radio Items */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span {{
        color: {THEME['text_primary']} !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stRadio"] > div {{
        gap: 8px !important;
    }}

    /* Expander Elements */
    div[data-testid="stExpander"] {{
        background-color: {THEME['bg_card']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        animation: smoothFadeIn 0.25s ease-in-out !important;
    }}
    div[data-testid="stExpander"] details {{
        background-color: {THEME['bg_card']} !important;
    }}
    div[data-testid="stExpander"] summary {{
        background-color: {THEME['bg_card_secondary']} !important;
        color: {THEME['text_primary']} !important;
        border-bottom: 1px solid {THEME['border']} !important;
        padding: 12px 16px !important;
    }}
    div[data-testid="stExpander"] summary * {{
        background-color: transparent !important;
        color: {THEME['text_primary']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        background-color: {THEME['bg_card']} !important;
        padding: 16px !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] * {{
        color: {THEME['text_primary']} !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stCaptionContainer"],
    div[data-testid="stExpander"] div[data-testid="stCaptionContainer"] * {{
        color: {THEME['text_secondary']} !important;
        line-height: 1.6 !important;
    }}

    /* Bottom Chat Bar */
    div[data-testid="stBottom"], div[data-testid="stBottom"] > div, footer {{
        background-color: {THEME['bg_app']} !important;
        border-top: 1px solid {THEME['border_subtle']} !important;
    }}
    div[data-testid="stChatInput"] > div {{
        background-color: {THEME['bg_card']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.03) !important;
    }}
    div[data-testid="stChatInput"] textarea {{
        color: {THEME['text_primary']} !important;
        -webkit-text-fill-color: {THEME['text_primary']} !important;
        background-color: transparent !important;
    }}
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {THEME['text_secondary']} !important;
    }}

    /* High-contrast chat bubbles with fade-in */
    div[data-testid="stChatMessage"] {{
        background: {THEME['bg_card']} !important;
        border: 1px solid {THEME['border']} !important;
        border-radius: 14px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
        animation: smoothFadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }}
    div[data-testid="stChatMessage"] * {{
        color: {THEME['text_primary']} !important;
    }}
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] ul,
    div[data-testid="stChatMessage"] ol,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] strong,
    div[data-testid="stChatMessage"] em {{
        color: {THEME['text_primary']} !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
    }}
    div[data-testid="stChatMessage"] li::marker {{
        color: {THEME['accent']} !important;
        font-weight: bold !important;
    }}

    /* Telemetry Metric Cards */
    div[data-testid="metric-container"] {{
        background: {THEME['bg_card_secondary']} !important;
        border: 1px solid {THEME['border']} !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        animation: smoothFadeIn 0.25s ease !important;
    }}
    div[data-testid="stMetricValue"] > div {{
        color: {THEME['accent']} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.3rem !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: {THEME['text_secondary']} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Citation Badges */
    code {{
        background: {THEME['bg_card_secondary']} !important;
        color: {THEME['accent']} !important;
        padding: 3px 7px !important;
        border-radius: 6px !important;
        font-size: 0.85em !important;
        border: 1px solid {THEME['border']} !important;
    }}

    /* Hero & Structure Cards */
    .hero-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 20px;
        padding: 40px 32px;
        text-align: center;
        margin-bottom: 28px;
        animation: smoothFadeIn 0.4s ease-out;
    }}
    .hero-title {{
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0 0 10px 0;
        color: {THEME['text_primary']};
    }}
    .hero-title span {{
        color: {THEME['accent']};
    }}
    .hero-subtitle {{
        font-size: 1.08rem;
        color: {THEME['text_secondary']};
        max-width: 620px;
        margin: 0 auto 24px auto;
        line-height: 1.6;
    }}
    .spec-pill {{
        background: {THEME['bg_card_secondary']};
        border: 1px solid {THEME['border']};
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        color: {THEME['text_primary']};
        display: inline-block;
    }}
    .pillar-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        animation: smoothFadeIn 0.45s ease-out;
    }}
    .pillar-title {{
        font-size: 0.98rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        margin-bottom: 8px;
    }}
    .pillar-desc {{
        font-size: 0.86rem;
        color: {THEME['text_secondary']};
        line-height: 1.55;
        margin: 0;
    }}
    .clinical-notice {{
        background: {THEME['warning_bg']};
        border: 1px solid {THEME['warning_border']};
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 0.85rem;
        font-weight: 500;
        color: {THEME['warning_text']};
        margin-bottom: 24px;
        animation: smoothFadeIn 0.3s ease-out;
    }}
    .file-item {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }}
    .file-name {{
        font-size: 0.84rem;
        font-weight: 600;
        color: {THEME['text_primary']};
        font-family: 'JetBrains Mono', monospace;
    }}
    .file-desc {{
        font-size: 0.74rem;
        color: {THEME['text_secondary']};
    }}
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

if "active_view" not in st.session_state:
    st.session_state.active_view = "landing"

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("### 🩺 **MediRAG Cockpit**")
    st.caption("Clinical Decision Support Retrieval Pipeline")

    toggle_label = "🌙 Dark Theme" if is_dark else "☀️ Light Theme"
    toggle_val = st.toggle(toggle_label, value=is_dark)
    new_mode = "Dark" if toggle_val else "Light"
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        st.rerun()

    st.divider()

    if st.session_state.active_view == "chat":
        if st.button("⬅ Return to Overview", use_container_width=True):
            st.session_state.active_view = "landing"
            st.rerun()
    else:
        if st.button("💬 Launch Workspace", use_container_width=True):
            st.session_state.active_view = "chat"
            st.rerun()

    st.divider()

    st.markdown("#### 📚 **Reference Library**")
    guidelines = [
        ("sample_hypertension.pdf", "Hypertension Clinical Guidelines"),
        ("diabetes_guidelines.pdf", "Type 2 Diabetes Diagnostics"),
        ("asthma_factsheet.pdf", "Asthma Spirometry & Triggers"),
        ("9789240033986-eng.pdf", "WHO Pharmacological Manual"),
    ]

    for fname, desc in guidelines:
        st.markdown(f"""
        <div class="file-item">
            <div class="file-name">📄 {fname}</div>
            <div class="file-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Sync Corpus Chunks", use_container_width=True):
        with st.spinner("Re-syncing vector database..."):
            docs_folder = "documents"
            total = 0
            if os.path.exists(docs_folder):
                for f in os.listdir(docs_folder):
                    if f.endswith(".pdf"):
                        p = os.path.join(docs_folder, f)
                        pgs = extract_text_from_pdf(p)
                        chks = chunk_documents(pgs)
                        store.add_documents(chks)
                        st.session_state.ingested_files.add(f)
                        total += len(chks)
            st.success(f"Indexed {total} passages!")

    st.divider()

    st.markdown("#### 📥 **Ingest Custom Guideline**")
    custom_pdf = st.file_uploader("Upload Clinical Document", type=["pdf"], label_visibility="collapsed")
    if custom_pdf is not None:
        if st.button("Process Document", use_container_width=True):
            save_path = os.path.join("documents", custom_pdf.name)
            os.makedirs("documents", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(custom_pdf.get_buffer() if hasattr(custom_pdf, "get_buffer") else custom_pdf.read())

            with st.spinner(f"Ingesting '{custom_pdf.name}'..."):
                pages = extract_text_from_pdf(save_path)
                chunks = chunk_documents(pages)
                store.add_documents(chunks)
                st.session_state.ingested_files.add(custom_pdf.name)
            st.success(f"Vectorized {len(chunks)} chunks!")

    st.divider()

    st.markdown("#### 🤖 **Inference Engine**")
    llm_provider = "gemini"
    model_name = st.radio(
        "Gemini Model Variant",
        options=["gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-flash"],
        index=0
    )


# ---------------- MAIN VIEW ROUTING ----------------

if st.session_state.active_view == "landing":
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 3.2rem; line-height: 1; margin-bottom: 12px;">🩺</div>
        <h1 class="hero-title">Medi<span>RAG</span></h1>
        <p class="hero-subtitle">
            Deterministic clinical retrieval-augmented generation. Grounding medical inquiries in authoritative guidelines with exact page-level citations and verified evidence.
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <span class="spec-pill">⚡ Dense BGE-small Embeddings</span>
            <span class="spec-pill">🎯 ChromaDB Cosine Store</span>
            <span class="spec-pill">🔒 Zero-Hallucination Guardrail</span>
            <span class="spec-pill">⏱ ~3.8s Mean Latency</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2, b_col3 = st.columns([1, 1.2, 1])
    with b_col2:
        if st.button("🚀 Open Clinical Workspace", use_container_width=True):
            st.session_state.active_view = "chat"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-title">📑 Page-Aware Audit Trail</div>
            <p class="pillar-desc">
                Preserves original PDF metadata and exact physical page indices via PyMuPDF for verifiable clinical references.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-title">🛡️ Grounding Guardrails</div>
            <p class="pillar-desc">
                Enforces context-only answers with deterministic references, systematically preventing clinical hallucinations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-title">📊 Benchmark Validated</div>
            <p class="pillar-desc">
                Attains 100% retrieval hit rate across evaluated Hypertension, Diabetes, and Asthma clinical guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.markdown("### 🩺 **MediRAG Clinical Workspace**")
    with head_col2:
        st.markdown(f"<div style='text-align:right; margin-top:8px;'><span class='spec-pill'>● Pipeline Active ({model_name})</span></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="clinical-notice">
        ⚠️ <strong>Notice:</strong> MediRAG is designed for clinical reference and research exploration. It does not replace certified professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<small style='font-weight:700; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Sample Inquiries:</small>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🩸 Diabetes Diagnostic Cutoffs", use_container_width=True):
            st.session_state.suggested_prompt = "What diagnostic blood glucose and HbA1c values confirm Type 2 Diabetes?"
            st.rerun()
    with p_col2:
        if st.button("🫁 Asthma Diagnostic Criteria", use_container_width=True):
            st.session_state.suggested_prompt = "What are the primary symptom triggers and spirometry criteria for Asthma?"
            st.rerun()
    with p_col3:
        if st.button("❤️ Hypertension First-Line Drugs", use_container_width=True):
            st.session_state.suggested_prompt = "What pharmacological treatments are recommended as first-line for Stage 1 Hypertension?"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Render History using strictly unique enumeration keys
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "metrics" in msg and msg["metrics"]:
                m = msg["metrics"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Response Time", f"{m['latency']}s")
                m2.metric("Retrieved Chunks", m['chunks'])
                m3.metric("Vector Similarity", f"{m['top_score']}")

            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 Clinical Evidence & Citations"):
                    for s_idx, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[{s_idx}] {src['source']}** `Page {src['page_number']}` · *Score: {src['score']}*")
                        st.caption(src["text"])
                        st.divider()

                audit_report = f"""====================================================
MEDIRAG CLINICAL AUDIT REPORT
====================================================
Model: {model_name} (Provider: {llm_provider})

ASSISTANT RESPONSE:
{msg['content']}

CITATIONS:
""" + "\n".join([f"[{i+1}] {s['source']} (Page {s['page_number']}) - Score: {s['score']}\nExcerpt: {s['text'][:250]}...\n" for i, s in enumerate(msg["sources"])])

                st.download_button(
                    label="📥 Export Clinical Audit Report",
                    data=audit_report,
                    file_name=f"clinical_audit_{idx}.txt",
                    mime="text/plain",
                    key=f"audit_download_history_{idx}"
                )

    user_query = st.chat_input("Ask a clinical query regarding your ingested documents...")
    if st.session_state.suggested_prompt:
        user_query = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        pipeline = MediRAGPipeline(
            vector_store=store,
            llm_provider=llm_provider,
            model_name=model_name
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing vectors & generating grounded answer..."):
                t0 = time.perf_counter()
                result = pipeline.answer_query(user_query)
                elapsed = round(time.perf_counter() - t0, 2)
                top_score = round(result["sources"][0]["score"], 3) if result["sources"] else 0.0

                c1, c2, c3 = st.columns(3)
                c1.metric("Response Time", f"{elapsed}s")
                c2.metric("Retrieved Chunks", len(result["sources"]))
                c3.metric("Vector Similarity", f"{top_score}")

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("🔍 Clinical Evidence & Citations"):
                        for s_idx, src in enumerate(result["sources"], start=1):
                            st.markdown(f"**[{s_idx}] {src['source']}** `Page {src['page_number']}` · *Score: {src['score']}*")
                            st.caption(src["text"])
                            st.divider()

                    audit_report = f"""====================================================
MEDIRAG CLINICAL AUDIT REPORT
====================================================
Query: {user_query}
Model: {model_name} (Provider: {llm_provider})

ASSISTANT RESPONSE:
{result['answer']}

CITATIONS:
""" + "\n".join([f"[{i+1}] {s['source']} (Page {s['page_number']}) - Score: {s['score']}\nExcerpt: {s['text'][:250]}...\n" for i, s in enumerate(result["sources"])])

                    st.download_button(
                        label="📥 Export Clinical Audit Report",
                        data=audit_report,
                        file_name="clinical_audit_live.txt",
                        mime="text/plain",
                        key=f"audit_download_live_{len(st.session_state.messages)}"
                    )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "metrics": {
                "latency": elapsed,
                "chunks": len(result["sources"]),
                "top_score": top_score
            }
        })