# 🩺 MediRAG: Medical Knowledge & Document Assistant

MediRAG is a modular, retrieval-augmented generation (RAG) assistant designed for clinical guidelines, research papers, and medical document exploration. It enforces strict grounding rules and citation tracing to eliminate hallucinations.

---

## 🌟 Key Features

* **Page-Aware Ingestion:** Extracts and chunks PDF documents while preserving exact page and file metadata (`pymupdf` + `RecursiveCharacterTextSplitter`).
* **Local Dense Embeddings:** Powered by `BAAI/bge-small-en-v1.5` running locally via `sentence-transformers` for fast semantic similarity search.
* **Persistent Vector Store:** ChromaDB instance utilizing cosine similarity distance metric.
* **Dual-LLM Abstraction:** Seamlessly switch between cloud inference (Google Gemini) and local offline inference (Ollama).
* **Strict Source Grounding:** Enforces strict prompt constraints requiring `[Source: document.pdf, Page: X]` citations on every response.
* **Automated Evaluation Suite:** Built-in benchmarking measuring retrieval hit rate, keyword grounding match, and end-to-end latency.

---

## 🏗️ Architecture
[Medical PDF]│▼ (PyMuPDF)[Page-Aware Text Extraction]│▼ (RecursiveCharacterSplitter - 600 char / 100 overlap)[Text Chunks + Metadata]│▼ (BAAI/bge-small-en-v1.5)[ChromaDB Vector Store] <─── (Cosine Similarity Search) <─── [User Query]│▼ (Top-K Chunks + Metadata)[System Prompt Context Injection]│▼[LLM: Gemini / Ollama] ───► [Grounded Answer with Page Citations]
---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/MediRAG.git
cd MediRAG

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
2. Configure API KeysCreate a .env file in the root directory:Code snippetGEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
3. Launch the ApplicationBashstreamlit run app.py
4. Run Benchmark SuiteBashpython evaluate.py
📊 Multi-Topic Clinical Benchmark ResultsMetricResultTargetMulti-Topic Retrieval Hit Rate (Top-3)100.0%> 85%Clinical Grounding Match Score100.0%> 90%Average End-to-End Latency3.83s< 5.0sEvaluated DomainsHypertension, Type 2 Diabetes, AsthmaMulti-domain⚠️ Medical DisclaimerMediRAG is built strictly for educational, informational, and research purposes. It is not a clinical diagnostic tool and does not replace professional medical judgment, advice, or diagnosis.