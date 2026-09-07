"""
MediRAG Evaluation Suite
Measures Retrieval Hit Rate and Answer Groundedness across multi-topic clinical queries.
"""

import time
from src.vector_store import MedicalVectorStore
from src.rag import MediRAGPipeline

BENCHMARK_CASES = [
    {
        "query": "What are the common symptoms of hypertension?",
        "expected_source": "sample_hypertension.pdf",
        "expected_keywords": ["headache", "chest pain", "dizziness", "breathing"]
    },
    {
        "query": "What diagnostic readings confirm Type 2 Diabetes Mellitus?",
        "expected_source": "diabetes_guidelines.pdf",
        "expected_keywords": ["126", "hba1c", "6.5%"]
    },
    {
        "query": "What triggers asthma symptoms and what test confirms diagnosis?",
        "expected_source": "asthma_factsheet.pdf",
        "expected_keywords": ["spirometry", "allergens", "cold air", "wheezing"]
    }
]


def run_evaluation():
    print("=" * 60)
    print("      🩺 RUNNING MULTI-TOPIC MEDIRAG BENCHMARK       ")
    print("=" * 60)

    store = MedicalVectorStore()
    pipeline = MediRAGPipeline(vector_store=store, llm_provider="gemini")

    total_tests = len(BENCHMARK_CASES)
    retrieval_hits = 0
    keyword_matches = 0
    latencies = []

    for idx, test in enumerate(BENCHMARK_CASES, start=1):
        print(f"\n[Test {idx}/{total_tests}] Query: '{test['query']}'")
        
        start_time = time.time()
        result = pipeline.answer_query(test["query"], top_k=3)
        latency = round(time.time() - start_time, 2)
        latencies.append(latency)

        # Evaluate Retrieval Accuracy
        retrieved_sources = [s["source"] for s in result["sources"]]
        hit = any(test["expected_source"] in src for src in retrieved_sources)
        if hit:
            retrieval_hits += 1
            print(f"  ✓ Retrieval : PASS (Matched {test['expected_source']})")
        else:
            print(f"  ✗ Retrieval : FAIL (Found: {retrieved_sources})")

        # Evaluate Grounding
        answer_lower = result["answer"].lower()
        matched_kw = [kw for kw in test["expected_keywords"] if kw.lower() in answer_lower]
        if matched_kw:
            keyword_matches += 1
            print(f"  ✓ Grounding : PASS (Keywords: {matched_kw})")
        else:
            print(f"  ✗ Grounding : FAIL (Missing target keywords)")

        print(f"  ⏱ Latency   : {latency}s")

    # Metrics Summary
    hit_rate = (retrieval_hits / total_tests) * 100
    grounding_score = (keyword_matches / total_tests) * 100
    avg_latency = round(sum(latencies) / len(latencies), 2)

    print("\n" + "=" * 60)
    print("               MULTI-TOPIC BENCHMARK RESULTS                ")
    print("=" * 60)
    print(f"  • Retrieval Hit Rate (Top-3) : {hit_rate:.1f}%")
    print(f"  • Grounding Match Score      : {grounding_score:.1f}%")
    print(f"  • Average End-to-End Latency : {avg_latency}s")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()