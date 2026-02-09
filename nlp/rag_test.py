"""
RAG Model Testing - Quick evaluation of LLM models for barrier/motivator extraction.

Usage: Import in notebook and pass RAGConfig objects to test functions.
"""

import time
import subprocess
from typing import List, Dict
import os

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from nlp.rag_1 import RAGConfig, RAGPipeline

load_dotenv()

# Test data (used by format test and extraction test)
TEST_COMPANY = "012"
TEST_YEAR = "2021"

# Available models (for reference):
# Groq:   llama-3.1-8b-instant, llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it
# Ollama: qwen2.5:3b, gemma3:4b, gemma3:1b, phi3:mini


def unload_ollama(model: str):
    """Unload Ollama model from VRAM."""
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=10)
        time.sleep(1)
    except:
        pass


def get_llm(config: RAGConfig):
    """Create LLM instance from config."""
    if config.llm_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=config.model, api_key=api_key, temperature=config.llm_temperature)
    elif config.llm_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(model=config.model, google_api_key=api_key, temperature=config.llm_temperature)
    return ChatOllama(model=config.model, temperature=config.llm_temperature, num_ctx=config.llm_num_ctx, base_url=config.ollama_base_url)


# Format test uses chunk IDs matching TEST_COMPANY
FORMAT_TEST_PROMPT = f"""Extract BARRIERS to decarbonisation from this text.
OUTPUT FORMAT: [chunk_id]|||verbatim text
If none: NONE_FOUND

TEXT:
[{TEST_COMPANY}_001]
The high cost of green hydrogen remains a significant barrier. Production costs are 4-6x higher than grey hydrogen.

[{TEST_COMPANY}_002]
Infrastructure limitations present another major challenge. Gas pipelines cannot be repurposed for hydrogen.

[{TEST_COMPANY}_003]
Regulatory uncertainty hampers investment. The EU ETS has seen significant price volatility."""


def test_format(config: RAGConfig) -> Dict:
    """Test if model follows output format. Returns dict with format_ok, time, sample."""
    if config.llm_provider == "ollama":
        unload_ollama(config.model)

    llm = get_llm(config)
    llm.invoke("Hi")  # Warm-up

    start = time.time()
    response = llm.invoke(FORMAT_TEST_PROMPT)
    elapsed = time.time() - start

    format_ok = f"[{TEST_COMPANY}_00" in response.content and "|||" in response.content

    if config.llm_provider == "ollama":
        unload_ollama(config.model)

    return {"format_ok": format_ok, "time": elapsed, "sample": response.content[:150]}


def test_extraction(config: RAGConfig) -> Dict:
    """Test real extraction on TEST_COMPANY/TEST_YEAR. Returns dict with barriers, motivators, time."""
    if config.llm_provider == "ollama":
        unload_ollama(config.model)

    pipeline = RAGPipeline(config)
    pipeline.load_from_cache()

    start = time.time()
    barriers, motivators = pipeline.extract_company_year(TEST_COMPANY, TEST_YEAR)
    elapsed = time.time() - start

    if config.llm_provider == "ollama":
        unload_ollama(config.model)

    return {"barriers": barriers, "motivators": motivators, "time": elapsed}


def test_models(configs: List[RAGConfig], skip_extraction: bool = False) -> Dict:
    """Run format and extraction tests on multiple configs."""
    results = {}

    print("=" * 70)
    print(f"RAG MODEL TESTING ({len(configs)} models) - {TEST_COMPANY}/{TEST_YEAR}")
    print("=" * 70)

    for config in configs:
        name = f"{config.llm_provider}/{config.model}"
        print(f"\n--- {name} (ctx={config.llm_num_ctx:,} → {config.max_chunks_per_group} chunks) ---")

        try:
            fmt = test_format(config)
            print(f"    Format: {'PASS' if fmt['format_ok'] else 'FAIL'} ({fmt['time']:.1f}s)")
            print(f"    Output: {fmt['sample']}...")

            result = {"format_ok": fmt["format_ok"], "format_time": fmt["time"]}

            if not skip_extraction and fmt["format_ok"]:
                ext = test_extraction(config)
                print(f"    Extraction: {len(ext['barriers'])}B, {len(ext['motivators'])}M ({ext['time']:.1f}s)")
                result.update({"barriers": ext["barriers"], "motivators": ext["motivators"], "extraction_time": ext["time"]})

            results[name] = result

        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Model':<40} {'Fmt':<6} {'Time':<7} {'B':<5} {'M':<5}")
    print("-" * 70)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<40} ERROR")
        else:
            fmt = "PASS" if r["format_ok"] else "FAIL"
            t = f"{r['format_time']:.1f}s"
            b = len(r["barriers"]) if "barriers" in r else "-"
            m = len(r["motivators"]) if "motivators" in r else "-"
            print(f"{name:<40} {fmt:<6} {t:<7} {b:<5} {m:<5}")

    return results


def compare_extractions(results: Dict):
    """Compare extraction results across models."""
    with_results = {k: v for k, v in results.items() if v.get("barriers")}

    if len(with_results) < 2:
        print("Need 2+ models with extraction results to compare")
        return

    print("\n" + "=" * 70)
    print("EXTRACTION COMPARISON")
    print("=" * 70)

    for name, r in with_results.items():
        print(f"\n--- {name} ({len(r['barriers'])}B / {len(r['motivators'])}M) ---")
        for cid, text in r["barriers"][:3]:
            print(f"  [B] [{cid}] {text[:70]}...")
        for cid, text in r["motivators"][:2]:
            print(f"  [M] [{cid}] {text[:70]}...")

    # Jaccard overlap for barriers and motivators
    print("\n--- Overlap (Jaccard) ---")
    names = list(with_results.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            b1 = set(t.lower()[:50] for _, t in with_results[n1]["barriers"])
            b2 = set(t.lower()[:50] for _, t in with_results[n2]["barriers"])
            m1 = set(t.lower()[:50] for _, t in with_results[n1]["motivators"])
            m2 = set(t.lower()[:50] for _, t in with_results[n2]["motivators"])
            jb = len(b1 & b2) / len(b1 | b2) if b1 | b2 else 0
            jm = len(m1 & m2) / len(m1 | m2) if m1 | m2 else 0
            print(f"  {n1} vs {n2}: B={jb:.0%}, M={jm:.0%}")
