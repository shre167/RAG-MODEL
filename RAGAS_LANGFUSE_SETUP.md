# RAGAS + Langfuse Integration Setup

## Installation

1. **Install dependencies:**
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables** in `.env`:
```env
# Langfuse Configuration
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=your_public_key_here
LANGFUSE_SECRET_KEY=your_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL

# RAGAS Configuration
ENABLE_RAGAS_EVAL=true
RAGAS_EVAL_MODES=bm25,vector,hybrid  # Modes to evaluate
RAGAS_COMPARE_ALL_RETRIEVERS=true  # Compare all methods for each query

# Existing LLM/Embedding config (no changes needed)
LLM_API_KEY=your_gemini_api_key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=text-embedding-3-small
```

## What Was Implemented

### ✅ Langfuse Observability
**Location:** `src/rag_pipeline/langfuse_tracing.py`

Creates hierarchical trace for every query:
```
Query
  ├── BM25 Retrieval (chunk IDs, ranks, scores)
  ├── Dense Retrieval (chunk IDs, ranks, distances)
  ├── Hybrid/RRF Retrieval (RRF scores, fusion calculations)
  ├── Evidence Selection (selected chunks, evidence level, reason)
  ├── LLM Generation (model, prompt, answer, latency)
  └── RAGAS Evaluation (all metrics attached as scores)
```

**API Used:** Langfuse SDK v2+ (`get_client()`, `start_as_current_observation()`)

**Integration Point:** `src/rag_pipeline/observability.py::observe_pipeline_answer()`
- Called from `pipeline.py::answer_question()` after response generation
- Does NOT modify retrieval logic
- Observes actual system behavior

### ✅ RAGAS Evaluation
**Location:** `src/rag_pipeline/ragas_evaluation.py`

**Reference-Free Metrics (always run):**
- `faithfulness`: Answer grounded in retrieved contexts
- `answer_relevancy`: Answer relevance to query
- `context_precision`: Relevance of retrieved contexts

**Reference-Based Metrics (requires benchmark labels):**
- `context_recall`: Ground truth coverage by contexts (requires `ground_truth_contexts`)

**Integration Point:** `src/rag_pipeline/observability.py::observe_pipeline_answer()`
- Evaluates ACTUAL pipeline output (no fabrication)
- Uses real contexts sent to LLM
- Uses real generated answer

### ✅ RAGAS ↔ Langfuse Connection
**Location:** `src/rag_pipeline/langfuse_tracing.py::_attach_ragas_scores()`

Each RAGAS metric becomes a Langfuse score:
```python
observation.score(
    name=f"ragas_{retrieval_mode}_{metric_name}",
    value=float(value),
    metadata={"retrieval_method": mode, "query": query, ...}
)
```

**Result:** All RAGAS scores appear in the same Langfuse trace as the RAG execution.

### ✅ BM25 vs Dense vs Hybrid Comparison
**Configuration:** Set `RAGAS_COMPARE_ALL_RETRIEVERS=true`

**Behavior:**
- User query runs with selected mode (e.g., `hybrid`)
- Pipeline automatically re-runs with `bm25` and `vector` modes
- Each mode gets independent RAGAS evaluation
- All results appear in same Langfuse trace
- Comparison data: `response["ragas"]["by_retrieval_mode"]`

**Independence Guarantee:**
- Each retrieval method runs separately
- No fallback mixing (BM25 doesn't secretly help Dense)
- No forced chunks
- Evidence thresholds unchanged

### ✅ Terminal Output
**Format:**
```
================================================================================
Query: What causes a solar eclipse?
Method: HYBRID
--------------------------------------------------------------------------------
Answer: A solar eclipse occurs when...
--------------------------------------------------------------------------------
RAGAS Metrics:
  Faithfulness: 0.8750
  Answer Relevancy: 0.9200
  Context Precision: 0.8333
--------------------------------------------------------------------------------
Langfuse Trace ID: abc123-def456-...
================================================================================
```

## Testing

### Basic Test (Single Query)
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.answer_question(
    "What is cosmology?",
    retrieval_mode="hybrid"
)

# Check RAGAS scores
print(response["ragas"]["by_retrieval_mode"]["hybrid"]["scores"])

# Check Langfuse trace ID
print(response["observability"]["langfuse_trace_id"])
```

### Comparison Test (All Retrievers)
```python
# Automatically compares BM25, Dense, Hybrid if RAGAS_COMPARE_ALL_RETRIEVERS=true
response = pipeline.answer_question("What is JWST?")

for mode, ragas_result in response["ragas"]["by_retrieval_mode"].items():
    print(f"{mode}: {ragas_result['scores']}")
```

## Failure Classification

Failures are recorded in the trace without hidden fallbacks:

1. **Retrieval Failure:** `retrieval.candidate_count == 0`
2. **Evidence Selection Failure:** `evidence.should_answer == False`
3. **Generation Failure:** `generation.status == "error"`
4. **Faithfulness Failure:** `ragas.faithfulness < threshold`
5. **Answer Relevance Failure:** `ragas.answer_relevancy < threshold`
6. **Successful Answer:** All checks pass

## RAGAS Compatibility Solution

**Problem:** RAGAS 0.4+ imports `langchain_community.chat_models.vertexai` which was moved to `langchain-google-vertexai`

**Solution:** Using RAGAS 0.1.20 which:
- Works with Python 3.13
- Compatible with current LangChain ecosystem
- Uses stable API (`evaluate()`, `Dataset.from_dict()`)
- No vertexai import issues

**LLM/Embeddings:** Uses existing Gemini configuration via `langchain_openai.ChatOpenAI` with OpenAI-compatible endpoint

## What Was NOT Changed

✅ BM25 retrieval algorithm (unchanged)  
✅ Dense/vector retrieval algorithm (unchanged)  
✅ RRF fusion formula (k=60, unchanged)  
✅ Evidence selection thresholds (unchanged)  
✅ LLM generation logic (unchanged)  
✅ Chroma vector database (no reindexing needed)  
✅ Existing evaluation code (preserved)

## Files Modified

1. `requirements.txt` - Added langfuse, ragas, datasets, langchain-community
2. `src/rag_pipeline/ragas_evaluation.py` - Updated for RAGAS 0.1.20 API
3. `src/rag_pipeline/observability.py` - Added terminal output formatting

## Files Unchanged (Already Implemented)

1. `src/rag_pipeline/langfuse_tracing.py` - Already complete
2. `src/rag_pipeline/observability_config.py` - Already complete
3. `src/rag_pipeline/pipeline.py` - Integration already wired
4. All retrieval logic - Untouched

## Next Steps

1. Install dependencies
2. Configure `.env` with Langfuse credentials
3. Run test query
4. Check Langfuse dashboard for trace
5. Verify RAGAS scores in terminal output
