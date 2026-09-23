# Langfuse Instrumentation Guide

## When to Use This Guide
Use this when instrumenting an existing function or application with Langfuse tracing.

## Core Concepts

### 1. Trace Hierarchy
```
Trace (user session/request)
├── Span (major operation)
│   ├── Observation (LLM call, retrieval, etc.)
│   └── Generation (LLM response details)
└── Score (evaluation metric)
```

### 2. What to Instrument
- **User queries/requests**: Top-level traces
- **Retrieval operations**: Vector search, BM25, hybrid
- **LLM calls**: Chat completions, embeddings
- **Data processing**: Chunking, filtering, transformation
- **External API calls**: External services
- **Evaluation metrics**: RAGAS scores, custom metrics

## Python SDK Best Practices

### 1. Initialization
```python
from langfuse import Langfuse

# Recommended: Environment variables
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
langfuse = Langfuse()

# Or explicit initialization
langfuse = Langfuse(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com"
)
```

### 2. Basic Tracing Patterns

**Pattern A: Decorator (simple)**
```python
from langfuse.decorators import observe

@observe(name="retrieve_documents")
def retrieve_documents(query: str):
    # Your retrieval logic
    return documents
```

**Pattern B: Context Manager (flexible)**
```python
def answer_question(query: str):
    trace = langfuse.trace(name="answer_question")
    
    with trace.span(name="retrieval") as span:
        documents = retrieve_documents(query)
        span.input = {"query": query}
        span.output = {"document_count": len(documents)}
    
    with trace.span(name="generation") as span:
        answer = generate_answer(query, documents)
        span.input = {"query": query, "context": documents}
        span.output = {"answer": answer}
    
    return answer
```

**Pattern C: Manual Observation (granular)**
```python
def complex_pipeline(query: str):
    trace = langfuse.trace(name="rag_pipeline")
    
    # Retrieval observation
    retrieval_obs = trace.observation(
        name="bm25_retrieval",
        input={"query": query, "k": 10},
        output={"documents": [...]},
        metadata={"retrieval_method": "bm25"}
    )
    
    # LLM generation
    generation_obs = trace.observation(
        name="llm_generation",
        input={"prompt": prompt},
        output={"response": answer},
        metadata={"model": "gpt-4", "temperature": 0.7}
    )
    
    return answer
```

### 3. RAG-Specific Instrumentation

**Full RAG Pipeline Example:**
```python
def rag_pipeline(query: str, retrieval_mode: str = "hybrid"):
    trace = langfuse.trace(
        name="rag_pipeline",
        metadata={"query": query, "retrieval_mode": retrieval_mode}
    )
    
    # 1. Retrieval phase
    with trace.span(name=f"{retrieval_mode}_retrieval") as span:
        if retrieval_mode == "bm25":
            results = bm25_retrieve(query)
        elif retrieval_mode == "vector":
            results = vector_retrieve(query)
        else:  # hybrid
            bm25_results = bm25_retrieve(query)
            vector_results = vector_retrieve(query)
            results = fuse_rrf(bm25_results, vector_results)
        
        span.metadata = {
            "candidate_count": len(results),
            "retrieval_mode": retrieval_mode
        }
    
    # 2. Evidence selection
    with trace.span(name="evidence_selection") as span:
        evidence = evaluate_evidence(results, query)
        span.metadata = {
            "should_answer": evidence.should_answer,
            "confidence": evidence.confidence,
            "selected_chunks": evidence.selected_chunks
        }
    
    # 3. Generation
    with trace.span(name="llm_generation") as span:
        answer = ask_llm(query, evidence.context)
        span.metadata = {
            "model": config.LLM_MODEL,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        span.output = {"answer": answer}
    
    # 4. Evaluation (optional)
    if config.ENABLE_RAGAS_EVAL:
        with trace.span(name="ragas_evaluation") as span:
            scores = evaluate_with_ragas(query, answer, evidence.context)
            for metric, value in scores.items():
                trace.score(
                    name=f"ragas_{metric}",
                    value=value,
                    metadata={"retrieval_mode": retrieval_mode}
                )
    
    return answer
```

### 4. Metadata Best Practices

**Always include:**
- Operation parameters
- Timestamps/durations
- Resource identifiers
- Configuration versions

**RAG-specific metadata:**
```python
metadata = {
    # Retrieval
    "retrieval_method": "hybrid",
    "top_k": 10,
    "rrf_k": 60,
    
    # Evidence
    "evidence_threshold": 0.7,
    "selected_chunk_count": 3,
    "evidence_confidence": 0.85,
    
    # Generation
    "llm_model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    
    # Evaluation
    "ragas_version": "0.1.20",
    "evaluation_metrics": ["faithfulness", "answer_relevancy"],
    
    # System
    "vectorstore": "chroma",
    "embedding_model": "text-embedding-3-small",
    "app_version": "1.2.0"
}
```

### 5. Error Handling

**Never break your app:**
```python
def safe_trace(name: str):
    try:
        return langfuse.trace(name=name)
    except Exception as e:
        logger.warning(f"Langfuse trace failed: {e}")
        return None

def instrumented_function(query: str):
    trace = safe_trace("instrumented_function")
    if not trace:
        # Fall back to normal execution
        return normal_function(query)
    
    try:
        # Instrumented logic
        with trace.span(name="operation"):
            result = normal_function(query)
            trace.score(name="success", value=1.0)
        return result
    except Exception as e:
        if trace:
            trace.score(name="error", value=0.0, metadata={"error": str(e)})
        raise
```

### 6. Performance Considerations

**Batching:**
```python
# Manual flush control
langfuse.trace(name="operation")
# ... operations ...
langfuse.flush()  # Explicit flush

# Or use async
import asyncio
from langfuse import Langfuse

async def async_operation():
    langfuse = Langfuse()
    trace = await langfuse.atrace(name="async_op")
    # ... async operations ...
    await langfuse.aflush()
```

**Sampling (high-volume apps):**
```python
import random

def should_sample() -> bool:
    return random.random() < 0.1  # 10% sampling rate

def process_query(query: str):
    if should_sample():
        trace = langfuse.trace(name="sampled_query")
        # ... instrumented logic ...
    else:
        # ... uninstrumented logic ...
```

### 7. Testing Instrumentation

**Verify traces appear:**
```bash
# After running your app, query traces
npx langfuse-cli api traces list --limit 5

# Check specific trace
npx langfuse-cli api traces get <trace_id>
```

**Integration tests:**
```python
def test_instrumentation():
    # Mock Langfuse in tests
    with patch('langfuse.Langfuse'):
        result = instrumented_function("test query")
        assert result is not None
```

## Common Pitfalls to Avoid

1. **❌ Forgetting to flush**: Traces won't be sent
2. **❌ Not handling errors**: App crashes if Langfuse fails
3. **❌ Too much metadata**: Performance impact
4. **❌ Inconsistent naming**: Hard to query/analyze
5. **❌ Missing timestamps**: Can't calculate durations
6. **❌ Not sampling**: Overwhelming trace volume

## Next Steps After Instrumentation

1. **Verify traces** appear in Langfuse dashboard
2. **Add scores** for evaluation metrics
3. **Set up alerts** for anomalies
4. **Create dashboards** for key metrics
5. **Implement experiments** with prompt versions
6. **Build datasets** from production traces