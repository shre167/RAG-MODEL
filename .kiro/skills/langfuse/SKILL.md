---
name: langfuse
description: >-
  Interact with Langfuse and access its documentation: tracing, monitoring, creating datasets, running experiments, and evaluating AI applications. Use when needing to (1) query or modify Langfuse data, (2) look up Langfuse documentation, concepts, integration guides, a feature or SDK usage, or (3) do any AI engineering task (AI observability, prompt engineering/management, evaluation and evaluator management, experimentation, dataset management, evaluation-driven CI/CD, feedback collection). Invoke it for tasks in this scope even when Langfuse is not configured or explicitly mentioned.
allowed-tools:
  - WebFetch(domain:langfuse.com)
  - Bash(curl *langfuse.com/*)
  - Bash(npx langfuse-cli api __schema *)
  - Bash(npx langfuse-cli api * --help *)
  - Bash(npx langfuse-cli api * list *)
  - Bash(npx langfuse-cli api * get *)
  - Bash(bunx langfuse-cli api __schema *)
  - Bash(bunx langfuse-cli api * --help *)
  - Bash(bunx langfuse-cli api * list *)
  - Bash(bunx langfuse-cli api * get *)
---

# Langfuse

This skill helps you use Langfuse effectively across all common workflows: instrumenting applications, migrating prompts, debugging traces, and accessing data programmatically.

## Core Principles

Follow these principles for ALL Langfuse work:

1. **Documentation First**: NEVER implement based on memory. Always fetch current docs before writing code (Langfuse updates frequently) See the section below on how to access documentation.
2. **CLI for Data Access**: Use `langfuse-cli` when querying/modifying Langfuse data. See the section below on how to use the CLI.
3. **Best Practices by Use Case**: Read the relevant reference below use-case-specific guidelines before asking the user for more details or implementing.
4. **Use latest Langfuse versions**: Unless the user specified otherwise or there's a good reason, always use the latest version of Langfuse SDKs/APIs. Even if you're only creating a plan for another agent to execute, be explicit about the exact version to use.
5. **If you guide the user through UI** and are unsure about a label or location, inspect the user's screenshots or ask to see the relevant screen. Do not assume UI labels have the exact same names as API, SDK, or CLI fields.

## Use case specific references

- instrumenting an existing function/application: references/instrumentation.md
- creating or getting to a good (evaluation) dataset to measure quality or test for regressions in AI systems: references/create-dataset.md
- migrating prompts from a codebase into Langfuse: references/prompt-migration.md
- creating a prompt or changing any part of an existing prompt, including small edits and debugging/tuning: references/prompt-engineering.md
- setting up evals when the user needs to identify gaps across signal capture, monitoring, and evaluator metrics ("I have traces, how do I set up evals?"): references/setting-up-evals.md
- capturing user feedback signals (explicit ratings, behavioral events, conversation signals, task outcomes) as scores: references/user-feedback.md
- further tips on using the Langfuse CLI: references/cli.md
- preparing a Langfuse project for the v4 platform migration: references/v4-project-migration.md
- judge calibration (LLM-as-a-Judge reliability, simple accuracy checks, advanced split-based validation, confusion matrices, and metric ingestion): references/judge-calibration.md
- systematic error analysis when requested directly or eval setup still lacks concrete failure modes after agent-led trace inspection: references/error-analysis.md
- setting up CI/CD experiment gates with `langfuse/experiment-action`: references/ci-cd.md
- submitting feedback about this skill: references/skill-feedback.md

## 1. Langfuse API via CLI

Use the `langfuse-cli` to interact with the full Langfuse REST API from the command line. Run via npx (no install required):

Start by discovering the schema and available arguments:

```bash
# Discover all available resources
npx langfuse-cli api __schema

# List actions for a resource
npx langfuse-cli api <resource> --help

# List resources (e.g., traces, observations, scores)
npx langfuse-cli api <resource> list --limit 5

# Get a specific resource
npx langfuse-cli api <resource> get <id>
```

**Important:** The CLI needs environment variables set (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`) just like the SDK.

## 2. Accessing Documentation

Fetch the latest documentation directly from Langfuse.com before implementing or advising:

```bash
# Fetch specific pages
curl -s https://langfuse.com/docs/api-and-data-platform/sdks/python | head -100

# Or use WebFetch tool with domain restriction
WebFetch(domain:langfuse.com, url:/docs/api-and-data-platform/sdks/python)
```

## 3. Common Workflows

### Instrumenting an Application
1. **Add SDK**: `pip install langfuse` (Python) or equivalent
2. **Initialize**: Set environment variables or initialize in code
3. **Trace everything**: Wrap functions with `@observe()` or create traces manually
4. **Add scores**: Attach evaluation metrics to observations
5. **Test**: Verify traces appear in Langfuse

### Querying Traces
```bash
# Get recent traces
npx langfuse-cli api traces list --limit 10

# Get trace details
npx langfuse-cli api traces get <trace_id>

# Filter by timestamp, metadata, etc.
npx langfuse-cli api traces list --limit 10 --filter 'timestamp > "2024-01-01"'
```

### Managing Prompts
```bash
# List prompts
npx langfuse-cli api prompts list

# Create/update prompts
npx langfuse-cli api prompts create --name "rag-prompt" --prompt "Answer: {context}\n\nQuestion: {question}"
```

## 4. Best Practices

### Tracing
- Use hierarchical traces (trace → span → observation)
- Add meaningful metadata to all operations
- Use consistent naming conventions
- Set trace/span IDs for correlation

### Evaluation
- Attach scores to specific observations
- Use standardized metric names
- Include context in score metadata
- Compare across versions/experiments

### Error Handling
- Always wrap Langfuse calls in try-except
- Don't let tracing failures break core functionality
- Log errors but continue execution

## 5. Quick Reference

**Environment Variables:**
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
export LANGFUSE_HOST="cloud.langfuse.com"  # Legacy
```

**Python SDK Basic Usage:**
```python
from langfuse import Langfuse

# Initialize
langfuse = Langfuse()

# Create trace
trace = langfuse.trace(name="user_query")

# Create span
span = trace.span(name="retrieval")

# Add observation
observation = trace.observation(
    name="llm_call",
    input={"prompt": "..."},
    output={"response": "..."}
)

# Add score
trace.score(name="accuracy", value=0.95)

# Flush
langfuse.flush()
```

**Versioning:** Always use the latest stable Langfuse SDK version unless specified otherwise.