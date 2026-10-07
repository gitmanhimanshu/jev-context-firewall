# 🛡️ Jev Context Firewall Proxy (Python Edition)

A high-performance, asynchronous reverse proxy built with **Python (FastAPI + HTTPX)** that intercepts AI coding agent traffic (Cursor, Trae, Cline, Antigravity), evaluates conversation context using **Codiv OpenJev (64K context)**, guarantees **100% unbreakable agentic flow**, eliminates redundant tool bloat & superseded file reads, and streams pruned payloads (**-75% to -90% tokens**) to upstream reasoning LLMs (Anthropic Claude, OpenAI, Google Gemini, OpenRouter).

---

## 🎯 What Problem Does This Solve?

When working with autonomous AI coding agents (such as Cursor, Trae, Cline, or Antigravity), multi-turn conversation payloads quickly swell to **50,000 to 150,000+ tokens**. 

These bloated payloads are filled with:
- Multi-thousand-line file dumps from earlier turns that have since been modified or deleted.
- Verbose terminal and build outputs from finished tasks.
- Dead-end troubleshooting logs and outdated conversational context.

### The Consequences:
1. **Excessive API Bills**: You repeatedly pay high per-token pricing on every turn for stale, irrelevant data.
2. **High Latency & Slow Generation**: Large context windows slow down time-to-first-token (TTFT) and degrade model reasoning speed.
3. **Context Degradation & Hallucinations**: Information overload causes LLMs to lose focus on the active task.

---

## 💡 How It Works

The **Jev Context Firewall** operates as a local or remote transparent proxy between your IDE and upstream LLM providers.

1. **Transparent Interception & Routing**: Intercepts requests sent by your IDE and dynamically resolves target providers (Anthropic, OpenAI, or Google Gemini).
2. **Claude BPE Token Estimation**: Accurately estimates token weight using calibrated BPE ratios ($2.36$ bytes/token).
3. **Atomic Turn Grouping**: Bundles human prompts, assistant tool calls, and tool outputs together into unbreakable atomic units.
4. **Adaptive Micro-Chunking**: Protects the active and immediate conversational tail while slicing older history into atomic $\sim 1,000$-token chunks.
5. **Secret Sanitization**: Automatically scrubs Bearer tokens, API keys, and database connection URIs prior to evaluation.
6. **OpenJev 6D Cognitive Evaluation**: Sends chunk summaries in parallel to **Codiv OpenJev (`openjev-latest`)** via `/v1/systemone` across 6 cognitive dimensions:
   - **Relevance Score (`noul`)**: Continuous relevance score from $0.00$ to $1.00$.
   - **Topic Relationship**: `identical_thread`, `shared_background`, or `completely_disjoint`.
   - **Context Dependency**: `critical_loss`, `minor_context`, or `zero_loss`.
   - **Lifecycle State**: `foundational_rule`, `active_thread`, `completed_subtask`, or `irrelevant_tangent`.
   - **Token Cost Waste**: `heavy_waste`, `moderate_cost`, or `essential_tokens`.
   - **Preservation Target**: `keep_full_detail` or `drop_completely`.
7. **Consensus Arbitration**: Determines whether each chunk should be kept in full (`KEEP_FULL`), compressed (`COMPRESS`), or safely pruned (`EVICT`).
8. **Tail Optimizer with Structural Code Skeletonization**: Detects files that were read in earlier turns and subsequently edited in later turns, extracting high-signal function/class signatures (Python, Go, TypeScript/JS, Rust) using `skeletonizer.py` (-95% token bloat while keeping the model aware of definitions).
9. **Protocol Stitching & Anti-400 Sanitization**: Validates tool call pairings and enforces strict role alternation (`user` $\leftrightarrow$ `assistant`), guaranteeing **Zero HTTP 400 errors**.
10. **Zero-Delay SSE Streaming**: Streams upstream Server-Sent Events (SSE) back to the client byte-for-byte with immediate flushing (`X-Accel-Buffering: no`).

---

## 🌟 What's New in v1.1.0

- 🧩 **Multi-Language Structural Code Skeletonizer (`skeletonizer.py`)**: When an AI agent reads a 3,000-line file that gets modified later, raw text is no longer bluntly dropped. The proxy extracts exact function signatures, parameters, classes, and interfaces (supporting Python, Go, TypeScript/JS, and Rust) with bodies replaced by `...`.
- 🦙 **Offline Ollama & Hybrid Evaluator Engine (`jev_client.py`)**: Support for `JEV_EVALUATOR_MODE="cloud" | "ollama" | "hybrid"`. Evaluate context using local LLMs (e.g., `qwen2.5-coder:1.5b`, `llama3.2:1b`) on your laptop with 0ms internet latency and 100% air-gapped privacy.
- 📥 **Telemetry Export & Audit Reports (`/api/stats/export`)**: Download comprehensive JSON audit logs of tokens saved, cost reductions, and latency metrics right from the web dashboard.

## 🏛️ Architecture & Component Flow

```
[ AI IDE (Cursor / Trae / Cline) ]
                 │  (HTTP POST /v1/messages)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ Jev Context Firewall (Python FastAPI Proxy)             │
│                                                             │
│  1. Provider Router (router.py)                             │
│     Resolves destination (Anthropic, OpenAI, or Gemini)     │
│                                                             │
│  2. Turn Grouping & Token Estimator (parser.py)             │
│     Locks user prompt + tool calls + results into turns     │
│                                                             │
│  3. Adaptive Chunker (chunker.py)                           │
│     Protects active tail, micro-chunks history (~1000 tok)  │
│                                                             │
│  4. Secret Sanitizer (extractor.py)                         │
│     Redacts Bearer tokens, API keys, and connection URIs    │
│                                                             │
│  5. OpenJev 6D Cognitive Evaluation (jev_client.py)         │
│     Evaluates relevance, lifecycle, topic, and waste        │
│                                                             │
│  6. Causal Consensus Resolver (causal_graph.py)             │
│     Arbitrates KEEP_FULL vs EVICT decisions                 │
│                                                             │
│  7. Tail & Tool Bloat Optimizer (tail_optimizer.py)         │
│     Replaces superseded file reads (>400 tok) with stubs    │
│                                                             │
│  8. Protocol Stitcher (parser.py)                           │
│     Enforces role alternation & pairs tool results (Anti-400│
└─────────────────────────────────────────────────────────────┘
                 │  (Pruned payload: -75% to -90% tokens)
                 ▼
[ Upstream LLM (Anthropic / OpenAI / Gemini / OpenRouter) ]
                 │  (Real-Time SSE Stream)
                 ▼
[ Developer IDE ] (Instant response)
```

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
Ensure you have Python 3.10+ installed:
```bash
git clone https://github.com/gitmanhimanshu/jev-context-firewall.git
cd jev-context-firewall
pip install -r requirements.txt
```

### 2. Configure Environment (`.env` or `config.json`)
Copy the environment template:
```bash
cp .env.example .env
```

Open `.env` and insert your OpenJev API key:
```env
# OpenJev / Codiv API Key (Obtain from https://api.codiv.ai)
JEV_API_KEY=your_codiv_api_key_here

# Optional multi-key rotation pool for high concurrency
JEV_API_KEYS=key1,key2,key3
JEV_CONCURRENCY_PER_KEY=3

# Proxy Server Port
PORT=8192
```

### 3. Launch the Proxy
Run with Python:
```bash
python main.py --config config.json
```
Or on Windows, double-click **`start.bat`**.

Console Output:
```text
==================================================================
       [+] JEV CONTEXT FIREWALL PROXY (Python Native)             
==================================================================
Proxy Port         : 8192
Jev Base URL       : https://api.codiv.ai
Jev Model          : openjev-latest
Jev API Key Pool   : 1 key active (Concurrency: 3)
Relevance Gate     : noul >= 0.60
Web Dashboard      : http://127.0.0.1:8192/dashboard
------------------------------------------------------------------
Status: Ready to intercept and prune AI IDE requests transparently.
Press Ctrl+C to stop.
==================================================================
```

---

## ⚙️ IDE Configuration (Trae, Cursor, Cline, Antigravity)

Point your IDE's Base URL to:
👉 **`http://127.0.0.1:8192`** (or `http://127.0.0.1:8192/v1`)

### A. Trae IDE (Anthropic Protocol)
- Navigate to: **Settings** $\rightarrow$ **Custom Models** $\rightarrow$ **Add Model**
- Provider: **Anthropic**
- Model Name: `claude-opus-4-8` or `claude-3-5-sonnet-20241022`
- Base URL: `http://127.0.0.1:8192`
- API Key: Your Anthropic API Key

### B. Cursor IDE / Cline
- Provider: **OpenAI** or **Anthropic**
- Override Base URL: `http://127.0.0.1:8192/v1`
- API Key: Your provider API key

### C. Dynamic Multi-Provider Routing
You can route dynamically through the proxy to any provider via custom headers or query parameters:

```bash
# Example 1: Route to Anthropic using custom upstream header
curl -X POST http://localhost:8192/v1/messages \
  -H "X-Upstream-URL: https://api.anthropic.com" \
  -H "x-api-key: your-anthropic-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'

# Example 2: Route to OpenRouter via query parameter
curl -X POST "http://localhost:8192/v1/chat/completions?upstream=https://openrouter.ai/api" \
  -H "Authorization: Bearer your-openrouter-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-r1", "messages": [{"role": "user", "content": "Hello"}]}'

# Example 3: Route via dedicated path prefix
curl -X POST http://localhost:8192/proxy/openai/v1/chat/completions \
  -H "Authorization: Bearer your-openai-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 📊 Real-Time Web Telemetry Dashboard

Open your browser and navigate to:
👉 **`http://localhost:8192/dashboard`**

### Live Metrics Include:
- **Total Requests**: Intercepted agent turn counter.
- **Tokens Saved**: Cumulative token volume pruned from context payloads.
- **Average Reduction %**: Context reduction percentage (typically **75% to 90%**).
- **Estimated Cost Saved**: Dollar (\$ USD) savings calculated automatically ($3.00/1M tokens).
- **Recent Turns Table**: Real-time table displaying endpoints, original vs. pruned token counts, latencies, and statuses.

---

## 🛡️ Safeguards & Reliability Guarantees

1. **Fail-Open Resilience**: If Codiv OpenJev is temporarily unreachable, times out, or encounters an error, the proxy silently forwards the full original payload without interrupting the developer's IDE workflow.
2. **Unbreakable Tool Protocol**: Ensures every tool result is paired with an active assistant tool call. Orphaned tool outputs are converted into historical user text to prevent Anthropic/OpenAI HTTP 400 errors.
3. **Multi-Key Pool & Semaphore Rate Limiting**: Coordinates concurrent user traffic using async semaphores, preventing upstream rate limits (HTTP 429) with automatic key failover.
4. **Data Privacy**: Redacts sensitive credentials (Bearer tokens, passwords, database URIs) before transmitting chunk summaries for cognitive evaluation.

---

## 🧪 Automated Test Suite

Run the test suite to verify protocol integrity and pruning behavior:
```bash
python run_tests.py
```

### Verified Test Cases:
- ✅ **Consensus Arbitration**: 6D cognitive scoring and decision rules.
- ✅ **Provider Routing**: Dynamic routing across Anthropic, OpenAI, and Gemini.
- ✅ **Key Pool Rotation**: Thread-safe round-robin API key selection & multi-key concurrency.
- ✅ **Role Alternation**: Strict `user` $\leftrightarrow$ `assistant` alternation.
- ✅ **Tool Integrity**: Two-way tool use and tool result pairing.
- ✅ **Tail Optimization**: Replacement of superseded file reads with summary stubs.
- ✅ **Secret Sanitization**: Regex scrubbing of sensitive keys and URIs.
- ✅ **End-to-End Pruning**: Payload validation across Anthropic, OpenAI, and Gemini native formats.

---

## 📄 License
This project is open-source software licensed under the **MIT License**.
