# 🛡️ Jev Context Firewall Proxy (Python Edition)

A high-performance asynchronous reverse proxy in Python (FastAPI + HTTPX) that intercepts AI coding agent traffic (Cursor, Trae, Cline, Antigravity), evaluates conversation context using **Codiv OpenJev (64K context)**, guarantees **100% unbreakable agentic flow**, eliminates redundant tool bloat & superseded file reads, and streams pruned payloads (-75% to -90% tokens) to upstream reasoning LLMs (Anthropic Claude, OpenAI, Google Gemini, OpenRouter).

---

## 💡 Yeh Karta Kya Hai? (In Simple Words)

Jab aap Cursor ya Trae jaise AI IDEs use karte hain, toh thodi der mein context size bohot bada (**50k se 150k+ tokens**) ho jata hai. Isme puraani files ke lambe reads, purane command outputs aur logs hote hain jo ab kisi kaam ke nahi hote. Is wajah se:
- LLM slow ho jata hai (high latency).
- API bills bohot zyada badh jaate hain.

👉 **Yeh Proxy aapke IDE aur LLM ke beech mein baithta hai:**
1. Context ko intercept karke **~1000 tokens ke micro-chunks** mein todta hai.
2. **OpenJev AI Brain** se evaluate karwata hai ki kaun sa chunk kaam ka hai aur kaun sa kachra.
3. Faltu chunks ko delete (evict) karta hai aur puraane file reads ko chote stubs se replace karta hai.
4. **75% se 90% tokens bacha kar** Anthropic ya OpenAI ko fast stream kar deta hai.
5. Strict protocol checks ensure karte hain ki **Zero HTTP 400 errors** aayein.

---

## 🏛️ Architecture & Component Flow

```
[ AI IDE (Cursor / Trae) ]
            │  (HTTP POST /v1/messages)
            ▼
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ Jev Context Firewall (Python FastAPI Proxy)             │
│                                                             │
│  1. Provider Router (router.py)                             │
│     Resolves destination (Anthropic, OpenAI, or Gemini)     │
│                                                             │
│  2. Turn Grouping & Claude Token Estimator (parser.py)      │
│     Bundles user prompt + tool calls + results atomically   │
│                                                             │
│  3. Adaptive Chunker (chunker.py)                           │
│     Protects recent tail, micro-chunks history (~1000 tok)  │
│                                                             │
│  4. Secret Sanitizer (extractor.py)                         │
│     Redacts Bearer tokens, API keys, and connection URIs    │
│                                                             │
│  5. OpenJev 6D Cognitive Evaluation (jev_client.py)         │
│     Scores: Relevance (noul), Topic, Dependency, Waste      │
│                                                             │
│  6. Causal Consensus Resolver (causal_graph.py)             │
│     Arbitrates KEEP_FULL vs EVICT decisions                 │
│                                                             │
│  7. Tail & Tool Bloat Optimizer (tail_optimizer.py)         │
│     Replaces superseded file reads (>400 tok) with stubs    │
│                                                             │
│  8. Protocol Stitcher (parser.py)                           │
│     Fixes role alternation & tool couplings (Anti-400)      │
└─────────────────────────────────────────────────────────────┘
            │  (Pruned payload: -75% to -90% tokens)
            ▼
[ Upstream LLM (Anthropic / OpenAI / Gemini / OpenRouter) ]
            │  (Byte-for-byte SSE Stream)
            ▼
[ Developer IDE ] (Real-time response)
```

---

## 🚀 Quickstart

### 1. Requirements & Setup
Make sure you have Python 3.10+ installed:
```bash
git clone https://github.com/gitmanhimanshu/jev-context-firewall.git
cd jev-context-firewall
pip install -r requirements.txt
```

### 2. Configuration (`.env` or `config.json`)
Copy `.env.example` to `.env` and add your OpenJev API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
JEV_API_KEY=your_codiv_api_key_here
PORT=8192
```

### 3. Launch the Proxy
Run using Python:
```bash
python main.py --config config.json
```
Or on Windows, simply double-click `start.bat`.

Output:
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

## ⚙️ IDE Configuration (Cursor, Trae, Cline, etc.)

Point your IDE's Base URL to:
👉 **`http://127.0.0.1:8192`** (or `http://127.0.0.1:8192/v1`)

### A. Trae IDE
- Settings $\rightarrow$ Custom Models $\rightarrow$ Add Model
- Provider: **Anthropic**
- Model Name: `claude-opus-4-8` or `claude-3-5-sonnet-20241022`
- Base URL: `http://127.0.0.1:8192`
- API Key: Your Anthropic API Key

### B. Dynamic Multi-Provider Routing
You can dynamically route to any upstream provider via headers or query parameters:

```bash
# Example 1: Forward to Anthropic via custom header
curl -X POST http://localhost:8192/v1/messages \
  -H "X-Upstream-URL: https://api.anthropic.com" \
  -H "x-api-key: your-anthropic-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'

# Example 2: Forward to OpenRouter via query param
curl -X POST "http://localhost:8192/v1/chat/completions?upstream=https://openrouter.ai/api" \
  -H "Authorization: Bearer your-openrouter-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-r1", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 📊 Real-Time Web Telemetry Dashboard

Open your browser at:
👉 **`http://localhost:8192/dashboard`**

Features:
- **Total Requests**: Intercepted IDE requests counter.
- **Tokens Saved**: Real-time token reduction counter.
- **Estimated Cost Saved**: Dollar (\$ USD) savings calculated automatically.
- **Average Reduction %**: Live reduction efficiency (usually 75%–90%).
- **Recent Turns Table**: Latency, endpoint paths, reduction %, and status.

---

## 🧪 Automated Test Suite

Run the full automated test suite:
```bash
python run_tests.py
```

Tests cover:
- ✅ Consensus Arbitration Logic (6D Cognitive Scoring)
- ✅ Multi-Provider Routing (Anthropic, OpenAI, Gemini)
- ✅ Key Pool Round-Robin Rotation & Concurrency
- ✅ Strict Role Alternation (User $\leftrightarrow$ Assistant)
- ✅ Tool Integrity & Unbreakable Coupling
- ✅ Tail Superseded Read Optimization
- ✅ Secret Scrubbing (Bearer, API keys, DB URIs)
- ✅ End-to-End Payload Pruning for all 3 formats

---

## 📄 License
MIT License.
