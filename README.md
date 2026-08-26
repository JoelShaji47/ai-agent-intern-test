# Aster & Row Support Agent

A reliability-focused RAG customer support agent built for the Aster & Row take-home assignment. It answers policy and product questions grounded in the supplied knowledge base, looks up order status through a privacy-safe tool, resists prompt injection, and knows when to hand off to a human.

---

## Demo

> [🎬 Watch the demo](https://drive.google.com/file/d/1KRNdDKckbzPJJbIe7JUrR05h6x8PECQS/view?usp=sharing)

---

## Setup & Run (Clean Clone)

**Prerequisites:** Python 3.11+, Node.js 18+, and a Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```powershell
git clone https://github.com/JoelShaji47/ai-agent-intern-test.git
cd ai-agent-intern-test
```

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# open .env and set GEMINI_API_KEY=<your real key>

uvicorn app.main:app --reload
# → http://127.0.0.1:8000
```

### Frontend (separate terminal)

```powershell
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

Start the backend first. The frontend loads and displays a friendly error if the backend isn't reachable yet, but chat requires it.

### Evaluation Suite

```powershell
& .\venv\Scripts\python.exe -m evaluation.run_eval --final          # all 25 cases
& .\venv\Scripts\python.exe -m evaluation.run_eval --only <id>      # single case
```

Runs 15 visible + 10 original cases (25 total) with deterministic substring/citation/tool-call assertions and a narrow LLM-graded fallback only for paraphrase-level checks that can't be matched deterministically. Results saved to `evaluation/results/`.

---

## Environment Variables

| Variable | Where | Required | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | `backend/.env` | Yes | Gemini API access. Startup fails fast with a clear error if missing. |
| `DEBUG_TRACE` | shell env, before starting uvicorn | No | Set to `1` to enable structured JSON trace logging (retrieved chunks, tool calls, handoff mechanism, errors) at DEBUG level. |
| `VITE_API_BASE` | shell env or `.env` in `frontend/` | No | Overrides the backend URL the frontend calls. Defaults to `http://localhost:8000`. |

See [`backend/.env.example`](backend/.env.example) for the template — no real credentials included.

---

## Architecture

The system implements a RAG (retrieval-augmented generation) pipeline where user
queries are matched against the knowledge base via BM25 search, and the
retrieved passages ground the LLM's answer.

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Message                              │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────┴──────────────────────────────────┐
│  Session Manager (In-Memory)                                     │
│  Load prior conversation history if session_id given             │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────┴──────────────────────────────────┐
│  Retrieval (search_knowledge_base)                               │
│  BM25 stemmed search → relevance rank → authority resort         │
│  (active > superseded, without dropping either)                  │
└───────────────────────────────┬──────────────────────────────────┘
                                │ Retrieved passages (untrusted)
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  LLM (Gemini 3.5 Flash Lite, temp 0.2)                           │
│  Prompt enforces: injection resistance, citation accuracy,       │
│  handoff-marker instruction. Reasons over tool results.          │
└────────────┬─────────────────────────────────────┬───────────────┘
             │ Tool call?                          │ No tool call
             ▼                                     │
┌────────────┴────────────────────┐                │
│  Order Lookup Tool              │                │
│  Normalize/validate ID          │                │
│  PII + internal fields          │                │
│  structurally absent            │                │
│  from response model            │                │
│  Not-found → handoff            │                │
└────────────┬────────────────────┘                │
             │ Sanitized result                    │
             ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  LLM synthesizes final answer + citations                        │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────┴──────────────────────────────────┐
│  Handoff derivation (independent of self-report)                 │
│                                                                  │
│  Tier 1 STRUCTURAL: not-found / volunteered credentials /        │
│                     restricted data request                      │
│  Tier 2 MARKER:    model emits exact handoff sentence            │
│  Tier 3 PHRASE:    18-phrase fallback list                       │
│                                                                  │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  Final Response                                                  │
│  (final_answer, sources_cited, tool_calls_made,                  │
│   handoff_recommended)                                           │
└──────────────────────────────────────────────────────────────────┘
```

| | |
|---|---|
| **Model** | Gemini 3.5 Flash Lite (Temperature: 0.2) with native function calling. |
| **Retrieval** | BM25 lexical search across 53 heading-level chunks, avoiding the overhead of embeddings or vector databases.|
| **Order lookup** | Dedicated tool over `orders.json`with strict, schema-enforced privacy guardrails. |
| **Framework** | FastAPI backend, React/Vite frontend (Tailwind, shadcn), and in-memory session state. |

**Retrieval design :** Every chunk is classified `authoritative`, `superseded`, or `non_authoritative` from its front matter. Results are ranked by relevance first, then re-sorted so authoritative content leads — without ever dropping relevant superseded content, so the agent can see it and explicitly dismiss it.

**Privacy design :** Customer PII and internal fields (risk scores, warehouse notes) don't exist on the order-lookup response model at all — structurally impossible to leak, not filtered at runtime.

**Why lexical retrieval :** The data is seperated into 14 documents and 53 chunks. The hard problem is precedence, not semantic search — ranking active policy above superseded content. BM25 plus authority metadata solves that directly.

---

## Evaluation Results

| Category | Baseline (22/25) | Final (25/25) |
|---|---|---|
| **Overall** | **22/25 (88%)** | **25/25 (100%)** |
| retrieval | 4/5 | 5/5 |
| tool-use | 4/4 | 4/4 |
| tool-reliability | 4/4 | 4/4 |
| privacy | 2/3 | 3/3 |
| groundedness | 2/2 | 2/2 |
| conversation | 3/3 | 3/3 |
| multi-source-grounding | 1/1 | 1/1 |
| abstention | 1/1 | 1/1 |
| source-conflict | 1/1 | 1/1 |
| prompt-security | 0/1 | 1/1 |

The baseline reflects the first complete run of the 25-case suite before final prompt and handoff fixes; the final reflects the same suite after those changes. At temperature 0.2, a small number of cases showed occasional non-determinism across repeated runs during development, ranging from pure wording variance on substring-based assertions to genuine intermittent gaps in handoff recommendation that were tracked down and fixed. The final run reported here passed 25/25; isolated `--only` reruns were used throughout to distinguish real regressions from sampling noise before making further changes.

In addition to the formal evaluation suite, 15 manual stress-test probes were executed to evaluate edge cases. These probes tested boundary conditions such as ambiguous identity, multiple order lookups, simulated authority bypasses, out-of-scope catalog queries, and multi-document synthesis. The agent performed correctly across all scenarios, and this extended testing surfaced a substantive issue that was addressed in Bug Diary #4.

---

## Bug Diary

Four reproduced failures, in the order they were found. A recurring theme: the underlying agent behavior was often closer to correct than the *mechanism checking it*, which needed to become more deterministic over time.

### 1. Retrieval missed the right document on paraphrased queries

- **Found:** *"Do you ship to Germany?"* failed to surface `06-international-shipping.md`; a paraphrased damaged-item question missed `04-damaged-or-wrong-items.md#reporting-window`.
- **Root cause:** Plain BM25 has no stemming (`ship` never matches `shipping`), and even after fixing that, low term-overlap can score below the retrieval cutoff.
- **Fix:** Added a lightweight custom stemmer shared between indexing and querying. Widened the candidate pool with a two-stage relevance-then-authority sort, preventing lower-authority-but-relevant content from being crowded out.
- **Regression test:** Covered by `unsupported-country` and `final-sale-damaged-exception` in the eval suite.

### 2. Handoff detection was chasing the model's wording, not its intent

- **Found:** Multiple eval cases intermittently failed their `handoff` assertion despite the agent clearly recommending human help — in phrasing the phrase list hadn't anticipated.
- **Root cause:** A phrase list can never keep pace with paraphrase variety in unconstrained natural language.
- **Fix:** Replaced phrase-scanning with a three-tier system: code-level structural signals → explicit marker phrase the model must emit → phrase-scanning as last-resort fallback.
- **Regression test:** Previously-flaky cases now pass consistently; the marker is exercised on every full suite run.

### 3. The model occasionally hallucinated a citation filename

- **Found:** One answer cited a nonexistent document, blending two real filenames from context.
- **Root cause:** LLM citations are reconstructed from context and can transpose or blend real values.
- **Fix:** Tightened the prompt to require copying filenames character-for-character. Added a frontend guard that only renders a citation as a chip if it matches one of the 14 real KB filenames — hallucinated ones degrade to plain text.
- **Regression test:** Original failing case now passes; guard verified against the exact fabricated string from the second occurrence.
- **Note:** Probabilistic model behavior — prompt instruction reduces but doesn't eliminate it; the guard limits visible symptoms, not the underlying rate.

### 4. `sources_cited` conflated "shown to the model" with "actually used"

- **Found:** A live query about Canada shipping displayed 9–15 source pills, though the answer referenced only 1–2 documents.
- **Root cause:** The field is populated with everything retrieved (for observability) but was being used unfiltered where "actually used" was needed.
- **Fix:** The eval suite and frontend now each parse the model's inline `[filename#heading]` citations from the answer text instead of relying on the raw field. Backend field unchanged.
- **Regression test:** Verified via live UI check and confirmed unaffected in eval suite re-run.
- **Note:** Three independent places deriving "actually used" isn't ideal long-term architecture.

---

## Known Limitations

- **Retrieval and citation aren't bulletproof :** BM25 misses queries with low vocabulary overlap, and the model has occasionally cited nonexistent or transposed filenames despite explicit character-for-character instructions — mitigated by a frontend guard, but not eliminated at the source.
- **In-memory session state only :** Conversation history does not persist across a server restart, per the assignment's explicit scope.
- **Authentication :** The demo assumes possession of an order ID is sufficient. Production would require OAuth/SSO or email verification.
- **Gemini Flash Lite's free-tier quota (500 requests/day) constrains scale testing :** Development and evaluation runs were sometimes throttled mid-run, requiring retry and backoff handling. A production deployment would need a paid tier or a different rate-limiting strategy.

---

## AI Coding Tools Used

| Tool | Usage |
|---|---|
| **opencode** | Agentic coding CLI for implementation, system prompt engineering, evaluation harness, and debugging. |
| **Gemini 3.5 Flash Lite** | Primary runtime LLM for the support agent. |
| **Claude (Anthropic)** | Planning and review partner: design review before execution, output verification at every phase, regression analysis. |

### Example of an AI suggestion that was wrong or incomplete

- **Suggestion:** Filter retrieved chunks by authority tier before passing to the model — drop superseded and non-authoritative content to prevent the agent from citing outdated policy.
- **Problem:** Also silently removed superseded documents that were the only source discussing a specific detail, leaving the agent unable to acknowledge or explicitly dismiss outdated content.
- **Fix:** Two-stage ranking instead — rank by relevance first, then resurface authoritative content preferentially, so the model retains visibility into all content and can reason about it explicitly.

---

## Project Structure

```
ai-agent-intern-test/
├─ backend/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ main.py                  # FastAPI entry point, /chat endpoint, CORS
│  │  ├─ agent.py                 # Gemini tool-calling loop, system prompt, handoff logic
│  │  ├─ ingest.py                # KB ingestion: markdown parsing, heading-level chunking
│  │  ├─ retrieval.py             # BM25 search, two-stage ranking, conflict detection
│  │  ├─ orders.py                # Order lookup tool, PII-safe Pydantic model
│  │  ├─ session.py               # In-memory conversation store for multi-turn
│  │  └─ multiturn_cli.py         # Verification CLI for multi-turn scenarios
│  └─ evaluation/
│     └─ run_eval.py              # 25-case eval suite with deterministic + LLM assertions
├─ frontend/
│  ├─ public/
│  └─ src/
├─ data/
├─ knowledge-base/
├─ evaluation/
│  ├─ visible-cases.json
│  └─ results/
└─ README.md
```

*Dependencies, caches, and generated files (`venv`, `node_modules`, `__pycache__`, `dist`) excluded for clarity.*
