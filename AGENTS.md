# AGENTS.md

> A living guide to the agents that power this project: what they do, how they talk to each other, and how to evaluate them—kept **small, explicit, and reproducible**.

---

## 0) Why this document exists

- To make it **trivial** to add, replace, or compare agents (baselines or LLM-powered).
- To keep **evaluation contracts** stable while the system evolves.
- To document **data shapes**, **orchestration patterns**, **scheduling**, and **observability** in one place.

This file avoids repo-specific code. It specifies **interfaces, contracts, and workflows** so any implementation that honors these will slot in cleanly.

---

## 1) Core ideas

1. **One simple contract:** agents read inputs, produce results in a standard shape, and never define evaluation. The **evaluator** reads agent results and computes metrics.
2. **Reproducible runs:** every agent writes a run folder with: inputs snapshot → outcomes → eval → summary.
3. **Small steps:** start with sparse/dense baselines; layer on hybrids/LLM rerankers only after the metric loop is green.

---

## 2) Directory & naming conventions

> These are conventions, not requirements, but sticking to them keeps diffs readable.

- `agents/` – agent implementations (e.g., `bm25/`, `dense/`, `hybrid/`, `llm_rerank/`).
- `data/` – prepared inputs (reviews, metadata), golden tests.
- `runs/<agent_name>/<run_id>/`
  - `requests.jsonl` – the evaluation inputs used for this run
  - `outcomes.jsonl` – the agent’s ranked outputs
  - `eval_per_request.jsonl` – per-request judgments & metrics
  - `summary.json` – aggregated metrics and config snapshot
  - `NOTES.md` – (optional) analyst notes

`<run_id>` is a timestamp or UUID. Keep it machine-generated.

---

## 3) Data contracts (shared across agents)

### 3.1 Reviews (input corpus)
Minimal fields:
```json
{"review_id": "r123", "item_id": "i42", "text": "…"}
