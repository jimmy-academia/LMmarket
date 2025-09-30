# AGENTS.md — BM25 Baseline Agent

**Location:** `systems/bm25.py`  
**Role:** Sparse-retrieval baseline (reviews → BM25 → aggregate to items)  
**Status:** v0.1 (MVP; evaluation-ready)

---

## 1) Purpose

Establish a strong, simple baseline for **abstract quality queries** (e.g., “quiet, cozy cafe with natural light”). The agent:

- Treats each **review as a document**.
- Retrieves top-R reviews using **BM25**.
- **Aggregates to items** by summing the **top-M review scores** per item.
- Returns top-K items with **evidence review IDs** for inspection and future re-ranking.

This agent is intentionally minimal to enable rapid iteration and clear comparisons.

---

## 2) Scope & Non-Goals

**In scope**
- Review-level sparse retrieval (BM25).
- Item-level ranking via aggregation of review scores.
- Evidence packaging (review IDs/snippets) for each item.
- Works within the BaseSystem evaluation loop.

**Out of scope (for this agent)**
- Chunk/segment-level indexing (paragraph/sentence).
- Embeddings/ANN.
- LLM re-ranking.
- Filters (distance, price, hours, etc.).
- Complex ontologies/aspect taxonomies.

---

## 3) Data Contracts

### 3.1 Inputs

- `DATA["reviews"]`: list of dicts with **minimum** fields  
  - `review_id` (string)  
  - `item_id` (string)  
  - `text` (string; full review)

- `tests`: list of dicts (golden evaluation set)  
  - `request_id` (string)  
  - `query` (string; user request text)  
  - `gold` (list of item_ids for binary relevance **or** dict `{item_id: gain}` for graded relevance)  
  - `topk` (int; desired result cutoff for this request)

> Optional fields in `DATA["reviews"]` (e.g., rating, timestamp) are ignored by this baseline.

### 3.2 Outputs (runtime state, consumed by the evaluator)

- `self.result[request_id]` contains:
  - `request`: the normalized request dict used.
  - `candidates`: ordered list of items, each with:
    - `item_id` (string)
    - `model_score` (float; aggregated BM25 score for the item)
    - `evidence` (list of review_ids used to compute the item score)
    - (optional) `reason` (string; short human-readable hint if available)

- The BaseSystem writes artifacts (naming may vary per framework):
  - `requests.jsonl` — normalized test set actually used
  - `outcomes.jsonl` — per-request ranked candidates with evidence
  - `eval_per_request.jsonl` — per-request metrics and judgments
  - `summary.json` — aggregate metrics (e.g., mean nDCG@K, coverage)

---

## 4) Algorithm Overview

1. **Index Unit:** one **review** per document (no segmentation).
2. **Retrieval:** run **BM25** on the query string to get **top-R reviews** (default `R=500`).
3. **Aggregation:** for each item, sort its matched reviews by BM25 score, take **top-M** (default `M=3`), and **sum** those scores as the **item score**.
4. **Ranking:** sort items by aggregated score; return **top-K** (default `K=3` for evaluation headline).
5. **Evidence:** attach the review IDs (and optionally short snippets) of the **top-M contributing reviews**.

**Key decisions:**
- **Sum(top-M)** is robust to single outlier reviews and avoids long-tail noise.
- No segmentation initially (keeps context; reduces engineering choices).
- Minimal tokenization: lowercased alphanumerics + small stoplist.

---

## 5) Configuration (expected args / defaults)

- `retrieve_k` (int, default **500**)  
  Number of reviews retrieved by BM25 before item aggregation.

- `bm25_top_m` (int, default **3**)  
  Number of top reviews per item to aggregate (sum of their BM25 scores).

- `topk` (int, default **3** or **10** depending on your evaluation/reporting needs)  
  Number of items returned per request.

- `bm25_k1` (float, default **1.5**)  
- `bm25_b` (float, default **0.75**)  
  Standard BM25 parameters.

- `stopwords` (set or None)  
  Optional custom stopword set; falls back to a small default.

- `k_eval` (int, default **3**)  
  Evaluator’s metric cutoff (e.g., Precision@3, nDCG@3).

- `seed` (int, default **123**)  
  For deterministic tie-breaking or any randomized behavior.

---

## 6) Interaction with the Evaluation Loop (BaseSystem)

- The evaluator (BaseSystem) is responsible for:
  - Iterating through the test queries.
  - Calling this agent’s **work(request)** once per request.
  - Storing the returned candidates in `self.result`.
  - Computing metrics against `gold` (e.g., **Precision@K**, **nDCG@K**, **coverage**).
  - Writing run artifacts.

- The agent’s responsibility is limited to **producing the ranked candidates** for a request, with **item_id**, **model_score**, and **evidence**.  
  The agent **does not** compute evaluation metrics.

---

## 7) Evaluation Protocol (for this agent)

- **Task focus:** abstract quality queries (e.g., ambience/vibe/style) where keywords alone are challenged but still provide a useful baseline.
- **Gold format:** start with **binary** lists of correct item_ids; optionally support graded `{item_id: gain}` later.
- **Metrics:** report **nDCG@3** (headline) and **Precision@3** (sanity).  
- **Coverage:** fraction of requests for which the agent returns ≥1 candidate.
- **Process:**
  1. Curate a small **golden set** (20–30 queries).
  2. Run the agent → `serve()` → `evaluate()`.
  3. Inspect **evidence reviews** for top-ranked items to verify that matched text reflects the query intent.
  4. Tune `retrieve_k` and `bm25_top_m` once if needed; avoid overfitting.

---

## 8) Operational Notes

- **Tokenization:** lowercase, alphanumeric word boundaries, tiny stoplist. Keep it simple.
- **Tie-breaking:** on equal item scores, prefer the item with **more supporting reviews**; then a stable identifier order.
- **Performance:** precompute IDF once at init; in-memory DOC store is fine for initial scale. Persist only if needed.
- **Determinism:** set a seed; keep sorting/tie-breaks stable for reproducibility.
- **Artifacts:** ensure outcomes and eval files are written for every run for easy diffing across parameter changes.

---

## 9) Extension Points (future work)

- **Query expansion:** synonyms and common phrases (“natural light”, “sunlit”, “daylight”).
- **Segmentation:** compare sentence/paragraph chunks vs whole reviews.
- **Evidence diversification:** reduce redundancy among supporting reviews.
- **Re-ranking:** add an LLM re-ranker using the evidence bundle for top-50 items.
- **Filters/constraints:** hours, distance, price, dietary; introduced after the baseline is stable.
- **Temporal/recency:** lightweight decay for very old reviews if helpful.

---

## 10) Troubleshooting

- **Empty/weak results:** check tokenizer/stoplist; ensure reviews aren’t reduced to empty tokens; raise `retrieve_k`.
- **Spammy matches:** strengthen stoplist, reduce `retrieve_k`, consider phrase matching for frequent bigrams.
- **Over-reliance on one review:** increase `bm25_top_m` (e.g., 5) or cap per-review contribution.
- **Slow indexing:** cache tokenization/IDF if the corpus is large; still defer until scale demands it.
- **Ambiguous gold:** document uncertain cases; consider graded relevance for ambiguous/subjective queries.

---

## 11) Glossary

- **R**: number of review documents retrieved by BM25 before aggregation (`retrieve_k`).  
- **M**: number of top contributing reviews per item used for aggregation (`bm25_top_m`).  
- **K**: number of items returned/evaluated (`topk`, `k_eval`).  

---

## 12) Definition of Done (DoD)

- Returns top-K items (with `item_id`, `model_score`) for any free-text query.
- Attaches **evidence review IDs** that justify each item’s score.
- Integrates with BaseSystem evaluation; produces **nDCG@3**, **P@3**, and **coverage** on the golden set.
- Deterministic runs with the same inputs and parameters.
- Run artifacts are persisted for comparison across runs.

---

## 13) Change Log

- **v0.1 (MVP):** Review-level BM25, sum(top-M) to items, evidence packaging, BaseSystem integration, evaluation protocol documented.

---
