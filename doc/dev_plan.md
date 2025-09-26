# Development Plan for Arbitrary-Aspect RAG for E-Commerce

## 1. Data Ingestion and Graph Backbone
- **Graph schema implementation**: create entities for Segment, Review, User, Item, Category, and Specs; encode edges linking them to preserve lineage and metadata required for applicability/reliability scoring. Prioritize scalable storage (e.g., relational tables with foreign keys or property graph abstraction) and ensure audit-friendly provenance fields.【F:doc/goal.md†L20-L41】
- **Segmentation pipeline input**: design ingestion jobs to parse raw reviews, attach timestamps, price snapshots, and reviewer statistics so downstream models have the necessary signals. Use segment any text model.

## 2. Review Segmentation and Aspect Labeling
- **Span extraction service**: build sentence/EDU segmentation plus opinion cue detection using lightweight classifiers; keep hooks for LLM auditing on sampled outputs to mitigate drift.【F:doc/goal.md†L43-L73】
- **Aspect and polarity tagging**: implement multi-task encoder that predicts aspect tags, polarity intensity, and claim/measurement roles. Support taxonomy seeding and open-set validation workflows.【F:doc/goal.md†L49-L73】
- **Quality filtering**: integrate spam, duplication, toxicity, and language filters to maintain reliable evidence segments.【F:doc/goal.md†L56-L73】【F:doc/goal.md†L93-L123】

## 3. Embedding Geometry and Retriever Training
- **Hybrid embedding model**: prototype encoder that outputs concatenated Euclidean and hyperbolic (or order-structured) representations, with query-dependent gating for asymmetric retrieval.【F:doc/goal.md†L75-L112】
- **Loss functions**: implement combined listwise ranking loss with parent-child order penalties; include margin-based constraints for hierarchy awareness.【F:doc/goal.md†L84-L109】
- **Retriever distillation**: collect LLM-ranked segment lists (“lightning” supervision), compute soft targets, and minimize KL divergence during retriever fine-tuning; support iterative hard-negative mining.【F:doc/goal.md†L113-L146】

## 4. Applicability and Reliability Modeling
- **Feature assembly**: construct per-segment feature vectors combining embeddings with reviewer/item/user metadata; ensure data pipeline supplies historical behavior signals where available.【F:doc/goal.md†L147-L185】
- **Scoring heads**: develop calibrated MLP heads for reliability and applicability predictions, with monitoring dashboards for drift and calibration error.【F:doc/goal.md†L147-L185】

## 5. Retrieval, Reranking, and Aggregation Workflow
- **Segment scoring**: integrate similarity, applicability, reliability, and diversity terms into the scoring API; expose tunable weights α, β, γ, δ for experimentation.【F:doc/goal.md†L187-L220】
- **Cross-encoder reranker**: deploy compact reranker over top-K segments to boost precision while retaining segment→review→item lineage.【F:doc/goal.md†L200-L214】
- **Item aggregation**: implement aspect coverage tracking, submodular utility optimization, and risk-adjusted scoring to surface balanced recommendations.【F:doc/goal.md†L204-L226】
- **LLM reasoning harness**: enforce citation-only generation over retrieved spans, with trade-off articulation templates and guardrails for hallucination avoidance.【F:doc/goal.md†L214-L226】

## 6. Evaluation and Monitoring
- **Retrieval metrics**: set up pipelines for nDCG/Recall, aspect coverage, asymmetry checks, and faithfulness auditing of cited spans.【F:doc/goal.md†L228-L253】
- **Ranking metrics**: benchmark item-level nDCG/ERR, trade-off explanation quality, and calibration performance using human/LLM judges plus behavioral proxies.【F:doc/goal.md†L228-L262】
- **Efficiency tracking**: log latency and cost savings from distillation versus LLM-only baselines; monitor P50/P95 latency targets.【F:doc/goal.md†L264-L266】

## 7. Iterative Experimentation Roadmap
- **Phase 1**: baseline segmentation and Euclidean retriever with manual evaluation to validate data pipeline.
- **Phase 2**: introduce hyperbolic/order components and lightning distillation; run ablations for hierarchical queries.【F:doc/goal.md†L75-L146】
- **Phase 3**: add applicability/reliability scoring, submodular item aggregation, and LLM citation enforcement; measure faithfulness and trade-off articulation.【F:doc/goal.md†L147-L226】【F:doc/goal.md†L228-L262】
- **Phase 4**: stress-test robustness against spam, duplicates, and distribution shifts; refine calibration under conflicting evidence.【F:doc/goal.md†L19-L20】【F:doc/goal.md†L93-L123】【F:doc/goal.md†L228-L262】
