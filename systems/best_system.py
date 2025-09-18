from wtpsplit import SaT
from debug import check
class BaseSystem:
    def __init__(self, args, DATA, city):
        self.args = args
        self.reviews = DATA['REVIEWS'][city]

class BestSystem(BaseSystem):
    def __init__(self, args, DATA, city):
        super().__init__(args, DATA, city)

class SATBaseline(BaseSystem):
    def __init__(self, args, DATA, city):
        print('operating segment any text baseline segmentation')
        super().__init__(args, DATA, city)
        self.onnx = True

        if self.onnx:
            # ONNX Runtime on GPU (falls back to CPU if CUDA not available)
            self.sat = SaT("sat-3l", ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        else:
            self.sat = SaT("sat-3l")
            self.sat.half().to("cuda")  # comment out if no GPU

        self.segmentation()

    def segmentation(self):
        review_texts = [r['text'] for r in self.reviews]
        # sat.split accepts a list and yields per-text segments (iterator)
        segmented_iter = self.sat.split(review_texts[:100])
        segmented_reviews = list(segmented_iter)
        # (optional) attach back to reviews
        # for r, segs in zip(self.reviews, segmented_reviews):
        #     r["segments"] = segs
        print('checking...')
        check()
        return segmented_reviews

# https://github.com/segment-any-text/wtpsplit?tab=readme-ov-file
# https://github.com/PKU-TANGENT/NeuralEDUSeg?tab=readme-ov-file

class OUBaseline(BaseSystem):
    """
    Opinion-Units baseline (faithful):
    - Few-shot style instructions that mirror Figure 4 (Appendix A).
    - Extract a JSON list of {aspect, sentiment, excerpt}.
    - Attach to each review as r["opinion_units"].
    - Includes create_query_string for building strings to embed.
    """
    def __init__(self, args, DATA, city):
        print('operating opinion unit baseline segmentation')
        super().__init__(args, DATA, city)
        self.ou_model = "gpt-5-nano"
        self.ou_temperature = 0
        self.num_workers = 8
        self.max_review_num = 100
        self.segmentation()

    # ---- inline prompt (faithful to Figure 4) ----
    PROMPT_TEMPLATE = """Perform aspect-based sentiment analysis (ABSA) for the restaurant review provided as input.
Extract **opinion units**, where each unit consists of:
  - "aspect": the specific aspect that is being evaluated (use concise aspect terms; may be "overall experience" if no specific aspect fits),
  - "sentiment": one of {"positive","neutral","negative"},
  - "excerpt": a short, contextualized evidence snippet copied verbatim (or near-verbatim) from the review that justifies the sentiment.
Notes:
  - Excerpts should be concise but sufficient to justify the sentiment; they may span multiple sentences if needed.
  - Ignore non-opinionated text (factual or irrelevant content).
  - If no opinion units are present, return an empty list.
  - Return STRICT JSON ONLY: a JSON list of objects with keys exactly "aspect","sentiment","excerpt" (no extra text).

Example output (illustrative formatting of multiple units):
[
  {"aspect": "outdoor patio seating", "sentiment": "positive", "excerpt": "The gorgeous outdoor patio seating was fantastic"},
  {"aspect": "view", "sentiment": "positive", "excerpt": "What a fantastic view of the ocean"},
  {"aspect": "drinks", "sentiment": "positive", "excerpt": "We came for brunch and the Bloody Marys were superb"},
  {"aspect": "overall experience", "sentiment": "positive", "excerpt": "Altogether, we had a great experience"},
  {"aspect": "staff friendliness", "sentiment": "negative", "excerpt": "The staff could have been a little friendlier"},
  {"aspect": "table cleanliness", "sentiment": "negative", "excerpt": "The tables could have been cleaner"}
]

Input (review text):
{TEXT}
"""

    def _make_prompt(self, doc_text: str) -> str:
        return self.PROMPT_TEMPLATE.replace("{TEXT}", doc_text)

    def _normalize_unit(self, u: dict) -> dict:
        aspect = (u.get("aspect") or "").strip()
        sentiment = (u.get("sentiment") or "").strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        excerpt = (u.get("excerpt") or "").strip()
        return {"aspect": aspect, "sentiment": sentiment, "excerpt": excerpt}

    def segmentation(self):
        from llm import run_llm_batch, safe_json_parse

        reviews = self.reviews if self.max_review_num is None else self.reviews[: self.max_review_num]
        review_texts = [r["text"] for r in reviews]

        prompts = [self._make_prompt(t) for t in review_texts]

        raw_outputs = run_llm_batch(
            prompts,
            model=self.ou_model,
            temperature=self.ou_temperature,
            num_workers=self.num_workers,
            verbose=True
        )

        all_units = []
        for r, out_str in zip(reviews, raw_outputs):
            data = safe_json_parse(out_str)  # expects a JSON list; tolerant to dicts
            units_raw = data if isinstance(data, list) else (data.get("units", []) if isinstance(data, dict) else [])
            units = [self._normalize_unit(u) for u in units_raw if isinstance(u, dict)]
            # de-duplicate (aspect, sentiment, excerpt)
            units = list({(u["aspect"], u["sentiment"], u["excerpt"]): u for u in units}.values())
            r["opinion_units"] = units
            all_units.append(units)

        check()
        return all_units

    # ---- helper for embedding store (faithful spirit) ----
    @staticmethod
    def create_query_string(u: dict, include_sentiment: bool = False) -> str:
        """
        Build a string to embed per opinion unit.
        The repo’s flow embeds opinion-unit content for retrieval; two common variants:
          - excerpt-only (default),
          - aspect + sentiment + excerpt (when doing sentiment-filtered retrieval).
        """
        if not include_sentiment:
            # default: focus on the evidence text itself
            return u.get("excerpt", "").strip()
        # sentiment-aware: useful for "opinion units + sentiment filter" runs
        a = u.get("aspect", "").strip()
        s = u.get("sentiment", "").strip()
        e = u.get("excerpt", "").strip()
        return f"aspect: {a} | sentiment: {s} | {e}"


# https://github.com/emilhagl/Opinion-Units