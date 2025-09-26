# https://github.com/emilhagl/Opinion-Units
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseSystem
from llm import run_llm_batch, safe_json_parse

from utils import load_or_build, dumpj

PROMPT_TEMPLATE = """Perform aspect-based sentiment analysis (ABSA) for the restaurant review provided as input.
Extract **opinion units**, where each unit consists of:
  - "aspect": the specific aspect that is being evaluated (use concise aspect terms; may be "overall experience" if no specific aspect fits),
  - "sentiment": one of {"positive","neutral","negative"},
  - "sentiment_score": a score between -1 and 1, where -1 to -0.33 is "negative", -0.33 to 0.33 is "neutral", and 0.33 to 1 is "positive",
  - "excerpt": a short, contextualized evidence snippet copied verbatim (or near-verbatim) from the review that justifies the sentiment.
Notes:
  - Excerpts should be concise but sufficient to justify the sentiment; they may span multiple sentences if needed.
  - Ignore non-opinionated text (factual or irrelevant content).
  - If no opinion units are present, return an empty list.
  - Return STRICT JSON ONLY: a JSON list of objects with keys exactly "aspect","sentiment","excerpt" (no extra text).

Example output (illustrative formatting of multiple units):
[
{"aspect": "outdoor patio seating", "sentiment": "positive", "sentiment_score": 0.78, "excerpt": "The gorgeous outdoor patio seating was fantastic"},
{"aspect": "view", "sentiment": "positive", "sentiment_score": 0.72, "excerpt": "What a fantastic view of the ocean"},
{"aspect": "drinks", "sentiment": "positive", "sentiment_score": 0.63, "excerpt": "We came for brunch and the Bloody Marys were superb"},
{"aspect": "overall experience", "sentiment": "positive", "sentiment_score": 0.41, "excerpt": "Altogether, we had a great experience"},
{"aspect": "staff friendliness", "sentiment": "negative", "sentiment_score": -0.44, "excerpt": "The staff could have been a little friendlier"},
{"aspect": "table cleanliness", "sentiment": "negative", "sentiment_score": -0.52, "excerpt": "The tables could have been cleaner"}
]

Input (review text):
{TEXT}
"""

OPINION_UNIT_JSON_SCHEMA = {
    "name": "opinion_units",
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "aspect": {"type": "string"},
                "sentiment": {"type": "string"},
                "sentiment_score": {"type": "number"},
                "excerpt": {"type": "string"},
            },
            "required": ["aspect", "sentiment", "excerpt", "sentiment_score"],
            "additionalProperties": False,
        },
    },
}


class OUBaseline(BaseSystem):
    """
    Opinion-Units baseline (retrofit):
    """
    def __init__(self, args, reviews, tests):
        super().__init__(args, reviews, tests)
        '''
        in basesystem
        - self.reviews
        - self.rich_reviews
        '''
        self.ou_model = "gpt-5-nano"
        self.ou_temperature = 0
        self.num_workers = 128
        self.prompt_template = PROMPT_TEMPLATE
        self.ou_json_schema = OPINION_UNIT_JSON_SCHEMA
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def _make_prompt(self, doc_text):
        return self.prompt_template.replace("{TEXT}", doc_text)

    def _normalize_unit(self, u):
        aspect = (u.get("aspect") or "").strip()
        sentiment = (u.get("sentiment") or "").strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        excerpt = (u.get("excerpt") or "").strip()
        sentiment_score = float(u.get("sentiment_score") or 0)
        rules = {
            "neutral":  ((-0.33, 0.33), 0.0),
            "negative": ((-1.0, -0.33), -0.66),
            "positive": ((0.33, 1.0),   0.66),
        }
        (low, high), default = rules[sentiment]
        if not (low <= sentiment_score <= high):
            sentiment_score = default
        return {"aspect": aspect, "sentiment": sentiment, "excerpt": excerpt, "sentiment_score": sentiment_score}

    def segmentation(self, reviews):
        review_texts = [r["text"] for r in reviews]
        prompts = [self._make_prompt(t) for t in review_texts]

        raw_outputs = run_llm_batch(
            prompts,
            model=self.ou_model,
            temperature=self.ou_temperature,
            num_workers=self.num_workers,
            verbose=True,
            json_schema=self.ou_json_schema,
            use_json=True,
        )
        all_units = []
        for r, out_str in zip(reviews, raw_outputs):
            data = safe_json_parse(out_str)  # expects a JSON list; tolerant to dicts
            units_raw = data if isinstance(data, list) else (data.get("units", []) if isinstance(data, dict) else [])
            units = [self._normalize_unit(u) for u in units_raw if isinstance(u, dict)]
            # de-duplicate (aspect, sentiment, excerpt)
            units = list({(u["aspect"], u["sentiment"], u["excerpt"]): u for u in units}.values())
            # save back into original place
            r["opinion_units"] = units
            all_units.append(units)

        return all_units

    # ---- helper for embedding store (faithful spirit) ----
    @staticmethod
    def create_query_string(u, include_sentiment=False):
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

    def embed_texts(self, texts, batch_size=256, normalize=True):
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
        return embs.cpu().numpy().astype("float32")

    # ----- Final retrofit for aspect-based sentiment prediction -----
    def _pick_top_even(self, reviews, aspects, total=100):
        R = self.embed_texts([r["text"] for r in reviews])   # [N,D]
        T = self.embed_texts(aspects)                        # [Q,D]
        S = T @ R.T                                          # [Q,N]  
        Q, N = S.shape[0], S.shape[1]
        total = min(total, N)
        quota = max(1, total // Q)

        # per-aspect picks (even split), then de-dup
        per = np.concatenate([np.argsort(S[q])[::-1][:quota] for q in range(Q)])  # top quota per row
        sel = np.unique(per)

        # one-line global top-up to reach exactly `total`
        g = S.max(axis=0); order = np.argsort(g)[::-1]
        sel = np.concatenate([sel, order[~np.isin(order, sel)][:max(0, total - len(sel))]])

        sel = sorted(sel, key=lambda i: i)[:total]
        return sel

    def predict_given_aspects(self, user_id, item_id, aspects):
        all_reviews = [r for r in self.rich_reviews if r.get("item_id") == item_id]
        sel_idx = self._pick_top_even(all_reviews, aspects, total=100)
        item_reviews = [all_reviews[i] for i in sel_idx]

        missing = [r for r in item_reviews if "opinion_units" not in r]

        if missing:
            built_units = self.segmentation(missing)  
            dumpj(self.args.rich_reviews_path, self.rich_reviews) 

        pool = [u for r in item_reviews for u in r["opinion_units"]]
        aspect_texts = [u.get("aspect") for u in pool]
        excerpt_texts = [u.get("excerpt") for u in pool]

        # encode once
        U_aspect = self.embed_texts(aspect_texts)
        U_excerpt = self.embed_texts(excerpt_texts)
        T_aspect = self.embed_texts(aspects)                    # [Q, D]
        

        alpha = 0.8
        S_a = T_aspect @ U_aspect.T
        S_e = T_aspect @ U_excerpt.T
        S = alpha * S_a + (1.0 - alpha) * S_e

        # threshold on how similar; must be at least min_sim
        min_sim = 0.5
        mask = S >= min_sim
        S = np.where(mask, S, 0.0)

        # threshold on how many, get top k
        k = 16 
        k = min(k, S.shape[1])
        idx = np.argpartition(S, -k, axis=1)[:, -k:]
        m = np.zeros_like(S, dtype=bool); m[np.arange(S.shape[0])[:,None], idx] = True
        S = np.where(m, S, 0.0)

        ymap = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        y_vec = np.array([ymap.get((u.get("sentiment") or "neutral").lower(), 0.0) for u in pool], dtype=np.float32)

        num = S @ y_vec                 # [Q]
        den = S.sum(axis=1) + 1e-9      # [Q]
        scores = num / den                     # [Q], in [-1,1]

        return scores.tolist()