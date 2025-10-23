import json
import logging
from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights

from networks.sentiment import LLM_ASPECT_LABEL_PROMPT, LLM_ASPECT_LABEL_SCHEMA

class MainMethod(BaseSystem):
    def __init__(self, args, data):
            super().__init__(args, data)

    def recommend(self, query):

        gameplan = """
        [o] 1. aspect split
        [.] 2. retrieve by aspect
                - (over load => filter => score)
                - reasonable: no significant misses
                //////////// IDEAS ///////////////////
                /// 1a LLM produce, reuse aspect pool => definition, lexical hints //exclusion items?
                /// 1b aspect query vector
                /// 1c iterative threshold adjustment with LLM, save to aspect pool
                //////////////////////////////////////
        [ ] 3. score aspects
                //////////// IDEAS ///////////////////////////
                /// find the axis for positive to negative?///
                //////////////////////////////////////////////
                /// cluster and sample //////
                /// LM generate and match ///
                /////////////////////////////
        [ ] 4. combine into hesitant fuzzy set
                stable and reproduceable process?
        [ ] 5. rank by hasitant set operation
        """
        print(gameplan)
        
        raw = load_or_build(self.args.cache_dir/'temp_aspect_dict.json', dumpj, loadj, infer_aspects_weights, query)
        self.aspect_dict = raw if isinstance(raw, dict) else json.loads(raw)
        sum_weights = sum([float(a["weight"]) for a in self.aspect_dict["aspects"]])
        for a in self.aspect_dict["aspects"]:
            a["weight"] = float(a.get("weight", 0)) / sum_weights
        logging.info("[aspects] %s", json.dumps(self.aspect_dict, indent=2))
        
        # print(self.embedding.shape) (1492456, 384)

        for aspect in self.aspect_dict['aspects']:
            self.work_one_aspect(aspect)        

    def work_one_aspect(self, aspect):
        result_dict = self.collect_relevant_segments_for_aspect(aspect)
        result_dict = self.iterative_pseudo_label_loop(aspect, result_dict['kept_ids'])


        # --- Stage 1: progressive collection by aspect ---
    def collect_relevant_segments_for_aspect(
        self,
        aspect_sentence: str,
        D: int = 100,
        probe_k: int = 20,
        max_windows: int | None = None,
        nprobe: int | None = None,
    ):
        """
        For an aspect, search once to get a long candidate list (idxs, scores),
        then iteratively check with LLM the tail k of every D-sized block:
          check [80..100], [180..200], [280..300], ...
        Stop when two consecutive checks return 0 relevant.
        Returns: dict with 'kept_ids', 'checked_ranges', 'stop_at'
        """
        q = self._encode_query(aspect_sentence)
        # First-pass retrieval large enough to allow many windows
        BIG_K = 5000
        if hasattr(self, "faiss_index") and self.faiss_index is not None:
            # Prefer your FAISS path if available
            import numpy as np
            D1, I1 = self.faiss_index.search(np.asarray([q], dtype="float32"), BIG_K)
            scores = D1[0]; idxs = I1[0]
        else:
            # Fallback: brute-force over self.embedding
            import numpy as np
            sims = self.embedding @ q
            idxs = np.argsort(-sims)[:BIG_K]
            scores = sims[idxs]

        kept_ids = []
        checked_ranges = []
        empty_streak = 0
        total = len(idxs)
        windows = total // D
        if max_windows is not None:
            windows = min(windows, max_windows)

        for w in range(windows):
            lo = w * D
            hi = min((w + 1) * D, total)
            tail_lo = max(lo, hi - probe_k)
            tail_hi = hi

            tail_ids = idxs[tail_lo:tail_hi]
            tail_scores = scores[tail_lo:tail_hi]

            # LLM check for aspect relevance (stub—replace with your judge)
            rel_mask = self._llm_label(tail_ids, aspect_sentence)

            n_rel = int(sum(rel_mask))
            checked_ranges.append({"range": [int(tail_lo), int(tail_hi)], "n_rel": n_rel})

            if n_rel == 0:
                empty_streak += 1
            else:
                empty_streak = 0
                # keep only relevant tail ids
                kept_ids.extend([int(i) for i, keep in zip(tail_ids, rel_mask) if keep])

            if empty_streak >= 2:
                # stop BEFORE the two empty checks => last kept upper-bound = start of first empty tail
                stop_at = int(max(0, tail_lo - D))  # previous block start
                return {
                    "kept_ids": kept_ids,
                    "checked_ranges": checked_ranges,
                    "stop_at": stop_at,
                }

        # Exhausted without two consecutive empties
        return {
            "kept_ids": kept_ids,
            "checked_ranges": checked_ranges,
            "stop_at": int(min(total, windows * D)),
        }

    # --- Stage 2: iterative pseudo-label training ---
    def iterative_pseudo_label_loop(
        self,
        aspect_sentence: str,
        candidate_ids: list[int],
        sample_k: int = 64,
        max_rounds: int = 10,
        agree_frac_threshold: float = 0.95,
    ):
        """
        Loop:
          1) sample k segments (uncertain-first, then random fallback)
          2) LLM labels -> (relevant, sentiment, confidence in [0,1])
          3) train simple logistic head on frozen embeddings
          4) pseudo-label entire candidate set
          5) stop when sampled LLM labels == pseudo labels (>= agree_frac_threshold)
        Returns: dict with 'labels', 'sentiment', 'probs', 'rounds', 'agree_frac'
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        X = self.embedding[candidate_ids]  # frozen features
        y_rel = np.full(len(candidate_ids), -1, dtype=int)  # -1 unknown, 0/1 known
        y_sent = np.full(len(candidate_ids), 0, dtype=int)  # -1 neg, 0 neutral, +1 pos (we'll map)

        # Start with a weak prior using similarity to aspect as a proxy confidence
        q = self._encode_query(aspect_sentence)
        proxy = (X @ q)
        proxy = (proxy - proxy.min()) / max(1e-8, (proxy.max() - proxy.min()))
        p_rel = proxy.copy()  # model probability placeholder

        rounds = 0
        for r in range(max_rounds):
            rounds += 1

            # 1) pick k to label: prefer uncertain (p~0.5); else random among unlabeled
            unl = np.where(y_rel < 0)[0]
            if unl.size == 0:
                break
            uncertainty = np.abs(p_rel[unl] - 0.5)
            order = np.argsort(uncertainty)  # most uncertain first
            pick = unl[order[:min(sample_k, unl.size)]]

            # 2) query LLM for those
            ids_to_label = [int(candidate_ids[i]) for i in pick]
            llm_rel, llm_sent, llm_conf = self._llm_label(ids_to_label, aspect_sentence)

            # map / store
            y_rel[pick] = np.array(llm_rel, dtype=int)
            y_sent[pick] = np.array(llm_sent, dtype=int)
            conf = np.clip(np.array(llm_conf, dtype=float), 0.0, 1.0)

            # 3) train simple head (relevance). Weighted by LLM confidence for robustness
            labeled = np.where(y_rel >= 0)[0]
            if labeled.size >= 4:  # tiny sanity
                clf = LogisticRegression(max_iter=200, class_weight="balanced")
                w = np.ones(labeled.size, dtype=float)
                # option: stronger weight for higher confidence labels
                w *= 0.5 + 0.5 * np.interp(labeled, pick, conf, left=0.5, right=1.0)
                clf.fit(X[labeled], y_rel[labeled], sample_weight=w)
                # 4) pseudo-label others
                p_rel = clf.predict_proba(X)[:, 1]
            else:
                # not enough supervision yet; keep proxy
                p_rel = proxy.copy()

            # agreement check on the *sampled* batch
            pseudo_batch = (p_rel[pick] >= 0.5).astype(int)
            agree = (pseudo_batch == y_rel[pick]).mean()
            if agree >= agree_frac_threshold:
                return {
                    "labels": (p_rel >= 0.5).astype(int).tolist(),
                    "sentiment": y_sent.tolist(),
                    "probs": p_rel.tolist(),
                    "rounds": rounds,
                    "agree_frac": float(agree),
                }

        # fallback return after max rounds
        return {
            "labels": (p_rel >= 0.5).astype(int).tolist(),
            "sentiment": y_sent.tolist(),
            "probs": p_rel.tolist(),
            "rounds": rounds,
            "agree_frac": float((p_rel[np.where(y_rel >= 0)[0]] >= 0.5).astype(int) == y_rel[np.where(y_rel >= 0)[0]]).mean() if (y_rel >= 0).any() else 0.0,
        }

    # --- Small LLM bridge stubs (replace with your real wiring) ---
    def _llm_label(self, segment_ids: list[int], aspect_sentence: str):
        """
        Return (relevant:0/1, sentiment:-1/0/+1, confidence:0..1) for each id.
        TODO: implement via your LLM JSON-mode; this is a placeholder.
        """
        texts = [self.segments[i]["text"] for i in segment_ids]
        rel, sent, conf = [], [], []
        key = aspect_sentence.lower().split()[0]
        for t in texts:
            r = int(key in t.lower())
            s = 1 if any(w in t.lower() for w in ("great", "love", "excellent")) else (-1 if any(w in t.lower() for w in ("bad","terrible","hate")) else 0)
            rel.append(r); sent.append(s); conf.append(0.6)
        return rel, sent, conf



