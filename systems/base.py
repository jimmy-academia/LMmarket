import re
import logging
import numpy as np
from collections import defaultdict
from rank_bm25 import BM25Okapi

class Searchable:
    def __init__(self, reviews):
        self.name = 'reviews'
        self.reviews = reviews 
        self.texts = [review['text'] for review in self.reviews]
        self._tokens = [self._tok(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def __len__(self):
        return len(self.texts)

    def _tok(self, t):
        return re.findall(r"\w+", (t or "").lower())

    def _make_snippet(self, text, q_tokens, window=300):
        low = text.lower()
        hit = next((t for t in q_tokens if t in low), None)
        if not hit:
            snippet = (text[:window*2] + "…") if len(text) > window*2 else text
            return snippet, None
        i = low.find(hit)
        start = max(0, i - window)
        end   = min(len(text), i + len(hit) + window)
        snippet = text[start:end]
        # bold the first hit
        snippet = snippet[:i-start] + "**" + text[i:i+len(hit)] + "**" + snippet[i-start+len(hit):]
        if start > 0: snippet = "…" + snippet
        if end < len(text): snippet = snippet + "…"
        return snippet, hit

    def search(self, query, topk=5, silent=False):
        q = self._tok(query)
        scores = self._bm25.get_scores(q)
        idx = np.argsort(scores)[::-1][:topk]
        if not silent: print(f"\n🔍 {self.name} — top {topk} for '{query}':\n")
        out = []
        for i in idx:
            review  = self.reviews[i]
            text = review['text']
            snippet, hit = self._make_snippet(text, q)
            if hit:
                rating_val = review.get("stars")
                rating_str = f"⭐ {rating_val}"
                score_str  = f"📈 {scores[i]:.2f}"

                prefix = f"{rating_str} | {score_str}" 
                if not silent: print(f"{prefix} | {snippet}")
                if not silent: print(f"→ keyword: {hit}\n")
                
                out.append({"review": review, "score": float(scores[i]),
                 "hit": hit, "rating": rating_val, "snippet": snippet})
                
        return out

class ItemSearchable:
    def __init__(self, items, review_searchable):
        self.name = 'items'
        self.item_star_name = {item['raw_info']['business_id']: [item['raw_info']['stars'], item['raw_info']['name']] for item in items}
        self.reviews = review_searchable

    def search(self, query, topk=5, topm=10, review_k=None, agg="sum", silent=False): # 'sum' | 'mean' | 'max' over topm review scores
        
        if review_k is None:
            review_k = topk*20

        hits = self.reviews.search(query, topk=review_k, silent=True)
        
        item_buckets = defaultdict(list)
        for h in hits:
            item_buckets[h['review']['item_id']].append(h)
        
        results = []
        for iid, hlist in item_buckets.items():
            hlist.sort(key=lambda z: z["score"], reverse=True)
            if agg == "max":
                item_score = hlist[0]["score"]
            elif agg == "mean":
                item_score = sum(x["score"] for x in hlist) / len(hlist)
            else:  # sum
                item_score = sum(x["score"] for x in hlist)

            best = hlist[0]
            snippet = best.get("snippet") 

            rating, item_name = self.item_star_name[iid]
            results.append({
                "item_id": iid,
                "score": float(item_score),
                "rating": rating,
                "snippet": snippet,
                "hits": hlist,     # all review hits for this item (desc by score)
                "item_name": item_name
            })

        # 4) Rank and print
        results.sort(key=lambda r: r["score"], reverse=True)
        if not silent: print(f"\n🔍 {self.name} — top {topk} for '{query}':\n")
        for r in results[:topk]:
            rating_str = f"⭐ {r['rating']}"
            score_str  = f"📈:{r['score']:.2f}"
            prefix = f"{rating_str} | {score_str}" 
            if not silent: print(f"{prefix} | name: {r["item_name"]} | {r['snippet']}")

        return results[:topk]

class BaseSystem:
    '''
    provides
    self.reviews [list of reviews (dict)]
    self.segments [list of segments (dict)]
    self.embedding (segments)
    '''
    def __init__(self, args, data):
        self.args = args
        
        self.reviews = Searchable(data['reviews'])
        self.items = ItemSearchable(data['items'], self.reviews)
        
        


# dense_search.py
import os
import re
import faiss
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import logging

# bring in your utils
from utils import load_or_build, dumpp, loadp, _ensure_dir

# ---------------------------
# Helpers
# ---------------------------

def _save_numpy(path: Path, arr: np.ndarray):
    np.save(str(path), arr)

def _load_numpy(path: Path) -> np.ndarray:
    return np.load(str(path), mmap_mode=None, allow_pickle=False)

def _save_faiss(path: Path, index):
    faiss.write_index(index, str(path))

def _load_faiss(path: Path):
    return faiss.read_index(str(path))

def _model_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)

# ---------------------------
# Encoder (HF) 
# ---------------------------

@dataclass
class EncoderCfg:
    model_name: str = "bert-base-uncased"
    device: Optional[str] = None
    doc_maxlen: int = 256      # for doc-level vector (stage 1)
    tok_maxlen: int = 256      # for token-level embeddings (late interaction)
    batch_size: int = 32

class HFTextEncoder:
    def __init__(self, cfg: EncoderCfg):
        self.cfg = cfg
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def doc_vectors(self, texts: List[str]) -> np.ndarray:
        """Mean-pool contextual token embeddings → L2-normalized doc vector."""
        bs = self.cfg.batch_size
        out = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.cfg.doc_maxlen,
                return_tensors="pt"
            ).to(self.device)
            out_h = self.model(**enc).last_hidden_state  # [B, L, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            summed = (out_h * mask).sum(dim=1)                  # [B, D]
            denom = mask.sum(dim=1).clamp(min=1e-6)             # [B, 1]
            mean = summed / denom
            mean = torch.nn.functional.normalize(mean, dim=-1)
            out.append(mean.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def token_embeddings(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (embeddings[T, D], offsets[T, 2]) for non-special tokens.
        Embeddings are L2-normalized per token.
        """
        enc = self.tokenizer(
            text, padding=False, truncation=True, max_length=self.cfg.tok_maxlen,
            return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True
        )
        enc = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in enc.items()}
        outputs = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        hidden = outputs.last_hidden_state[0]                        # [L, D]
        offs = enc["offset_mapping"][0].cpu().numpy()               # [L, 2]
        spm = enc["special_tokens_mask"][0].bool().cpu().numpy()    # [L]
        am = enc["attention_mask"][0].bool().cpu().numpy()          # [L]

        keep = (~spm) & am
        tok_emb = hidden[keep]
        tok_emb = torch.nn.functional.normalize(tok_emb, dim=-1).cpu().numpy().astype("float32")
        tok_offs = offs[keep].astype("int32")
        return tok_emb, tok_offs

    @torch.no_grad()
    def query_token_embeddings(self, query: str) -> np.ndarray:
        emb, _ = self.token_embeddings(query)
        return emb  # [Q, D], already L2-normalized

# ---------------------------
# Dense index (two-stage)
# ---------------------------

class DenseSearchable:
    """
    Two-stage dense retrieval:
      Stage 1: doc-level ANN filter (mean-pooled vector)
      Stage 2: ColBERT-style late interaction on candidates (token embeddings)
    Uses load_or_build to persist doc vecs, FAISS, and per-doc token embeddings.
    """
    def __init__(
        self,
        reviews: List[Dict[str, Any]],
        cache_dir: str,
        encoder: HFTextEncoder,
        index_type: str = "hnsw",   # "flat" | "hnsw"
        name: str = "reviews[dense]"
    ):
        self.name = name
        self.reviews = reviews or []
        self.encoder = encoder

        self.texts = [r.get("text", "") for r in self.reviews]
        # prefer stable per-review id
        self.doc_ids = [
            r.get("review_id") or (r.get("raw_info", {}) or {}).get("review_id") or f"idx_{i}"
            for i, r in enumerate(self.reviews)
        ]

        slug = _model_slug(self.encoder.cfg.model_name)
        base = Path(cache_dir) / f"dense_{slug}_D{self.encoder.cfg.doc_maxlen}_T{self.encoder.cfg.tok_maxlen}"
        self.base = _ensure_dir(base)
        self.tokens_dir = _ensure_dir(self.base / "doc_tokens")

        # ---- persist doc vectors ----
        self.docvecs_path = self.base / "docvecs.npy"
        self.docvecs = load_or_build(
            self.docvecs_path,
            save_fn=_save_numpy,
            load_fn=_load_numpy,
            build_fn=lambda texts: self.encoder.doc_vectors(texts),
            texts=self.texts,
        )

        # ---- persist doc ids ----
        self.docids_path = self.base / "doc_ids.pkl"
        self.doc_ids = load_or_build(
            self.docids_path,
            save_fn=dumpp,
            load_fn=loadp,
            build_fn=lambda ids: ids,
            ids=self.doc_ids,
        )

        # ---- persist ANN index over docvecs ----
        self.index_path = self.base / "faiss.index"
        self.index = load_or_build(
            self.index_path,
            save_fn=_save_faiss,
            load_fn=_load_faiss,
            build_fn=self._build_faiss,
            docvecs=self.docvecs, index_type=index_type
        )

    # --- FAISS ---
    def _build_faiss(self, docvecs: np.ndarray, index_type: str = "hnsw"):
        d = docvecs.shape[1]
        if index_type == "flat":
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexHNSWFlat(d, 32)  # cosine via IP on L2-normalized vectors
            index.hnsw.efConstruction = 200
        index.add(docvecs.astype("float32", copy=False))
        return index

    # --- per-doc tokens persisted lazily ---
    def _doc_token_path(self, doc_id: str) -> Path:
        return self.tokens_dir / f"{doc_id}.npz"

    def _save_doc_tokens(self, path: Path, obj):
        np.savez_compressed(str(path), emb=obj["emb"], offs=obj["offs"])

    def _load_doc_tokens(self, path: Path):
        z = np.load(str(path))
        return {"emb": z["emb"], "offs": z["offs"]}

    def _build_doc_tokens(self, text: str):
        emb, offs = self.encoder.token_embeddings(text)
        return {"emb": emb, "offs": offs}

    def _ensure_doc_tokens(self, doc_idx: int):
        did = self.doc_ids[doc_idx]
        path = self._doc_token_path(did)
        return load_or_build(
            path,
            save_fn=self._save_doc_tokens,
            load_fn=self._load_doc_tokens,
            build_fn=self._build_doc_tokens,
            text=self.texts[doc_idx]
        )

    # --- scoring + snippets ---
    def _late_interaction(self, q_tok: np.ndarray, d_tok: np.ndarray) -> Tuple[float, Tuple[int,int]]:
        """
        ColBERT-style: sum_q max_t q·d
        Returns (score, (best_q_idx, best_t_idx)) for snippet building.
        """
        # q_tok [Q, D], d_tok [T, D] ; both L2-normalized
        sims = q_tok @ d_tok.T                      # [Q, T]
        per_q_max = sims.max(axis=1)                # [Q]
        score = float(per_q_max.sum())
        # find the single strongest match for snippet anchor
        q_idx, t_idx = np.unravel_index(np.argmax(sims), sims.shape)
        return score, (int(q_idx), int(t_idx))

    def _dense_snippet(self, text: str, offs: np.ndarray, t_idx: int, window_chars: int = 200) -> str:
        """
        Expand around the highest-contributing doc token using HuggingFace offsets.
        """
        if t_idx < 0 or t_idx >= len(offs):
            # fallback: take the beginning
            return (text[:2*window_chars] + "…") if len(text) > 2*window_chars else text
        start, end = offs[t_idx]
        if end <= start:
            return (text[:2*window_chars] + "…") if len(text) > 2*window_chars else text
        left = max(0, start - window_chars)
        right = min(len(text), end + window_chars)
        snippet = text[left:right]
        # Bold the anchor substring
        anchor = text[start:end]
        snippet = snippet[:start-left] + "**" + anchor + "**" + snippet[start-left+len(anchor):]
        if left > 0: snippet = "…" + snippet
        if right < len(text): snippet = snippet + "…"
        return snippet

    # --- public search ---
    def search(
        self,
        query: str,
        topk: int = 5,
        stage1_k: Optional[int] = None,
        silent: bool = False,
        window_chars: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Dense retrieval with late interaction + snippet from best token match.
        Prints:  ⭐:<stars> | 📈:<dense_score> | <snippet with **bold** anchor>
        """
        if stage1_k is None:
            stage1_k = max(topk * 50, 200)

        # Stage 1: ANN over doc vectors
        q_doc = self.encoder.doc_vectors([query])  # [1, D] already L2-norm
        sims, idx = self.index.search(q_doc.astype("float32"), stage1_k)  # IP ~ cosine
        cand_idx = idx[0].tolist()

        # Stage 2: late interaction
        q_tok = self.encoder.query_token_embeddings(query)  # [Q, D]
        scored = []
        for di in cand_idx:
            tokens = self._ensure_doc_tokens(di)            # {"emb": [T,D], "offs": [T,2]}
            d_emb, d_offs = tokens["emb"], tokens["offs"]
            score, (_, t_idx) = self._late_interaction(q_tok, d_emb)
            review = self.reviews[di]
            snippet = self._dense_snippet(review["text"], d_offs, t_idx, window_chars=window_chars)
            rating = review.get("stars")
            scored.append({
                "review": review,
                "score": score,                 # dense late-interaction score
                "rating": rating,
                "snippet": snippet,
                "doc_idx": di,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:topk]

        if not silent:
            print(f"\n🔍 {self.name} — top {topk} (dense) for '{query}':\n")
            for r in results:
                rating_tok = f"⭐:{r['rating']:.1f}" if isinstance(r["rating"], (int, float)) else ""
                score_tok  = f"📈:{r['score']:.2f}"
                prefix = f"{rating_tok} | {score_tok}" if rating_tok else score_tok
                print(f"{prefix} | {r['snippet']}")

        return results

# ---------------------------
# Item proxy over dense review hits
# ---------------------------

class ItemsFromReviewsDenseSearchable:
    """
    Item-level dense search that PROXIES to dense review search and groups by item_id.
    No item index is ever built.
    """
    def __init__(self, items: List[Dict[str, Any]], review_dense: DenseSearchable, name="items[dense]"):
        self.name = name
        self.reviews_dense = review_dense
        # Map business/item → (stars, name)
        self.item_meta = {}
        for it in items or []:
            raw = it.get("raw_info", {}) or {}
            bid = raw.get("business_id")
            stars = raw.get("stars")
            name = raw.get("name") or bid
            if bid:
                self.item_meta[bid] = (stars, name)

    def _iid_from_review(self, r: Dict[str, Any]) -> Optional[str]:
        return r.get("item_id") or r.get("business_id") or (r.get("raw_info", {}) or {}).get("business_id")

    def search(
        self,
        query: str,
        topk: int = 5,
        topm: int = 10,
        stage1_k: Optional[int] = None,
        agg: str = "sum",              # 'sum' | 'mean' | 'max' over topm review scores
        silent: bool = False,
    ):
        # Pull many review hits, then group
        review_k = max(topk * topm * 4, 200) if stage1_k is None else stage1_k
        hits = self.reviews_dense.search(query, topk=review_k, stage1_k=review_k, silent=True)

        from collections import defaultdict
        buckets = defaultdict(list)
        for h in hits:
            iid = self._iid_from_review(h["review"])
            if iid:
                buckets[iid].append(h)

        results = []
        for iid, hlist in buckets.items():
            hlist.sort(key=lambda z: z["score"], reverse=True)
            use = hlist[:max(1, topm)]
            if agg == "max":
                item_score = use[0]["score"]
            elif agg == "mean":
                item_score = sum(x["score"] for x in use) / len(use)
            else:
                item_score = sum(x["score"] for x in use)

            # best snippet from best review
            best = use[0]
            snippet = best["snippet"]

            stars, name = self.item_meta.get(iid, (None, str(iid)))
            results.append({
                "item_id": iid,
                "item_name": name,
                "rating": stars,
                "score": float(item_score),
                "snippet": snippet,
                "hits": hlist,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        out = results[:topk]

        if not silent:
            print(f"\n🔍 {self.name} — top {topk} (dense) for '{query}':\n")
            for r in out:
                rating_tok = f"⭐:{r['rating']:.1f}" if isinstance(r["rating"], (int, float)) else ""
                score_tok  = f"📈:{r['score']:.2f}"
                prefix = f"{rating_tok} | {score_tok}" if rating_tok else score_tok
                print(f"{prefix} | name: {r['item_name']} | {r['snippet']}")

        return out
