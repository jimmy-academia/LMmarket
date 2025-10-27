import numpy as np, torch, logging, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from utils import load_or_build, dumpp, loadp, _ensure_dir

# =========================================================
# Encoder
# =========================================================

class DenseEncoder:
    def __init__(self, model_name="bert-base-uncased", device=None, maxlen=256, batch_size=16):
        self.model_name = model_name
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True, truncation=True, max_length=self.maxlen,
            return_tensors="pt", return_offsets_mapping=True
        )

    @torch.no_grad()
    def embed_tokens(self, text):
        enc = self.tokenizer(
            [text],
            padding=False, truncation=True, max_length=self.maxlen,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        # Keep CPU copies for masks/offsets
        offs_cpu = enc["offset_mapping"][0].numpy()            # [L,2]
        spm_cpu  = enc["special_tokens_mask"][0].bool().numpy()# [L]
        am_cpu   = enc["attention_mask"][0].bool().numpy()     # [L]

        # Only pass model-accepted inputs to the model
        model_inputs = {
            k: enc[k].to(self.device)
            for k in ("input_ids", "attention_mask", "token_type_ids")
            if k in enc
        }
        out = self.model(**model_inputs).last_hidden_state[0]  # [L, D]

        # Keep = attended tokens minus special tokens (drop [CLS]/[SEP])
        keep = am_cpu & ~spm_cpu
        emb = out[keep]                                        # [T, D]
        emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy().astype("float32")
        offs = offs_cpu[keep].astype("int32")                  # align with filtered tokens
        return emb, offs

    @torch.no_grad()
    def embed_batch(self, texts):
        """Batch embed multiple reviews → list of token embeddings and offsets."""
        embs, offs = [], []
        for text in texts:
            e, o = self.embed_tokens(text)
            embs.append(e)
            offs.append(o)
        return embs, offs

    @torch.no_grad()
    def embed_query_tokens(self, query):
        return self.embed_tokens(query)[0]


# =========================================================
# Dense Searchable (single persistent embedding file)
# =========================================================

class DenseSearchable:
    def __init__(self, reviews, cache_dir, encoder: DenseEncoder, name="reviews[dense]"):
        self.name = name
        self.reviews = reviews or []
        self.encoder = encoder
        self.texts = [r.get("text", "") for r in self.reviews]
        self.base = _ensure_dir(Path(cache_dir) / name)

        # paths
        self.emb_path = self.base / "review_embs.npy"
        self.off_path = self.base / "review_offsets.pkl"

        # Build or load embeddings (saves offsets as a side-effect)
        self.doc_embs = load_or_build(
            self.emb_path,
            save_fn=lambda p, arrs: np.save(str(p), arrs, allow_pickle=True),
            load_fn=lambda p: np.load(str(p), allow_pickle=True),
            build_fn=self._build_embeddings,
        )
        # Load offsets ONCE
        self.doc_offs = loadp(self.off_path)

    def __len__(self):
        return len(self.texts)

    # ---- embedding builder ----
    def _build_embeddings(self):
        logging.info(f"[DenseSearchable] building all token embeddings for {len(self.texts)} reviews")
        embs, offs = [], []
        for txt in tqdm(self.texts, ncols=88, desc="encoding..."):
            e, o = self.encoder.embed_tokens(txt)
            embs.append(e); offs.append(o)
        dumpp(self.off_path, offs)                              # persist offsets
        return np.array(embs, dtype=object)                     # persist embeddings

    # ---- late interaction ----
    def _late_interaction(self, q_tok, d_tok):
        sims = q_tok @ d_tok.T
        per_q_max = sims.max(axis=1)
        score = float(per_q_max.sum())
        q_idx, t_idx = np.unravel_index(np.argmax(sims), sims.shape)
        return score, t_idx

    def _make_snippet(self, text, offs, t_idx, window=200):
        if len(offs) == 0: return text[:2*window] + "…" if len(text) > 2*window else text
        start, end = offs[min(t_idx, len(offs)-1)]
        left, right = max(0, start-window), min(len(text), end+window)
        snippet = text[left:right]
        anchor = text[start:end]
        snippet = snippet[:start-left] + "**" + anchor + "**" + snippet[start-left+len(anchor):]
        if left > 0: snippet = "…" + snippet
        if right < len(text): snippet += "…"
        return snippet

    # ---- main search ----
    def search(self, query, topk=5, silent=False):
        q_tok = self.encoder.embed_query_tokens(query)
        results = []
        for i, review in enumerate(self.reviews):
            d_emb = self.doc_embs[i]
            d_offs = self.doc_offs[i]              # use cached offsets
            score, t_idx = self._late_interaction(q_tok, d_emb)
            snippet = self._make_snippet(review["text"], d_offs, t_idx)
            rating = review.get("stars")
            results.append({"review": review, "score": score, "rating": rating, "snippet": snippet})

        results.sort(key=lambda x: x["score"], reverse=True)
        if not silent:
            print(f"\n🔍 {self.name} — top {topk} (late interaction):\n")
            for r in results[:topk]:
                rt = f"⭐:{r['rating']:.1f}" if isinstance(r["rating"], (int, float)) else ""
                print(f"{rt} | 📈:{r['score']:.2f} | {r['snippet']}")
        return results[:topk]
