from pathlib import Path

import torch
import numpy as np

import faiss
from blingfire import text_to_sentences
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from utils import dumpj, loadj, dumpp, loadp
import warnings
# ---------------- Config ----------------


class Yelp_Embedder:
    def __init__(self, args, reviews):

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = 256
        self.max_chars = 100
        self.min_merge = 20
        self.normalize = True


        self.meta_path = args.cache_dir/f"meta_{args.div_name}.json"
        self.index_path = args.cache_dir/f"index_{args.div_name}.index"
        self.vec_path = args.cache_dir/f"vec_{args.div_name}.npy"
        self.offset_chunks_path = args.cache_dir/f"offset_chunks_{args.div_name}.pkl"

        if not self.meta_path.exists():
            print("[Yelp_Embedder] >>> meta does not exist. building...")
            self.build_embeddings(reviews)
            print("[Yelp_Embedder] >>> saving...")
            self.save()
            print("[Yelp_Embedder] >>> saved")
        else:
            print("[Yelp_Embedder] >>> meta exists. loading...")
            self.load()
            print("[Yelp_Embedder] >>> loaded")


    def build_embeddings(self, reviews):

        reviews = [r['text'] for r in reviews]
        self.chunks = [split_to_spans(rtext, self.max_chars, self.min_merge) for rtext in reviews]
        flat, self.offsets = flatten_with_offsets(self.chunks)
        self.vecs = embed_texts(flat, self.model_name, self.batch_size, self.normalize)
        index = build_index(self.vecs)
        self.index = faiss.index_gpu_to_cpu(index)
        self.meta = {
            "model_name": self.model_name,
            "dim": int(self.vecs.shape[1]),
            "n_vectors": int(self.vecs.shape[0]),
            "n_reviews": len(self.chunks),
            "normalize": bool(self.normalize),
        }

    def save(self):
        dumpj(self.meta_path, self.meta)
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.vec_path, self.vecs)
        dumpp(self.offset_chunks_path, (self.offsets, self.chunks))
        ## save the files

    def load(self):
        ## load the files to self.variables
        self.meta = loadj(self.meta_path)
        self.index = faiss.read_index(str(self.index_path))
        self.vecs = np.load(self.vec_path)
        self.offsets, self.chunks = loadp(self.offset_chunks_path)


# -------------- Quick search --------------

def load_index_and_search(bundle, queries, top_k=10):
    index = faiss.read_index(bundle["paths"]["index"])
    D, I = index.search(queries.astype(np.float32), top_k)
    return I, D

# -------------- Modules --------------

import re

def _smart_wrap_sentence(s: str, max_chars: int):
    """
    Split a single (possibly very long) sentence into chunks <= max_chars.
    Prefer strong punctuation, then weak punctuation/space, else hard cut.
    """
    if len(s) <= max_chars:
        return [s]

    strong = [". ", "! ", "? ", "。", "！", "？", "…"]
    weak   = ["; ", ": ", ", ", "，", "、", "—", "–", " - ", " "]

    out, i, n = [], 0, len(s)
    while i < n:
        # if remaining fits, done
        if n - i <= max_chars:
            out.append(s[i:].strip())
            break

        window = s[i:i+max_chars+1]

        # try strongest breakpoints within window (nearest to the end)
        cut = max((window.rfind(p) for p in strong), default=-1)
        if cut == -1:
            cut = max((window.rfind(p) for p in weak), default=-1)

        if cut == -1:
            # no good breakpoint; hard cut at max_chars
            cut = max_chars
        else:
            # cut index is at the start of the matched pattern; include it
            cut += len(window[cut:cut+1])  # keep the punctuation/space edge

        chunk = s[i:i+cut].strip()
        if chunk: out.append(chunk)
        i += cut
        # skip any extra spaces
        while i < n and s[i].isspace():
            i += 1
    return out


def split_to_spans(text, max_chars=100, min_merge=40):
    """
    1) Sentence-split with blingfire.
    2) Smart-wrap any sentence > max_chars using punctuation-aware splitting.
    3) Pack wrapped sentences into spans with max_chars cap.
    4) Merge tiny spans (< min_merge) with the following one.
    """
    sents = text_to_sentences(text.strip()).split("\n")

    # step 1–2: wrap any long sentence first
    wrapped = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_chars:
            wrapped.extend(_smart_wrap_sentence(s, max_chars))
        else:
            wrapped.append(s)

    # step 3: pack into spans up to max_chars
    spans, buf = [], ""
    for s in wrapped:
        if buf and len(buf) + 1 + len(s) > max_chars:
            spans.append(buf)
            buf = s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        spans.append(buf)

    # step 4: merge tiny spans
    out, i = [], 0
    while i < len(spans):
        if len(spans[i]) < min_merge and i + 1 < len(spans):
            out.append((spans[i] + " " + spans[i+1]).strip())
            i += 2
        else:
            out.append(spans[i])
            i += 1
    return out


def flatten_with_offsets(chunks):
    """Flatten list-of-lists and record start/end offsets for each group."""
    flat = [s for review in chunks for s in review]
    lengths = [len(review) for review in chunks]
    offsets = np.cumsum([0] + lengths)  # shape (n_reviews+1,)
    return flat, offsets

def embed_texts(texts, model_name, batch_size, normalize=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        warnings.warn("CUDA is not available, using CPU instead.", RuntimeWarning)


    model = SentenceTransformer(model_name, device=device)
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
    return embs.cpu().numpy().astype("float32")

# -------------- FAISS index --------------
def build_index(vecs):
    d = vecs.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)  # inner product (use L2-normalized vecs for cosine)
    index.add(vecs)
    return index

