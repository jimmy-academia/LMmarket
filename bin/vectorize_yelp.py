from pathlib import Path

import torch
import numpy as np

import faiss
from blingfire import text_to_sentences
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from utils import dumpj, loadj, dumpp, loadp
import warnings

# ---------------- Class ----------------

@dataclass
class ChunkInfo:
    text: str
    user_id: str = None
    business_id: str = None
    review_id: str = None

class Yelp_Embedder:
    def __init__(self, args, reviews):

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = 256
        self.max_chars = 50
        self.min_merge = 20
        self.normalize = True


        self.meta_path = args.cache_dir/f"meta_{args.div_name}.json"
        self.index_path = args.cache_dir/f"index_{args.div_name}.index"
        self.vec_path = args.cache_dir/f"vec_{args.div_name}.npy"
        self.chunk_info_path = args.cache_dir/f"chunk_info_{args.div_name}.pkl"

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
        self.chunk_infos = []
        for r in reviews:
            spans = split_to_spans(r["text"], self.max_chars, self.min_merge)
            for span in spans:
                self.chunk_infos.append(
                    ChunkInfo(
                        text=span,
                        user_id=r.get("user_id"),
                        business_id=r.get("business_id"),
                        review_id=r.get("review_id"),
                    )
                )

        flat = [ci.text for ci in self.chunk_infos]
        self.vecs = embed_texts(flat, self.model_name, self.batch_size, self.normalize)
        self.index = build_index(self.vecs)
        self.meta = {
            "model_name": self.model_name,
            "dim": int(self.vecs.shape[1]),
            "n_vectors": int(self.vecs.shape[0]),
            "n_reviews": len(reviews),
            "normalize": bool(self.normalize),
        }

    def save(self):
        dumpj(self.meta_path, self.meta)
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.vec_path, self.vecs)
        dumpp(self.chunk_info_path, self.chunk_infos)
        ## save the files

    def load(self):
        ## load the files to self.variables
        self.meta = loadj(self.meta_path)
        self.index = faiss.read_index(str(self.index_path))
        self.vecs = np.load(self.vec_path)
        self.chunk_infos = loadp(self.chunk_info_path)


# -------------- FAISS index --------------
def build_index(vecs, index_type = 'flatip'):
    clf = {
        # HNSW params
        "hnsw_M": 32,                 # number of neighbors in graph (16–48 typical)
        "hnsw_efConstruction": 80,    # construction search depth (80–200)
        "hnsw_efSearch": 64,          # search depth (32–128)

        # IVF-PQ params
        "ivf_nlist": 8192,            # number of coarse clusters (≈√N is a rule of thumb)
        "ivf_nprobe": 32,             # clusters to probe at search time (8–64 typical)
        "pq_m": 16,                   # number of subquantizers (must divide dim)
        "pq_nbits": 8,                # bits per subquantizer (8 is standard)
        "train_sample": 200_000,      # number of vectors used to train PQ
    }

    d = vecs.shape[1]
    if index_type.upper() == "HNSW":
        index = faiss.IndexHNSWFlat(d, cfg["hnsw_M"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg["hnsw_efConstruction"]
        index.add(vecs)
        return index
    if index_type.upper() == "IVF-PQ":
        quant = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quant, d, cfg["ivf_nlist"], cfg["pq_m"], 8, faiss.METRIC_INNER_PRODUCT)
        n = vecs.shape[0]
        if n <= cfg["train_sample"]:
            train_x = vecs
        else:
            rs = np.random.RandomState(123)
            train_x = vecs[rs.choice(n, size=cfg["train_sample"], replace=False)]
        index.train(train_x)
        index.add(vecs)
        return index
    else:
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
    return index

# -------------- Modules --------------

import re

def _smart_wrap_sentence(s, max_chars):
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

