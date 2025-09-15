from pathlib import Path

import torch
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

from utils import dumpj, loadj, dumpp, loadp
import warnings

from tqdm import tqdm


@dataclass
class ChunkInfo:
    text: str
    user_id: str = None
    business_id: str = None
    review_id: str = None


class Yelp_Embedder_SEA:
    def __init__(self, args, reviews):

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.batch_size = 256
        self.normalize = True

        self.meta_path = args.cache_dir / f"meta_{args.div_name}.json"
        self.index_path = args.cache_dir / f"index_{args.div_name}.index"
        self.vec_path = args.cache_dir / f"vec_{args.div_name}.npy"
        self.chunk_info_path = args.cache_dir / f"chunk_info_{args.div_name}.pkl"

        if not self.meta_path.exists():
            print("[Yelp_Embedder_SEA] >>> meta does not exist. building...")
            self.build_embeddings(reviews)
            print("[Yelp_Embedder_SEA] >>> saving...")
            self.save()
            print("[Yelp_Embedder_SEA] >>> saved")
        else:
            print("[Yelp_Embedder_SEA] >>> meta exists. loading...")
            self.load()
            print("[Yelp_Embedder_SEA] >>> loaded")

    def build_embeddings(self, reviews):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            warnings.warn("CUDA is not available, using CPU instead.", RuntimeWarning)

        model = SentenceTransformer(self.model_name, device=device)
        max_tokens = 10

        self.chunk_infos = []
        for r in tqdm(reviews, ncols=88, desc='building spans per each review'):
            spans = extract_all_ngrams(r["text"], max_tokens)
            for span in spans:
                self.chunk_infos.append(
                    ChunkInfo(
                        text=span,
                        user_id=r.get("user_id"),
                        business_id=r.get("business_id"),
                        review_id=r.get("review_id"),
                    )
                )

        print('spans prepared, building embeddings')
        flat = [ci.text for ci in self.chunk_infos]
        if flat:
            self.vecs = embed_texts(flat, model, self.batch_size, self.normalize)
            self.index = build_index(self.vecs)
        else:
            dim = model.get_sentence_embedding_dimension()
            self.vecs = np.zeros((0, dim), dtype="float32")
            self.index = faiss.IndexFlatIP(dim)

        self.meta = {
            "model_name": self.model_name,
            "dim": int(self.vecs.shape[1]) if self.vecs.size else model.get_sentence_embedding_dimension(),
            "n_vectors": int(self.vecs.shape[0]),
            "n_reviews": len(reviews),
            "normalize": bool(self.normalize),
        }

    def save(self):
        dumpj(self.meta_path, self.meta)
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.vec_path, self.vecs)
        dumpp(self.chunk_info_path, self.chunk_infos)

    def load(self):
        self.meta = loadj(self.meta_path)
        self.index = faiss.read_index(str(self.index_path))
        self.vecs = np.load(self.vec_path)
        self.chunk_infos = loadp(self.chunk_info_path)


def build_index(vecs, index_type='flatip'):
    cfg = {
        "hnsw_M": 32,
        "hnsw_efConstruction": 80,
        "hnsw_efSearch": 64,
        "ivf_nlist": 8192,
        "ivf_nprobe": 32,
        "pq_m": 16,
        "pq_nbits": 8,
        "train_sample": 200_000,
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


def extract_all_ngrams(text, max_tokens):
    tokens = [t for t in text.strip().split() if t]
    spans = []
    n = len(tokens)
    for i in range(n):
        limit = min(n, i + max_tokens)
        for j in range(i + 1, limit + 1):
            spans.append(" ".join(tokens[i:j]))
    return spans


def get_model_max_tokens(model):
    if hasattr(model, "get_max_seq_length"):
        max_tokens = model.get_max_seq_length()
    else:
        max_tokens = getattr(model, "max_seq_length", None)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = 256
    return max_tokens


def embed_texts(texts, model, batch_size, normalize=True):
    with torch.no_grad():
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
    return embs.cpu().numpy().astype("float32")
