# reproducibility.py
import numpy as np
from dataclasses import dataclass

import faiss

from foundation.vectorize_yelp import embed_texts

# ---------------- Config ----------------

@dataclass
class Config:
    top_k: int = 5
    n_samples: int = 10
    # llm_model: str = "gpt-5-mini"

# -------------- Experiment --------------

def run_reproducibility_experiment(embedder, info_by_id, cfg=Config()):
    vecs, index, chunk_infos = (
        embedder.vecs,
        embedder.index,
        embedder.chunk_infos,
    )
    """

    Given a set of reviews, embeddings, and a FAISS index, randomly sample
    an embedding and ask an LLM to guess the original text based on nearby
    embeddings.
    """
    # Reconstruct flattened spans to map embeddings back to text
    flat = [ci.text for ci in chunk_infos]

    rs = np.random.RandomState(0)
    for _ in range(cfg.n_samples):

        idx = int(rs.choice(len(vecs)))
        target_vec = vecs[idx][None, :]
        target_info = chunk_infos[idx]
        target_text = target_info.text
        user_id = target_info.user_id
        biz_id = target_info.business_id
        item_name = info_by_id.get(biz_id, {}).get("name") if biz_id else None

        D, I = index.search(target_vec.astype(np.float32), cfg.top_k + 1)
        neighbor_infos = [chunk_infos[j] for j in I[0][1:]]
        neighbors = [ni.text for ni in neighbor_infos]

        print("\n\n")
        print("=== Sample ===", idx)
        if user_id:
            print(f"User ID: {user_id}")
        if biz_id:
            if item_name:
                print(f"Business ID: {biz_id} ({item_name})")
            else:
                print(f"Business ID: {biz_id}")
        print(f"Target: {target_text}")
        print("Neighbors:")
        for ni in neighbor_infos:
            n_user = ni.user_id
            n_biz = ni.business_id
            n_name = info_by_id.get(n_biz, {}).get("name") if n_biz else None
            line = f"- {ni.text}"
            if n_user:
                line += f" | User ID: {n_user}"
            if n_biz:
                line += f" | Business ID: {n_biz}"
                if n_name:
                    line += f" ({n_name})"
            print(line)
