# reproducibility.py
import numpy as np
from dataclasses import dataclass

import faiss

from foundation.vectorize_yelp import split_to_spans, flatten_with_offsets, embed_texts, Config as VecConfig
from llm import query_llm

# ---------------- Config ----------------

@dataclass
class Config:
    model_name: str = VecConfig.model_name
    batch_size: int = VecConfig.batch_size
    max_chars: int = VecConfig.max_chars
    min_merge: int = VecConfig.min_merge
    normalize: bool = VecConfig.normalize
    top_k: int = 5
    n_samples: int = 3
    llm_model: str = "gpt-4.1-mini"

# -------------- Experiment --------------

def run_reproducibility_experiment(reviews, vecs, index, cfg: Config = Config()):
    """
    Given a set of reviews, embeddings, and a FAISS index, randomly sample
    an embedding and ask an LLM to guess the original text based on nearby
    embeddings.
    """
    # Reconstruct flattened spans to map embeddings back to text
    chunks = [split_to_spans(r["text"], cfg.max_chars, cfg.min_merge) for r in reviews]
    flat, _ = flatten_with_offsets(chunks)

    rs = np.random.RandomState(0)
    for _ in range(cfg.n_samples):
        idx = int(rs.choice(len(vecs)))
        target_vec = vecs[idx][None, :]
        target_text = flat[idx]

        D, I = index.search(target_vec.astype(np.float32), cfg.top_k + 1)
        neighbors = [flat[j] for j in I[0][1:]]

        prompt = "Nearby texts:\n" + "\n".join(f"- {t}" for t in neighbors)
        prompt += "\n\nGuess the hidden text in one sentence."

        guess = query_llm(prompt, model=cfg.llm_model).strip()
        guess_vec = embed_texts([guess], cfg.model_name, cfg.batch_size, cfg.normalize)[0]
        sim = float(np.dot(guess_vec, vecs[idx]))

        print("=== Sample ===")
        print(f"Target: {target_text}")
        print(f"Guess: {guess}")
        print(f"Similarity: {sim:.4f}\n")
