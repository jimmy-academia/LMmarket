# reproducibility.py
import numpy as np
from dataclasses import dataclass

import faiss

from foundation.vectorize_yelp import split_to_spans, flatten_with_offsets, embed_texts
from llm import query_llm

# ---------------- Config ----------------

@dataclass
class Config:
    top_k: int = 5
    n_samples: int = 3
    # llm_model: str = "gpt-5-mini"

# -------------- Experiment --------------

def run_reproducibility_experiment(embedder, cfg: Config = Config()):
    vecs, index, chunks = embedder.vecs, embedder.index, embedder.chunks
    """

    Given a set of reviews, embeddings, and a FAISS index, randomly sample
    an embedding and ask an LLM to guess the original text based on nearby
    embeddings.
    """
    # Reconstruct flattened spans to map embeddings back to text
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

        guess = query_llm(prompt).strip()
        guess_vec = embed_texts([guess], embedder.model_name, embedder.batch_size, embedder.normalize)[0]
        sim = float(np.dot(guess_vec, vecs[idx]))

        print("=== Sample ===")
        print(f"Target: {target_text}")
        print(f"neighbors: {neighbors}")
        print(f"Guess: {guess}")
        print(f"Similarity: {sim:.4f}\n")
