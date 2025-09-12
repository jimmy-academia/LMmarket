from dataclasses import dataclass
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import warnings


@dataclass
class Config:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 256
    normalize: bool = True


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


def main(cfg: Config = Config()):
    # --- Define comments ---
    comments = {
        "good_fry": [
            "These truffle fries are crispy and bursting with earthy aroma.",
            "I love the rich, garlicky truffle flavor coating every fry.",
            "The truffle fries are perfectly seasoned and addictive.",
            "Each bite of the truffle fries delivers a decadent, savory punch.",
            "These truffle fries taste gourmet and are my new favorite snack.",
            "The fries have a great crunch and a luxurious truffle finish.",
            "Their truffle fries are heavenly and full of flavor.",
            "Crispy edges and deep truffle notes make these fries outstanding.",
            "The truffle seasoning elevates the fries to something special.",
            "I can't stop eating these aromatic, delicious truffle fries.",
        ],
        "bad_fry": [
            "The truffle fries were soggy and tasted artificial.",
            "I found the truffle fries bland with a weird aftertaste.",
            "These truffle fries had an overpowering, unpleasant smell.",
            "The fries were greasy and the truffle flavor was fake.",
            "Their truffle fries are stale and lack any real taste.",
            "The truffle fries were cold and disappointingly flavorless.",
            "I disliked the mushy texture of the truffle fries.",
            "The truffle fries left a bitter, lingering flavor.",
            "These truffle fries are overpriced and taste terrible.",
            "The truffle seasoning made the fries taste like chemicals.",
        ],
        "good_burger": [
            "The burger is juicy with a perfectly seasoned patty.",
            "I love the smoky char and tender meat in this burger.",
            "This burger tastes fresh and is packed with flavor.",
            "The bun is soft and the burger’s toppings blend beautifully.",
            "Every bite of the burger is savory and satisfying.",
            "The burger's melted cheese and juicy beef are incredible.",
            "This burger is delicious and cooked to perfection.",
            "The flavors in this burger are balanced and mouthwatering.",
            "The burger has a great sear and a rich, meaty taste.",
            "I can't get enough of this tasty, well-crafted burger.",
        ],
        "bad_burger": [
            "The burger was dry and lacked seasoning.",
            "I disliked the bland patty in this burger.",
            "This burger tasted old and had a soggy bun.",
            "The burger was greasy with a strange aftertaste.",
            "The toppings on the burger were wilted and unappetizing.",
            "I found the burger overcooked and tough to eat.",
            "This burger is flavorless and poorly made.",
            "The bun was stale and the burger fell apart.",
            "The burger's meat tasted cheap and unappealing.",
            "I regret ordering this burger; it was simply bad.",
        ],
    }

    # Flatten comments and keep labels
    texts = []
    labels = []
    for group, group_texts in comments.items():
        texts.extend(group_texts)
        labels.extend([group] * len(group_texts))

    # --- Embed comments ---
    embs = embed_texts(texts, cfg.model_name, cfg.batch_size, cfg.normalize)

    # Map embeddings by group
    vec_by_group = {}
    start = 0
    for group, group_texts in comments.items():
        n = len(group_texts)
        vec_by_group[group] = embs[start:start + n]
        start += n

    # --- 1. Clustering check ---
    for group, vecs in vec_by_group.items():
        idx = [i for i, g in enumerate(labels) if g == group]
        other_idx = [i for i, g in enumerate(labels) if g != group]
        same_sim = embs[idx] @ embs[idx].T
        # take upper triangle excluding diagonal
        triu = np.triu_indices(len(idx), k=1)
        mean_in = float(same_sim[triu].mean()) if len(triu[0]) > 0 else 0.0
        mean_out = float((embs[idx] @ embs[other_idx].T).mean())
        print(f"{group}: mean same-group sim {mean_in:.3f}, mean other-group sim {mean_out:.3f}")

    # --- 2. Group means ---
    group_means = {group: vecs.mean(axis=0) for group, vecs in vec_by_group.items()}

    # --- 3. Difference vectors ---
    good_bad_fry = group_means["good_fry"] - group_means["bad_fry"]
    good_bad_burger = group_means["good_burger"] - group_means["bad_burger"]
    cos_good_bad = np.dot(good_bad_fry, good_bad_burger) / (
        np.linalg.norm(good_bad_fry) * np.linalg.norm(good_bad_burger)
    )
    print(f"cos(good_bad_fry, good_bad_burger) = {cos_good_bad:.3f}")

    all_fry = np.concatenate([vec_by_group["good_fry"], vec_by_group["bad_fry"]], axis=0).mean(axis=0)
    all_burger = np.concatenate([vec_by_group["good_burger"], vec_by_group["bad_burger"]], axis=0).mean(axis=0)
    fry_vs_burger = all_fry - all_burger

    cos_fry_burger = np.dot(fry_vs_burger, good_bad_fry) / (
        np.linalg.norm(fry_vs_burger) * np.linalg.norm(good_bad_fry)
    )
    print(f"cos(fry_vs_burger, good_bad_fry) = {cos_fry_burger:.3f}")


if __name__ == "__main__":
    main()
