from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from pathlib import Path
import numpy as np
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as csim
from utils import readf

from llm import query_llm
from debug import check

### --- Embedding Setup --- ###

embed_model = SentenceTransformer("BAAI/bge-small-en")

def embed(text: str) -> np.ndarray:
    return embed_model.encode(text, normalize_embeddings=True)

def cosine(a, b):
    return float(csim(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def clean_phrase(phrase: str) -> str:
    return phrase.lower().strip("* ").split(". ")[-1].strip()

### --- Ontology Node and Ontology Structure --- ###

class OntologyNode:
    def __init__(self, name: str, description: str, score: float, review_id: str, emb):
        self.name = name
        self.description = description
        self.aliases = {name}
        self.examples = [(review_id, name, score)]
        self.embedding = emb
        self.cluster_id = None
        self.entailments = set()

    def update(self, alias: str, score: float, review_id: str):
        cleaned = clean_phrase(alias)
        self.aliases.add(cleaned)
        self.examples.append((review_id, cleaned, score))

    def __repr__(self):
        return f"OntologyNode(name='{self.name}', description='{self.description}', aliases={self.aliases})"


class Ontology:
    def __init__(self):
        self.nodes: Dict[str, OntologyNode] = {}

    def add_or_update_node(self, phrase: str, description: str, score: float, review_id: str, threshold: float = 0.94):
        cleaned = clean_phrase(phrase)
        emb = embed(description)
        best_node, best_score = None, -1
        for node in self.nodes.values():
            sim = cosine(emb, node.embedding)
            if sim > best_score:
                best_score, best_node = sim, node

        if best_score > threshold:
            best_node.update(cleaned, score, review_id)
        elif cleaned in self.nodes:
            self.nodes[cleaned].update(cleaned, score, review_id)
        else:
            self.nodes[cleaned] = OntologyNode(cleaned, description, score, review_id, emb)

    def to_dict(self):
        return {
            name: {
                "name": n.name,
                "description": n.description,
                "aliases": list(n.aliases),
                "examples": n.examples,
                "entailments": list(n.entailments),
                "cluster_id": n.cluster_id,
            } for name, n in self.nodes.items()
        }

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def feature_hints(self, max_count=15):
        names = [n.name for n in self.nodes.values()]
        return sorted(set(names))[:max_count]

    def cluster_features(self, threshold: float = 0.9):
        names = list(self.nodes.keys())
        embeddings = [self.nodes[name].embedding for name in names]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                sim = cosine(embeddings[i], embeddings[j])
                if sim > threshold:
                    self.nodes[names[i]].cluster_id = i
                    self.nodes[names[j]].cluster_id = i

    def infer_entailments(self, threshold: float = 0.93):
        names = list(self.nodes.keys())
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    sim = cosine(self.nodes[names[i]].embedding, self.nodes[names[j]].embedding)
                    if sim > threshold:
                        self.nodes[names[i]].entailments.add(names[j])


### --- Feature + Sentiment Extraction --- ###

def extract_feature_mentions(text: str, ontology: Ontology = None, dset = None, model="openai") -> List[Tuple[str, float]]:
    hints = ontology.feature_hints() if ontology else []
    hint_str = f"Known features to look for include: {', '.join(hints)}." if hints else ""

    if dset == 'yelp':
        domain = 'restaurant'
        cares = 'service, food, timing, pricing, experience, etc.'
    elif dset == 'amazon':
        domain = 'product'
        cares = 'build quality, ease of use, packaging, delivery experience, pricing, performance, compatibility, aesthetics, etc.'
    else:
        domain = 'general'
        cares = 'anything.'

    prompt = f"""
You are a careful and accurate reviewer analysis assistant.

You are analyzing a {domain} review to extract key features that the user evaluated. Follow this two-step process:

Step 1: Identify distinct feature concepts described in the review — what aspects did the user seem to care about? These could relate to {cares}

Step 2: For each concept, provide:
- A short, lowercase phrase that neutrally names the feature (e.g. use "wait time" not "long wait time")
- A one-sentence abstract definition that clearly explains what the feature refers to in general. Briefly describe what a positive and a negative score would mean for that feature (e.g., “short wait times are positive; long delays are negative”)
- A sentiment score between -1.0 and +1.0 indicating how the feature is portrayed in this specific review

If relevant, align your phrasing with known features: {hint_str} — otherwise invent a reasonable new phrase.

Be precise. Avoid using the same phrase for unrelated meanings.

Review:
"{text.strip()}"

Output format:
feature name | definition | score (float between -1.0 and 1.0)


"""

    response = query_llm(prompt, model=model)

    results = []
    for line in response.strip().splitlines():
        if "|" in line:
            try:
                parts = line.split("|")
                phrase = clean_phrase(parts[0])
                definition = parts[1].strip()
                score = float(parts[2].strip())
                results.append((phrase, definition, score))
            except Exception:
                continue
    return results

### --- Main Ontology Building Function --- ###

def build_ontology_by_reviews(args, reviews: List[Dict]) -> Ontology:
    ontology = Ontology()
    print('ontology class initialized')

    count = 0
    for r in reviews:
        review_id, text = r["review_id"], r["text"]
        print(text)
        feature_scores = extract_feature_mentions(text, ontology, args.dset)
        print(feature_scores)

        for phrase, description, score in feature_scores:
            ontology.add_or_update_node(phrase, description, score, review_id)

        count += 1
        if count == 5:
            check()

    ontology.cluster_features()
    ontology.infer_entailments()

    output_path = Path("cache") / "ontology.json"
    ontology.save(output_path)
    print(f"Ontology saved to {output_path} with {len(ontology.nodes)} nodes.")
    return ontology

def build_ontology_by_users_and_items():
    pass