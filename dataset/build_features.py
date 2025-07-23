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
    def __init__(self, node_id, name, description, embedding):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.aliases = {name}
        self.embedding = embedding
        self.children = {}
        self.parent = None

    def update(self, alias: str):
        self.aliases.add(alias)

    def __repr__(self):
        return f"OntologyNode(name='{self.name}',\ndescription='{self.description}',\naliases={self.aliases})\n"


class Ontology:
    def __init__(self):
        self.nodes = {}
        self.review2node_id_score = defaultdict(list)

    def add_or_update_node(self, review_id, phrase, description, score):
        cleaned = clean_phrase(phrase)

        for node in self.nodes.values():
            if cleaned in node.aliases:
                self.review2node_id_score[review_id].append((node.node_id, score))
                print(f"Feature '{cleaned}' matched existing alias in node '{node.name}'")
                return

        top_candidates = self.search_top_ten(description)
        if top_candidates:
            candidates_text = "\n".join([f"{node.name}: {node.description}" for _, node in top_candidates])
            prompt = f"""
A new feature has been extracted from a review:

New Feature Name: {cleaned}
New Feature Definition: {description}

Below are existing features:
{candidates_text}

Decide the best relationship for the new feature:
- If it's a near synonym or alternative wording of an existing one, return: ALIAS: <existing name>
- If it's a more specific case of an existing feature, return: CHILD: <existing name>
- If it's a more general feature that should subsume an existing one, return: PARENT: <existing name>
- If no relationship, return: NEW
"""
            decision = query_llm(prompt).strip()
            print(f"Decision for '{cleaned}': {decision}")

            if decision.startswith("ALIAS:"):
                target = decision.split(":", 1)[1].strip()
                if target in self.nodes:
                    self.nodes[target].update(cleaned)
                    self.review2node_id_score[review_id].append((target, score))
                    print(f"'{cleaned}' added as ALIAS to '{target}'")
                    input('pause')
                    return

            elif decision.startswith("CHILD:"):
                parent = decision.split(":", 1)[1].strip()
                if parent in self.nodes:
                    node_id = cleaned
                    new_node = OntologyNode(node_id, cleaned, description, embed(cleaned))
                    new_node.parent = parent
                    self.nodes[parent].children[node_id] = new_node
                    self.nodes[node_id] = new_node
                    self.review2node_id_score[review_id].append((node_id, score))
                    print(f"'{cleaned}' added as CHILD to '{parent}'")
                    input('pause')
                    return

            elif decision.startswith("PARENT:"):
                child = decision.split(":", 1)[1].strip()
                if child in self.nodes:
                    node_id = cleaned
                    new_node = OntologyNode(node_id, cleaned, description, embed(cleaned))
                    new_node.children[child] = self.nodes[child]
                    self.nodes[child].parent = node_id
                    self.nodes[node_id] = new_node
                    self.review2node_id_score[review_id].append((node_id, score))
                    print(f"'{cleaned}' added as PARENT to '{child}'")
                    input('pause')
                    return

        node_id = cleaned
        self.nodes[node_id] = OntologyNode(node_id, cleaned, description, embed(cleaned))
        self.review2node_id_score[review_id].append((node_id, score))
        print(f"'{cleaned}' added as NEW node")
        input('pause')

    def search_top_ten(self, query_text, top_k=10):
        query_vec = embed(query_text)
        scored = []
        for node_id, node in self.nodes.items():
            sim = cosine(query_vec, node.embedding)
            scored.append((sim, node))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [node for node in scored[:top_k]]

    def to_dict(self):
        return {
            name: {
                "name": n.name,
                "description": n.description,
                "aliases": list(n.aliases),
                "children": list(n.children.keys()),
                "parent": n.parent,
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
        print("\n" + "="*60)
        print(f"Review: {text}")
        feature_scores = extract_feature_mentions(text, ontology, args.dset)
        print(f"Extracted Features: {feature_scores}")

        for phrase, description, score in feature_scores:
            phrase = clean_phrase(phrase)
            ontology.add_or_update_node(review_id, phrase, description, score)

        print("\nOntology after processing this review:")
        for node_id, node in ontology.nodes.items():
            print(f"- {node_id}: {node}")

        input("[Review Complete] Press Enter to continue...\n")

        count += 1
        if count == 50:
            check()

    ontology.cluster_features()
    ontology.infer_entailments()

    output_path = Path("cache") / "ontology.json"
    ontology.save(output_path)
    print(f"Ontology saved to {output_path} with {len(ontology.nodes)} nodes.")
    return ontology

def build_ontology_by_users_and_items():
    pass
