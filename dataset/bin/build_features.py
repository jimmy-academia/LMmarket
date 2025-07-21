from collections import defaultdict
from typing import List, Dict, Any
import networkx as nx
import re

def build_ontology_by_reviews(USERS: List[Dict], domain_name: str) -> nx.DiGraph:
    """
    Build a semantic DAG of features based on user review text and extracted preferences.
    - Alias groups similar phrases (via LLM or heuristic)
    - Entailment edges added as directed semantic relationships
    - Nodes are canonical normalized features
    """
    # --- Step 1: Initialize structures ---
    G = nx.DiGraph()
    alias_table = {}
    canonical_features = set()
    feature_to_examples = defaultdict(list)

    # --- Step 2: Extract features from user reviews ---
    for user in users:
        uid = user["user_id"]
        description = user.get("text_description", "")
        for raw_feat in user.get("preferences", []):
            norm_feat = normalize_feature(raw_feat)
            canonical_feat = resolve_alias(norm_feat, canonical_features, alias_table)
            canonical_features.add(canonical_feat)
            alias_table[norm_feat] = canonical_feat
            feature_to_examples[canonical_feat].append({
                "user_id": uid,
                "raw": raw_feat,
                "normalized": norm_feat,
                "context": description
            })

    # --- Step 3: Add nodes to graph ---
    for feat in canonical_features:
        G.add_node(feat)

    # --- Step 4: Add soft entailment edges ---
    feature_list = list(canonical_features)
    for i in range(len(feature_list)):
        for j in range(len(feature_list)):
            if i == j:
                continue
            a, b = feature_list[i], feature_list[j]
            entailment_score = check_entailment(a, b, domain=domain_name)
            if entailment_score >= 0.7:
                G.add_edge(a, b, weight=round(entailment_score, 2))

    # Optional: annotate aliases
    G.graph['alias_table'] = alias_table
    return G

# --- Utility functions ---

def normalize_feature(feat: str) -> str:
    feat = feat.lower().strip()
    feat = re.sub(r'[^\w\s]', '', feat)
    return re.sub(r'\s+', ' ', feat)

def resolve_alias(feat: str, known_feats: set, alias_table: Dict[str, str]) -> str:
    """
    Check if feat is a near-duplicate of any known feature.
    If so, return the existing one; else return itself.
    """
    for existing in known_feats:
        if is_alias(feat, existing):
            return existing
    return feat

# --- LLM placeholder (you'll later call LLM here) ---

def is_alias(a: str, b: str) -> bool:
    # Placeholder for LLM call or embedding similarity
    return a == b  # simple for now

def check_entailment(a: str, b: str, domain: str = "") -> float:
    # Placeholder for LLM entailment judgment
    # Should return score between 0 and 1
    return 0.0  # no entailment by default
