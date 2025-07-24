from collections import defaultdict
from pathlib import Path

from utils import readf, vprint, pause_if
from utils import dumpj, loadj

from llm import query_llm
from debug import check
from functools import partial

### --- Flags --- ###
VERBOSE = True
PAUSE = True
vlog = partial(vprint, flag=VERBOSE)
ppause = partial(pause_if, flag=PAUSE)
M = 10  # Max children under a root before reordering

### --- Ontology Node and Ontology Structure --- ###

class OntologyNode:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.aliases = {name}
        self.children = []
        self.parent = None

    def update(self, alias: str):
        self.aliases.add(alias)

    def __repr__(self):
        parent_str = f"\nparent='{self.parent.name}'" if self.parent else ""
        child_str = ""
        if self.children:
            child_str = "\nchildren=[" + ", ".join([child.name for child in self.children]) + "]"

        return (
            f"OntologyNode(name='{self.name}',"
            f"\ndescription='{self.description}',"
            f"\naliases={self.aliases}"
            f"{parent_str}"
            f"{child_str})\n"
        )

class Ontology:
    def __init__(self):
        self.nodes = []
        self.roots = []
        self.review2name_score = defaultdict(list)
        self.K = 20

    def add_or_update_node(self, review_id, phrase, description, score):
        cleaned = clean_phrase(phrase)

        node = self.find_or_create_node(cleaned, description)

        self.review2name_score[review_id].append((node.name, score))

    def find_or_create_node(self, cleaned, description):
        for node in self.nodes:
            if cleaned in node.aliases:
                vlog(f"Feature '{cleaned}' matched existing alias in node '{node.name}'")
                return node

        root_choice = self.choose_root(cleaned, description)
        vlog(f"root_choice decision: {root_choice}")
        if root_choice.lower() == "new":
            new_node = OntologyNode(cleaned, description)
            self.nodes.append(new_node)
            self.roots.append(new_node)
            vlog("created new root")
            return new_node
        else:
            return self.add_to_root(cleaned, description, root_choice)


    def choose_root(self, name, description):
        if not self.roots:
            return "NEW"
        candidates_text = "\n".join([f"{n.name}: {n.description}" for n in self.roots])
        prompt = f"""
You are organizing concepts into trees.

A new feature has been extracted:
Feature: {name}
Definition: {description}

Choose the best top-level concept it belongs under:
{candidates_text}

If none apply, return: NEW
Otherwise, return the name of the root node it belongs to.
"""
        decision = query_llm(prompt).strip()
        return decision if any(r.name == decision for r in self.roots) else "NEW"

    def add_to_root(self, cleaned, description, root_choice):
        new_node = OntologyNode(cleaned, description)

        # Find the root node by name
        try:
            root_node = next(n for n in self.roots if n.name == root_choice)
        except StopIteration:
            vlog(f"[ERROR] Root '{root_choice}' not found among roots.")
            return new_node  # Fallback: return the new node unconnected

        # Build candidate list for integration
        candidates = [n for n in self.nodes if n != new_node]
        candidates_text = "\n".join(f"{n.name}: {n.description}" for n in candidates)

        # Query LLM for integration decision
        prompt = f"""
You are organizing nodes in an ontology tree.

A new feature has been extracted:
Name: {cleaned}
Definition: {description}

Decide how to integrate the new node into the existing tree structure.
Return one of the following (with exact format):
- ALIAS_OF: <existing node name>
- CHILD_OF: <existing node name>
- PARENT_OF: <existing node name>
- NO_MATCH

Existing candidates:
    {candidates_text}
    """
        decision = query_llm(prompt).strip()

        vlog(f"add_to_root decision is {decision}")
        if decision.startswith("ALIAS_OF:"):
            target_name = decision.split(":", 1)[1].strip()
            target_node = next((n for n in self.nodes if n.name == target_name), None)
            if target_node:
                target_node.aliases.add(cleaned)
                vlog(f"'{cleaned}' added as ALIAS to '{target_node.name}'")
                return target_node

        elif decision.startswith("CHILD_OF:"):
            target_name = decision.split(":", 1)[1].strip()
            target_node = next((n for n in self.nodes if n.name == target_name), None)
            if target_node:
                new_node.parent = target_node
                target_node.children.append(new_node)
                self.nodes.append(new_node)
                vlog(f"'{new_node.name}' added as CHILD to '{target_node.name}'")
                return new_node

        elif decision.startswith("PARENT_OF:"):
            target_name = decision.split(":", 1)[1].strip()
            target_node = next((n for n in self.nodes if n.name == target_name), None)
            if target_node:
                # Reassign parent
                if target_node.parent:
                    target_node.parent.children.remove(target_node)
                    vlog(f"'{new_node.name}' added as PARENT to '{target_node.name}'")
                else:
                    self.roots.pop(target_node)
                    self.roots.append(new_node)
                    vlog(f"'{new_node.name}' replaced '{target_node.name}' as root")
                target_node.parent = new_node
                new_node.children.append(target_node)
                self.nodes.append(new_node)
                
                return new_node

        # If no match, default to adding as child to root_choice
        new_node.parent = root_node
        root_node.children.append(new_node)
        self.nodes.append(new_node)
        vlog(f"'{new_node.name}' added as CHILD to root '{root_node.name}' by default")
        return new_node


    def to_dict(self):
        return {
            "nodes": {
                n.name: {
                    "name": n.name,
                    "description": n.description,
                    "aliases": list(n.aliases),
                    "children": [c.name for c in n.children],
                    "parent": n.parent.name if n.parent else None,
                } for n in self.nodes
            },
            "roots": [r.name for r in self.roots],
            "review2name_score": dict(self.review2name_score),
        }

    def save(self, path: Path):
        dumpj(path, self.to_dict())
        vlog(f"[Ontology saved to {path}]")

    @classmethod
    def load(cls, path: Path):
        data = loadj(path)
        ontology = cls()

        # Step 1: Reconstruct all nodes
        name_to_node = {}
        for name, node_data in data["nodes"].items():
            node = OntologyNode(name=node_data["name"], description=node_data["description"])
            node.aliases = set(node_data["aliases"])
            name_to_node[name] = node
        ontology.nodes = list(name_to_node.values())

        # Step 2: Reconstruct tree structure (parent and children)
        for name, node_data in data["nodes"].items():
            node = name_to_node[name]
            parent_name = node_data["parent"]
            if parent_name:
                node.parent = name_to_node[parent_name]
            for child_name in node_data["children"]:
                node.children.append(name_to_node[child_name])

        # Step 3: Assign roots and review mapping
        ontology.roots = [name_to_node[rname] for rname in data.get("roots", [])]
        ontology.review2name_score = defaultdict(list, {
            k: v for k, v in data.get("review2name_score", {}).items()
        })

        vlog(f"[Ontology loaded from {path} with {len(ontology.nodes)} nodes]")
        return ontology


    def feature_hints(self, max_count=15):
        return sorted(set(n.name for n in self.nodes))[:max_count]

    def __repr__(self):
        def render_node(node, depth=0):
            indent = "    " * depth
            lines = [f"{indent}- {node.name}"]
            for child in node.children:
                lines.extend(render_node(child, depth + 1))
            return lines

        if not self.roots:
            return "[Ontology is empty]"

        lines = ["Ontology Tree:"]
        for root in self.roots:
            lines.extend(render_node(root))
        return "\n".join(lines)


### --- Feature + Sentiment Extraction --- ###

def clean_phrase(phrase):
    return phrase.lower().strip("* ").split(". ")[-1].strip()

def extract_feature_mentions(text, ontology=None, dset = None, model="openai"):
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

You are analyzing a {domain} review to extract key features that the user evaluated. These features must be **generalizable** — they should be applicable to many other {domain}s, not specific to just this one case.

Follow this two-step process:

Step 1: Identify distinct **general feature concepts** described in the review — what broad aspects did the user care about that are likely relevant across most {domain}s? These could relate to {cares} — avoid naming brand-specific, item-specific, or context-specific details.

Step 2: For each general concept, provide:
- A short, lowercase phrase that neutrally names the feature (e.g. "wait time", not "long wait time" or "15-minute delay")
- A one-sentence abstract definition that clearly explains what the feature refers to in general. Briefly describe what a positive and a negative score would mean for that feature (e.g., “short wait times are positive; long delays are negative”)
- A sentiment score between -1.0 and +1.0 indicating how the feature is portrayed in this specific review

If relevant, align your phrasing with known features: {hint_str}
Otherwise, invent a **general-purpose** phrase — avoid overly specific or one-off descriptions.

Be precise. Avoid using the same phrase for unrelated meanings.

Review:
\"\"\"{text.strip()}\"\"\"

Output format (one line per feature):
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

def build_ontology_by_reviews(args, reviews):

    feature_cache_path = Path(f'cache/{args.dset}_feature_score.json')
    ontology = Ontology()

    if feature_cache_path.exists():
        feature_data = loadj(feature_cache_path)

        for review_id, feature_scores in feature_data:
            for phrase, description, score in feature_scores:
                phrase = clean_phrase(phrase)
                ontology.add_or_update_node(review_id, phrase, description, score)

    else:
        feature_data = []
        vlog('ontology class initialized')

        count = 0
        for r in reviews:
            review_id, text = r["review_id"], r["text"]
            vlog("\n" + "="*60)
            vlog(f"Review: {text}")
            feature_scores = extract_feature_mentions(text, ontology, args.dset)
            vlog(f"Extracted Features: {feature_scores}")
            feature_data.append([review_id, feature_scores])

            for phrase, description, score in feature_scores:
                phrase = clean_phrase(phrase)
                ontology.add_or_update_node(review_id, phrase, description, score)

            vlog("\nOntology after processing this review:")
            if VERBOSE:
                print(ontology)

            dumpj(feature_cache_path, feature_data)

            # ppause('finished 1 review')

            count += 1
            if count == 50:
                check()

    output_path = Path("cache") / "ontology.json"
    ontology.save(output_path)
    print(f"Ontology saved to {output_path} with {len(ontology.nodes)} nodes.")
    return ontology

def build_ontology_by_users_and_items():
    pass