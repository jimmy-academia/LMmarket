# ontology.py
# from __future__ import annotations
import shutil, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as _csim

from utils import clean_phrase as _clean
from llm import query_llm

def _node_embed_text(name, description):
    """Stronger embedding signal: name + description."""
    name = (name or "").strip()
    desc = (description or "").strip()
    return f"{name} {desc}".strip() if desc else name


@dataclass
class OntologyNode:
    name: str
    description: str = ""
    aliases: Set[str] = field(default_factory=set)
    parent: Optional[str] = None           # keep simple: store parent by name
    children: List[str] = field(default_factory=list)  # child names

class Ontology:
    def __init__(
        self,
        batch_threshold = 10,
        max_depth = 4,
        edit_path = Path("cache/ontology_edit.txt"),
        backup_json = Path("cache/ontology_backup.json"),
        pause_message = "\n>>> Edit cache/ontology_edit.txt (add '(alias: target)' to lines as needed), save, then press Enter...",
        review_log_path = None
    ):
        self.MAX_DEPTH = max_depth
        self.batch_threshold = batch_threshold
        self.edit_path = edit_path
        self.backup_json = backup_json
        self.pause_message = pause_message
        self.review_log_path = review_log_path
        self.node_growth_log_path = Path("cache/node_growth.json")

        self.nodes: Dict[str, OntologyNode] = OrderedDict()
        self.review2node_id_score: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Growth tracking
        self._processed_reviews_in_session: Set[str] = set()
        self.node_growth_log: List[int] = []
        # haven't done continuable
        '''
        if self.node_growth_log_path.exists():
            try:
                content = self.node_growth_log_path.read_text(encoding="utf-8")
                if content:
                    self.node_growth_log = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                self.node_growth_log = [] # Start fresh if file is corrupt or empty
        '''

        # queue of brand-new features awaiting human action
        # name -> (definition, score)
        self._pending: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

        # embedding model & cache
        self._embed_model = SentenceTransformer("BAAI/bge-small-en")
        self._emb_cache: Dict[str, np.ndarray] = {}  # node_name -> normalized vector

    def _find_main_node(self, name):
        """Find the main node name for a given feature (either direct node or alias)."""
        # 1) Direct node match
        if name in self.nodes:
            return name
        
        # 2) Alias match
        for node_name, node in self.nodes.items():
            if name in node.aliases:
                return node_name
        
        return None

    def add_or_update_node(self, review_id, phrase, description, score):
        """
        Exact-match add/update with simple human-in-the-loop once threshold is hit.
        - If `phrase` matches an existing node name exactly → record the hit; return False.
        - Else queue it. If queue size reaches threshold:
            * write edit file: <new_feature> lines (with recommendations) then '===', then current structure
            * pause for human edits (expects '(alias: target)' added on top lines if desired)
            * read edited file and apply aliases / create nodes for any remaining
        Returns True if a new node was eventually created (after applying the edit file); False if it hit/aliased.
        """
        name = _clean(phrase)

        self._processed_reviews_in_session.add(review_id)
        
        # 1) Check if it's a known feature (node name or alias)
        main_node = self._find_main_node(name)
        if main_node:
            self.review2node_id_score[review_id].append((main_node, float(score)))
            return False

        # 2) Queue new feature
        if name not in self._pending:
            self._pending[name] = (description.strip(), float(score))

        # 3) If threshold reached -> write, pause, apply
        if len(self._pending) >= self.batch_threshold:
            # Log the number of reviews processed for this batch
            reviews_for_this_batch = len(self._processed_reviews_in_session)
            if reviews_for_this_batch > 0:
                self.node_growth_log.append(reviews_for_this_batch)
                with self.node_growth_log_path.open("w", encoding="utf-8") as f:
                    json.dump(self.node_growth_log, f)
                self._processed_reviews_in_session.clear()

            self._write_edit_file_with_recommendations()
            # backup current ontology json before applying edits
            self.save(self.backup_json)

            # Pause: `input()` is fine for CLI; guard for non-interactive envs.
            try:
                input(self.pause_message)
                pass
            except EOFError:
                # Non-interactive environment; proceed without pausing.
                pass

            self._apply_edit_file()   # apply aliases or create nodes
            self._pending.clear()

        # Decision (for current item) is unknown until after edit file is applied.
        # We conservatively return True if it's not an exact hit (it *may* become a node).
        # If you want accuracy, you can track changes inside _apply_edit_file and return accordingly.
        return name in self.nodes

    def get_node_depth(self, name):
        """Calculates the depth of a node (root is 0)."""
        depth = 0
        # Ensure the starting node exists in the ontology
        if name not in self.nodes:
            return -1 # Or raise an error, -1 indicates not found

        curr = self.nodes.get(name)
        while curr and curr.parent:
            depth += 1
            curr = self.nodes.get(curr.parent)
            # Safety break to prevent infinite loops on corrupted data
            if depth > 20: 
                return -2 # Indicates a potential cycle or excessive depth
        return depth

    def add_node(self, name, description="", parent_name=None):
        """
        Adds a new node to the ontology, optionally as a child of another node.
        If the node already exists, it does nothing.
        If the parent does not exist, it raises a KeyError.
        If adding the node would exceed MAX_DEPTH, it refuses and returns False.

        Args:
            name (str): The name of the new node.
            description (str, optional): The description for the new node. Defaults to "".
            parent_name (Optional[str], optional): The name of the parent node. If None, adds as a root. Defaults to None.

        Returns:
            bool: True if the node was added, False if it already existed or would exceed max depth.
        """
        name = _clean(name)
        if name in self.nodes:
            return False

        if parent_name:
            parent_name = _clean(parent_name)
            if parent_name not in self.nodes:
                raise KeyError(f"Parent node '{parent_name}' not found in ontology.")
            
            # Enforce MAX_DEPTH
            parent_depth = self.get_node_depth(parent_name)
            if parent_depth >= self.MAX_DEPTH - 1:
                print(f"    - WARNING: Cannot add '{name}'. Parent '{parent_name}' is at depth {parent_depth}, which is the maximum allowed depth.")
                return False
            
            self.nodes[name] = OntologyNode(name=name, description=description, parent=parent_name)
            self.nodes[parent_name].children.append(name)
        else: # It's a root node
            self.nodes[name] = OntologyNode(name=name, description=description)

        return True

    # ---------------------------
    # Minimal helpers to support the flow
    # ---------------------------
    def save(self, path):
        data = {
            name: {
                "description": n.description,
                "aliases": sorted(list(n.aliases)),
                "parent": n.parent,
                "children": list(n.children),
            } for name, n in self.nodes.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def __str__(self, node=None, print_depth=None):
        # simple tree print (no re-parenting logic here)
        if node is None:
            roots = [n for n in self.nodes.values() if n.parent is None]
            roots = sorted(roots, key=lambda x: x.name)
        else:
            roots = [node]

        out: List[str] = []
        def dfs(name, d):
            if print_depth is not None and d >= print_depth or d >= self.MAX_DEPTH - 1:
                return
            out.append("\t"*d + name)
            node = self.nodes[name]
            if d == self.MAX_DEPTH - 2:
                if node.children:
                    out.append("\t"*(d+1) + ", ".join(sorted(node.children)))
                return
            for c in sorted(node.children):
                dfs(c, d+1)
        for r in roots:
            dfs(r.name, 0)
        return "\n".join(out)

    # --- recommendation (very light-weight, name similarity only) ---
    def _recommend(self, feat, top_k=3, feat_desc=""):
        """
        Embedding-based nearest neighbors over existing node (name+description).
        Returns [(score, node_name)] sorted desc by cosine similarity.
        """
        if not self.nodes:
            return []

        # build query embedding
        q_text = _node_embed_text(feat, feat_desc)
        if not q_text:
            return []
        q_vec = self._embed_model.encode(q_text, normalize_embeddings=True)

        # score against cached/computed node embeddings
        scored: List[Tuple[float, str]] = []
        for name, node in self.nodes.items():
            # cache node emb
            if name not in self._emb_cache:
                n_text = _node_embed_text(node.name, node.description)
                self._emb_cache[name] = self._embed_model.encode(n_text, normalize_embeddings=True)
            s = float(_csim(q_vec.reshape(1, -1), self._emb_cache[name].reshape(1, -1))[0][0])
            scored.append((s, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


    # --- write edit file ---
    def _write_edit_file_with_recommendations(self):
        """
        Top block lines:
            <new_feature>\t<new_feature_description>
        You MAY add a third column (tab-separated):
            <new_feature>\t<new_feature_description>\t<alias_target_feature>
        Then a separator line '===', then the current structure (read-only for now).
        """
        lines: list[str] = []

        # Top block: write name + description, plus inline comment with recs
        for nf, (desc, _score) in self._pending.items():
            safe_desc = desc.replace("\t", " ").strip()
            # compute embedding recs
            recs = self._recommend(nf, top_k=3, feat_desc=safe_desc)
            if recs:
                rec_str = ", ".join(f"{name}({score:.2f})" for score, name in recs)
                lines.append(f"{nf}\t{safe_desc}    # rec: {rec_str}")
            else:
                lines.append(f"{nf}\t{safe_desc}")

        lines.append("===")

        # Structure block
        roots = [n for n in self.nodes.values() if n.parent is None]
        roots = sorted(roots, key=lambda x: x.name)

        def dfs(name, d):
            if d >= self.MAX_DEPTH - 1:
                return
            lines.append("\t"*d + name)
            node = self.nodes[name]
            if d == self.MAX_DEPTH - 2:
                if node.children:
                    lines.append("\t"*(d+1) + ", ".join(sorted(node.children)))
                return
            for c in sorted(node.children):
                dfs(c, d+1)

        for r in roots:
            dfs(r.name, 0)

        self.edit_path.parent.mkdir(parents=True, exist_ok=True)
        self.edit_path.write_text("\n".join(lines), encoding="utf-8")

    def _apply_edit_file(self):
        """
        Reads BOTH the top block and bottom block.
        - If '===' is present: top is new-features TSV; bottom is the structure.
        - If '===' is absent: treat the whole file as the structure (top empty).
        
        TOP block format (tab-separated):
            <feature>\t<description>[\t<alias_target>]
          Behavior:
            - If alias_target exists in the ontology after we rebuild the structure,
              add <feature> as an alias to that target and remove standalone node <feature> if created.
            - If alias_target is empty and <feature> does not appear in the tree block,
              create <feature> as a ROOT node with the given description (or update description if node exists).
        
        BOTTOM (structure) block:
          Tab-indented hierarchy. At the last printed depth (MAX_DEPTH-1), children can be on a single
          comma-joined line. No flags/parentheses here.
          
        Steps:
          1) Archive the edited file (timestamped copy).
          2) Parse top block into (name -> (description, alias_target)).
          3) Parse structure block and REBUILD the tree (create nodes as needed),
             preferring existing descriptions, falling back to top-block description, then empty.
          4) Apply alias directives from the top block:
             - Merge as alias into target; remove standalone node <feature> if present.
          5) For top-block features without alias targets and not present in the structure, ensure a ROOT node.
        """
        raw = self.edit_path.read_text(encoding="utf-8")

        # 1) Log the edit session instead of archiving
        if self.review_log_path:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            log_header = f"\n### Edit Session ({ts}) ###\n"
            log_content = f"# --- User Input (from {self.edit_path.name}) ---\n{raw}\n---\n"
            with self.review_log_path.open("a", encoding="utf-8") as f:
                f.write(log_header + log_content)

        # 2) Split file
        if "===" in raw:
            top, tree_text = raw.split("===", 1)
        else:
            top, tree_text = "", raw  # initial structure case or no top block

        # ---- Parse TOP block: format is `feature [(alias: target)] \t description` ----
        top_specs: dict[str, tuple[str, str]] = {}  # name -> (desc, alias_target or "")
        alias_pattern = re.compile(r"\((alias|aliases|is):\s*(.*?)\)", re.IGNORECASE)

        for ln in top.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            line_content = s.split("#", 1)[0].rstrip()
            parts = [p.strip() for p in line_content.split("\t", 1)]
            
            if not parts:
                continue

            feature_part = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            
            alias_tgt = ""
            feat_name = feature_part

            match = alias_pattern.search(feature_part)
            if match:
                # Extract feature name and alias target
                feat_name = feature_part[:match.start()].strip()
                alias_tgt = match.group(2).strip()

            if feat_name:
                cleaned_feat = _clean(feat_name)
                cleaned_alias = _clean(alias_tgt) if alias_tgt else ""
                top_specs[cleaned_feat] = (desc, cleaned_alias)

        # ---- Parse STRUCTURE block and rebuild the tree ----
        # Helper to expand tree text into (depth, name) items.
        def _parse_tree_block(tree_str):
            lines = [ln.rstrip("\r\n") for ln in tree_str.splitlines() if ln.strip()]
            parsed: list[tuple[int, str]] = []
            for ln in lines:
                depth = len(ln) - len(ln.lstrip("\t"))
                payload = ln[depth:].strip()
                # If we're at last printed depth, allow comma-joined children on the NEXT line
                if depth == self.MAX_DEPTH - 1 and "," in payload:
                    for item in [x.strip() for x in payload.split(",") if x.strip()]:
                        parsed.append((depth, _clean(item)))
                else:
                    parsed.append((depth, _clean(payload)))
            return parsed

        parsed = _parse_tree_block(tree_text)

        # Rebuild nodes (keep old descriptions if known; use top desc as fallback)
        old_nodes = self.nodes
        self.nodes = OrderedDict()
        stack: list[tuple[int, str]] = []  # (depth, name)

        def _desc_for(name):
            # prefer existing description, then top-block desc, else empty
            if name in old_nodes and old_nodes[name].description:
                return old_nodes[name].description
            if name in top_specs and top_specs[name][0]:
                return top_specs[name][0]
            return ""

        # Ensure node factory
        def _ensure_node(name):
            if name not in self.nodes:
                node = OntologyNode(name=name, description=_desc_for(name))
                # Preserve aliases from old nodes
                if name in old_nodes:
                    node.aliases = old_nodes[name].aliases.copy()
                self.nodes[name] = node

        # Build the tree
        for depth, nm in parsed:
            _ensure_node(nm)
            if depth > len(stack):
                raise ValueError(f"Invalid indentation jump at '{nm}': depth {depth} after stack depth {len(stack)}")
            while len(stack) > depth:
                stack.pop()
            parent = stack[-1][1] if stack else None
            if parent:
                # parent must exist
                _ensure_node(parent)
                # link
                self.nodes[nm].parent = parent
                if nm not in self.nodes[parent].children:
                    self.nodes[parent].children.append(nm)
            stack.append((depth, nm))

        # 4) Apply alias directives from the TOP block
        for name, (_desc, alias_tgt) in top_specs.items():
            if not alias_tgt:
                continue
            if alias_tgt in self.nodes:
                # If the feature exists as a standalone node, merge it into alias_tgt
                if name in self.nodes:
                    # detach from its parent
                    parent = self.nodes[name].parent
                    if parent and name in self.nodes[parent].children:
                        self.nodes[parent].children.remove(name)
                    # transfer aliases (include its own name)
                    self.nodes[alias_tgt].aliases.add(name)
                    self.nodes[alias_tgt].aliases.update(self.nodes[name].aliases)
                    # drop the node
                    del self.nodes[name]
                else:
                    # node didn't exist; just add alias to target
                    self.nodes[alias_tgt].aliases.add(name)
            else:
                # alias target not found in structure; create/ensure alias source as ROOT, then leave as-is
                if name not in self.nodes:
                    self.nodes[name] = OntologyNode(name=name, description=_desc_for(name))

        # 4.5) AI-powered placement for remaining top-level features
        unhandled_features = {
            name: (desc, alias_tgt)
            for name, (desc, alias_tgt) in top_specs.items()
            if not alias_tgt and name not in self.nodes
        }
        
        handled_by_ai = set()

        if unhandled_features:
            print(f"\n>>> Attempting to auto-place {len(unhandled_features)} new feature(s) using AI...")
            
            ontology_structure_level_1_and_2 = self.__str__(print_depth=2)

            for name, (desc, _) in unhandled_features.items():
                if not ontology_structure_level_1_and_2.strip():
                    print(f"    - Skipping '{name}', no Level 2 nodes exist for classification.")
                    continue

                # Stage 1: Classify to a Level 2 Node
                prompt1 = f"""You are a meticulous ontology architect. Your task is to classify a new feature into the most appropriate existing Level 2 category. This is a strict classification task.

**Hierarchy Explanation:**
The ontology is a tree. Indentation with tabs (\\t) shows parent-child relationships.
- Level 1 nodes (e.g., `environment`) have no indentation.
- Level 2 nodes (e.g., `ambiance`) are indented with one tab.

**New Feature:**
- **Name:** {name}
- **Description:** {desc}

**Existing Ontology Structure (Levels 1 & 2):**
{ontology_structure_level_1_and_2}

**Instructions:**
1.  First, analyze the new feature and identify the most relevant top-level root category (Level 1).
2.  Then, from the children within that root, you MUST select the single most specific and fitting Level 2 sub-category.
3.  Your answer must be the full name of exactly ONE Level 2 category from the structure provided (e.g., `flavor`, `staff_attitude`, `cleanliness`).
4.  You are not allowed to create a new category. You must choose from the list.

**Chosen Level 2 Category:**"""
                
                try:
                    chosen_level_2_node = _clean(query_llm(prompt1).strip())
                    # print("="*25)
                    # print(f"prompt1: {prompt1}")
                    print(f"    - Stage 1 for '{name}': Classified under Level 2 Node '{chosen_level_2_node}'")

                    if chosen_level_2_node not in self.nodes or self.nodes[chosen_level_2_node].parent is None:
                        print(f"    - SKIPPING: AI returned an invalid or non-Level 2 node ('{chosen_level_2_node}'.")
                        continue

                    # Stage 2: Place in Sub-tree
                    sub_tree_node = self.nodes[chosen_level_2_node]
                    sub_tree_structure = self.__str__(node=sub_tree_node)

                    prompt2 = f"""You are an expert ontology architect. Your primary task is to avoid creating duplicate concepts by identifying when a new feature is semantically similar to existing ones.

**Hierarchy Explanation:**
The sub-tree is shown with indentation. The root of this sub-tree, `{chosen_level_2_node}`, is a Level 2 node.
- Level 3 nodes are indented with one tab (\\t).
- Level 4 nodes are indented with two tabs (\\t\\t).

**Target Category:** {chosen_level_2_node}

**New Feature:**
- **Name:** {name}
- **Description:** {desc}

**Existing Sub-tree Structure:**
{sub_tree_structure}

**Instructions:**
Your PRIMARY goal is to identify if this new feature represents the same or very similar concept as an existing node.

1.  **FIRST PRIORITY - Check for semantic similarity:**
    - Does the new feature describe the same underlying concept, quality, or aspect as any existing node?
    - Consider synonyms, different wordings, or slightly different perspectives of the same concept.
    - Examples of what should be aliases:
      * "fast service" vs "quick service" vs "speedy service"
      * "spicy food" vs "hot food" (in spiciness context) vs "peppery"
      * "clean tables" vs "tidy dining area" vs "spotless surfaces"
    - If you find a semantically similar node, respond with: `ALIAS: <existing_node_name>`

2.  **SECOND PRIORITY - Only if NO similar concept exists:**
    - If the new feature represents a genuinely NEW concept not covered by existing nodes, then create a new node.
    - Place it at the deepest appropriate level (ideally Level 4 under a Level 3 parent).
    - If you find a suitable Level 3 parent, respond with: `PARENT: <level_3_parent_name>`
    - If no Level 3 parent fits, make it a Level 3 node: `PARENT: {chosen_level_2_node}`

**Decision Rules:**
- PREFER aliases over new nodes when concepts are similar (even if not identical)
- Only create new nodes for genuinely distinct concepts

Provide your decision in ONE format with no extra text:
- `ALIAS: <existing_node_name>` (if semantically similar)
- `PARENT: <existing_node_name>` (if genuinely new concept)

**Decision:**"""
                    
                    decision = query_llm(prompt2).strip()
                    # print(f"prompt2: {prompt2}")
                    print(f"    - Stage 2 for '{name}': Decision is '{decision}'")
                    # print("="*25)

                    if decision.startswith("PARENT:"):
                        parent_name = _clean(decision.split(":", 1)[1].strip())
                        if parent_name in self.nodes and self.get_node_depth(parent_name) >= self.MAX_DEPTH - 2:
                            self.add_node(name=name, description=desc, parent_name=parent_name)
                            handled_by_ai.add(name)
                            print(f"    - SUCCESS: Placed '{name}' as a child of '{parent_name}'.")
                        else:
                            print(f"    - FAILED: AI chose non-existent parent '{parent_name}' or parent of improper depth.")

                    elif decision.startswith("ALIAS:"):
                        target_name = _clean(decision.split(":", 1)[1].strip())
                        if target_name in self.nodes:
                            self.nodes[target_name].aliases.add(name)
                            handled_by_ai.add(name)
                            print(f"    - SUCCESS: Added '{name}' as an alias to '{target_name}'.")
                        else:
                            print(f"    - FAILED: AI chose non-existent alias target '{target_name}'.")

                except Exception as e:
                    print(f"    - ERROR: AI placement for '{name}' failed: {e}")

        # 5) Ensure top-block features without alias targets are present (if not placed in structure)
        for name, (desc, alias_tgt) in top_specs.items():
            if alias_tgt or name in handled_by_ai:
                continue
            if name not in self.nodes:
                # self.nodes[name] = OntologyNode(name=name, description=desc or _desc_for(name))
                pass
            else:
                # update description if empty
                if not self.nodes[name].description and desc:
                    self.nodes[name].description = desc

    def flush_pending_features(self):
        # Log the number of reviews processed for this batch
        reviews_for_this_batch = len(self._processed_reviews_in_session)
        if reviews_for_this_batch > 0:
            self.node_growth_log.append(reviews_for_this_batch)
            with self.node_growth_log_path.open("w", encoding="utf-8") as f:
                json.dump(self.node_growth_log, f)
        self._processed_reviews_in_session.clear()

        self._write_edit_file_with_recommendations()

        # backup current ontology json before applying edits
        self.save(self.backup_json)
        self._apply_edit_file()   # apply aliases or create nodes
        self._pending.clear()
