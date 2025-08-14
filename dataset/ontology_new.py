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

def _node_embed_text(name: str, description: str) -> str:
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
        batch_threshold: int = 10,
        max_depth: int = 3,
        edit_path: Path = Path("cache/ontology_edit.txt"),
        backup_json: Path = Path("cache/ontology_backup.json"),
        pause_message: str = "\n>>> Edit cache/ontology_edit.txt (add '(alias: target)' to lines as needed), save, then press Enter..."
    ):
        self.MAX_DEPTH = max_depth
        self.batch_threshold = batch_threshold
        self.edit_path = edit_path
        self.backup_json = backup_json
        self.pause_message = pause_message

        self.nodes: Dict[str, OntologyNode] = OrderedDict()
        self.review2node_id_score: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # queue of brand-new features awaiting human action
        # name -> (definition, score)
        self._pending: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

        # embedding model & cache
        self._embed_model = SentenceTransformer("BAAI/bge-small-en")
        self._emb_cache: Dict[str, np.ndarray] = {}  # node_name -> normalized vector


    def add_or_update_node(self, review_id: str, phrase: str, description: str, score: float) -> bool:
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

        # 1) Exact hit?
        if name in self.nodes:
            self.review2node_id_score[review_id].append((name, float(score)))
            return False

        # 2) Queue new feature
        if name not in self._pending:
            self._pending[name] = (description.strip(), float(score))

        # 3) If threshold reached -> write, pause, apply
        if len(self._pending) >= self.batch_threshold:
            self._write_edit_file_with_recommendations()
            # backup current ontology json before applying edits
            self.save(self.backup_json)

            # Pause: `input()` is fine for CLI; guard for non-interactive envs.
            try:
                input(self.pause_message)
            except EOFError:
                # Non-interactive environment; proceed without pausing.
                pass

            self._apply_edit_file()   # apply aliases or create nodes
            self._pending.clear()

        # Decision (for current item) is unknown until after edit file is applied.
        # We conservatively return True if it's not an exact hit (it *may* become a node).
        # If you want accuracy, you can track changes inside _apply_edit_file and return accordingly.
        return name in self.nodes

    # ---------------------------
    # Minimal helpers to support the flow
    # ---------------------------
    def save(self, path: Path) -> None:
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

    def __str__(self) -> str:
        # simple tree print (no re-parenting logic here)
        roots = [n for n in self.nodes.values() if n.parent is None]
        roots = sorted(roots, key=lambda x: x.name)
        out: List[str] = []
        def dfs(name: str, d: int):
            out.append("  "*d + name)
            node = self.nodes[name]
            if d >= self.MAX_DEPTH - 1:
                if node.children:
                    out.append("  "*(d+1) + ", ".join(sorted(node.children)))
                return
            for c in sorted(node.children):
                dfs(c, d+1)
        for r in roots:
            dfs(r.name, 0)
        return "\n".join(out)

    # --- recommendation (very light-weight, name similarity only) ---
    def _recommend(self, feat: str, top_k: int = 3, feat_desc: str = "") -> List[Tuple[float, str]]:
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
    def _write_edit_file_with_recommendations(self) -> None:
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

        def dfs(name: str, d: int):
            lines.append("\t"*d + name)
            node = self.nodes[name]
            if d >= self.MAX_DEPTH - 1:
                if node.children:
                    lines.append("\t"*(d+1) + ", ".join(sorted(node.children)))
                return
            for c in sorted(node.children):
                dfs(c, d+1)

        for r in roots:
            dfs(r.name, 0)

        self.edit_path.parent.mkdir(parents=True, exist_ok=True)
        self.edit_path.write_text("\n".join(lines), encoding="utf-8")

    def _apply_edit_file(self) -> None:
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

        # 1) Archive
        ts = time.strftime("%Y%m%d_%H%M%S")
        arc_dir = self.edit_path.parent / "edits_archive"
        arc_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.edit_path, arc_dir / f"{self.edit_path.stem}_{ts}{self.edit_path.suffix}")

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
        def _parse_tree_block(tree_str: str) -> list[tuple[int, str]]:
            lines = [ln.rstrip("\r\n") for ln in tree_str.splitlines() if ln.strip()]
            parsed: list[tuple[int, str]] = []
            for ln in lines:
                depth = len(ln) - len(ln.lstrip("\t"))
                payload = ln[depth:].strip()
                # If we're at last printed depth, allow comma-joined children on the NEXT line
                if depth == self.MAX_DEPTH - 1 and "," in payload:
                    for item in [x.strip() for x in payload.split(",") if x.strip()]:
                        parsed.append((depth + 1, _clean(item)))
                else:
                    parsed.append((depth, _clean(payload)))
            return parsed

        parsed = _parse_tree_block(tree_text)

        # Rebuild nodes (keep old descriptions if known; use top desc as fallback)
        old_nodes = self.nodes
        self.nodes = OrderedDict()
        stack: list[tuple[int, str]] = []  # (depth, name)

        def _desc_for(name: str) -> str:
            # prefer existing description, then top-block desc, else empty
            if name in old_nodes and old_nodes[name].description:
                return old_nodes[name].description
            if name in top_specs and top_specs[name][0]:
                return top_specs[name][0]
            return ""

        # Ensure node factory
        def _ensure_node(name: str):
            if name not in self.nodes:
                self.nodes[name] = OntologyNode(name=name, description=_desc_for(name))

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
            
            root_nodes = [n.name for n in self.nodes.values() if n.parent is None]
            
            for name, (desc, _) in unhandled_features.items():
                if not root_nodes:
                    print(f"    - Skipping '{name}', no root nodes exist for classification.")
                    continue

                # Stage 1: Classify Root
                prompt1 = f"""You are an expert ontology architect. Your task is to categorize a new feature under the most appropriate existing root category.

**New Feature:**
- **Name:** {name}
- **Description:** {desc}

**Existing Root Categories:**
{", ".join(root_nodes)}

**Instructions:**
1. Analyze the new feature's name and description.
2. Compare it against the meaning of each root category.
3. Choose the single best-fitting root category from the list.
4. If none of the categories are a good fit, respond with "NEW_ROOT".
5. Your answer must be **only** the name of the chosen category or "NEW_ROOT". Do not add any explanation or extra text.

**Chosen Category:**"""
                
                try:
                    chosen_root = query_llm(prompt1).strip()
                    print("prompt1:", prompt1)
                    print(f"    - Stage 1 for '{name}': Classified under '{chosen_root}'")

                    if chosen_root == "NEW_ROOT" or chosen_root not in self.nodes:
                        continue # Will be handled by step 5

                    # Stage 2: Place in Sub-tree
                    sub_tree_list = []
                    def build_subtree_str(node_name: str, d: int):
                        sub_tree_list.append("  "*d + node_name)
                        node = self.nodes[node_name]
                        if d >= self.MAX_DEPTH - 1:
                            return
                        for c in sorted(node.children):
                            build_subtree_str(c, d+1)
                    
                    build_subtree_str(chosen_root, 0)
                    sub_tree_structure = "\n".join(sub_tree_list)

                    prompt2 = f"""You are an expert ontology architect. Your task is to place a new feature into the correct position within a given category's sub-tree.

**Category:** {chosen_root}

**New Feature:**
- **Name:** {name}
- **Description:** {desc}

**Existing Sub-tree Structure:**
{sub_tree_structure}

**Instructions:**
Analyze the new feature and the existing sub-tree to determine the best relationship. Your response must be one of the following three formats, with no extra text or explanation:
1. **`PARENT: <existing_node_name>`**
2. **`ALIAS: <existing_node_name>`**
3. **`NEW_ROOT`**

**Decision:**"""
                    
                    decision = query_llm(prompt2).strip()
                    print("prompt1:", prompt1)
                    print(f"    - Stage 2 for '{name}': Decision is '{decision}'")

                    if decision.startswith("PARENT:"):
                        parent_name = _clean(decision.split(":", 1)[1].strip())
                        if parent_name in self.nodes:
                            _ensure_node(name)
                            self.nodes[name].parent = parent_name
                            if name not in self.nodes[parent_name].children:
                                self.nodes[parent_name].children.append(name)
                            handled_by_ai.add(name)
                            print(f"    - SUCCESS: Placed '{name}' as a child of '{parent_name}'.")
                    
                    elif decision.startswith("ALIAS:"):
                        target_name = _clean(decision.split(":", 1)[1].strip())
                        if target_name in self.nodes:
                            self.nodes[target_name].aliases.add(name)
                            handled_by_ai.add(name)
                            print(f"    - SUCCESS: Added '{name}' as an alias to '{target_name}'.")

                except Exception as e:
                    print(f"    - ERROR: AI placement for '{name}' failed: {e}")

        # 5) Ensure top-block features without alias targets are present (if not placed in structure)
        for name, (desc, alias_tgt) in top_specs.items():
            if alias_tgt or name in handled_by_ai:
                continue
            if name not in self.nodes:
                self.nodes[name] = OntologyNode(name=name, description=desc or _desc_for(name))
            else:
                # update description if empty
                if not self.nodes[name].description and desc:
                    self.nodes[name].description = desc
