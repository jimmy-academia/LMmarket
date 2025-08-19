from typing import List, Tuple, Dict, Optional
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as csim
from utils import readf, vprint, pause_if
from functools import partial
from tqdm import tqdm
import os
import re
os.environ['TOKENIZERS_PARALLELISM'] = "false"

from llm import query_llm
from debug import check

### --- Flags --- ###
VERBOSE = True
PAUSE = False
vlog = partial(vprint, flag=VERBOSE)
ppause = partial(pause_if, flag=PAUSE)

### --- Embedding Setup --- ###

embed_model = SentenceTransformer("BAAI/bge-small-en")

def embed(text: str) -> np.ndarray:
    return embed_model.encode(text, normalize_embeddings=True)

def cosine(a, b):
    return float(csim(a.reshape(1, -1), b.reshape(1, -1))[0][0])

def clean_phrase(phrase: str) -> str:
    return phrase.lower().strip("* ").split(". ")[-1].strip().replace("*", "").strip().replace(" ", "_")

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
        parent_str = f"\nparent='{self.parent.name}'" if self.parent else ""
        child_str = ""
        if self.children:
            child_str = "\nchildren=[" + ", ".join([child for child in self.children]) + "]"

        return (
            f"OntologyNode(name='{self.name}',"
            f"\ndescription='{self.description}',"
            f"\naliases={self.aliases}"
            f"{parent_str}"
            f"{child_str})\n"
        )

class Ontology:
    def __init__(self):
        self.MAX_DEPTH = 2
        self.nodes = {}
        self.review2node_id_score = defaultdict(list)

    def feature_hints(self, text, max_count=60):
        relevant_nodes = self.search_top_ten(text, top_k=max_count)
        names = [n.name for _, n in relevant_nodes]
        return names
    
    def search_top_ten(self, query_text, top_k=10):
        query_vec = embed(query_text)
        scored = []
        for node_id, node in self.nodes.items():
            sim = cosine(query_vec, node.embedding)
            scored.append((sim, node))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [node for node in scored[:top_k]]

    def get_node_depth(self, node: OntologyNode) -> int:
        depth = 0
        current = node 
        while current.parent is not None:
            print(f"get_node_depth: {current.name} -> {current.parent.name}")
            depth += 1
            current = current.parent
        return depth
    
    def root_max_depth(self, root: OntologyNode) -> int:
        """回傳以 root 為根的子樹，其所有節點(以整棵樹根為基準)深度的最大值。"""
        def dfs(n: OntologyNode) -> int:
            cur = self.get_node_depth(n)
            if not n.children:
                return cur
            return max(dfs(c) for c in n.children.values())
        return dfs(root)
    
    def get_root(self, node):
        while node.parent:
            node = node.parent
        return node

    def added_as_root(self, new_id: str, review_id: str, score: float, description: str) -> bool:
        """
        新增一個全新的 ROOT node。
        """
        if new_id in self.nodes:
            vlog(f"'{new_id}' already exists, not added as ROOT.")
            return False
        
        new_node = OntologyNode(new_id, new_id, description, embed(new_id))
        self.nodes[new_id] = new_node
        self.review2node_id_score[review_id].append((new_id, score))
        vlog(f"'{new_id}' added as ROOT node")

    def added_as_alias(self, target: str, alias: str, review_id: str, score: float) -> bool:
        """
        將一個 alias 加入到已存在的 node 中。
        """
        if target not in self.nodes:
            return False
        self.nodes[target].update(alias)
        self.review2node_id_score[review_id].append((target, score))
        vlog(f"'{alias}' added as ALIAS to '{target}'")
        return False

    def added_as_child(self, parent: str, new_id: str, review_id: str, score: float, description: str) -> bool:
        """
        將 new_id 設為 parent node 的 child。
        """
        if parent not in self.nodes:
            vlog(f"'{new_id}' not added as CHILD to '{parent}': parent node does not exist. Skip.")
            return False
        if new_id in self.nodes:
            vlog(f"'{new_id}' not added as CHILD to '{parent}': {new_id} node already exists. Skip.")
            return False

        if self.get_node_depth(self.nodes[parent]) >= self.MAX_DEPTH:
            vlog(f"'{new_id}' not added as CHILD to '{parent}': max depth reached ({self.MAX_DEPTH}). Added as alias instead.")
            return self.added_as_alias(parent, new_id, review_id, score)

        new_id = clean_phrase(new_id)
        new_node = OntologyNode(new_id, new_id, description, embed(new_id))
        new_node.parent = self.nodes[parent]
        self.nodes[parent].children[new_id] = new_node
        self.nodes[new_id] = new_node
        self.review2node_id_score[review_id].append((new_id, score))
        vlog(f"'{new_id}' added as CHILD to '{parent}'")
        return True

    def added_as_parent(self, new_id: str, child: str, review_id: str, score: float, description: str) -> bool:
        """
        將 new_id 設為 child node 的 parent。
        """
        if child not in self.nodes:
            vlog(f"'{new_id}' not added as PARENT to '{child}': child node does not exist. Skip.")
            return False
        if new_id in self.nodes:
            vlog(f"'{new_id}' not added as PARENT to '{child}': {new_id} node already exists. Skip.")
            return False

        root_node = self.get_root(self.nodes[child])
        if self.root_max_depth(root_node) >= self.MAX_DEPTH:
            vlog(f"'{new_id}' not added as PARENT to '{child}': max depth reached ({self.MAX_DEPTH}). Added as alias instead.")
            return self.added_as_alias(child, new_id, review_id, score)

        new_node = OntologyNode(new_id, new_id, description, embed(new_id))
        new_node.children[child] = self.nodes[child]
        self.nodes[child].parent = new_node
        self.nodes[new_id] = new_node
        self.review2node_id_score[review_id].append((new_id, score))
        vlog(f"'{new_id}' added as PARENT to '{child}'")
        return True

    def add_or_update_node(self, review_id, phrase, description, score) -> bool:
        """
        回傳 True = 新增了一個 node (包括 CHILD / PARENT 分支)，
        False = 只是加了 alias 或回傳到既有 node。
        """
        cleaned = clean_phrase(phrase)
        # 1) alias match
        for node in self.nodes.values():
            if cleaned in node.aliases:
                self.review2node_id_score[review_id].append((node.node_id, score))
                return False

        # 2) LLM 判斷
        top_candidates = self.search_top_ten(cleaned, top_k=20)
        if top_candidates:
            candidates_text = "\n".join(f"{n.name}: {n.description}" for _, n in top_candidates)
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

Only return the decision, no explanations or extra text.
"""
            decision = query_llm(prompt).strip()
            vlog(f"LLM decision for '{cleaned}': {decision}")

            if decision.startswith("ALIAS:"):
                target = decision.split(":",1)[1].strip()
                return self.added_as_alias(target, cleaned, review_id, score)

            elif decision.startswith("CHILD:"):
                parent = decision.split(":",1)[1].strip()
                return self.added_as_child(parent, cleaned, review_id, score, description)

            elif decision.startswith("PARENT:"):
                child = decision.split(":",1)[1].strip()
                return self.added_as_parent(cleaned, child, review_id, score, description)

        # 3) 全新 node
        #node_id = cleaned
        #self.nodes[node_id] = OntologyNode(node_id, cleaned, description, embed(cleaned))
        #self.review2node_id_score[review_id].append((node_id, score))
        #return True

    def save_json(self, path: Path):
        json_dict = {
            name: {
                "name": n.name,
                "description": n.description,
                "aliases": list(n.aliases),
                "children": list(n.children.keys()),
                "parent": ("None" if n.parent == None else n.parent.name),
            } for name, n in self.nodes.items()
        }
        with open(path, "w") as f:
            json.dump(json_dict, f, indent=2)
    
    def save_txt(self, path: Path, new_features: List[str] = None):
        json_dict = {
            name: {
                "name": n.name,
                "children": list(n.children.keys()),
                "parent": ("None" if n.parent == None else n.parent.name),
            } for name, n in self.nodes.items()
        }

        def norm_parent(p):
            if isinstance(p, str) and p == "None":
                return None
            return p

        # Keep deterministic ordering (as loaded from JSON)
        if not isinstance(json_dict, OrderedDict):
            json_dict = OrderedDict(json_dict.items())

        parents = {k: norm_parent(v.get("parent")) for k, v in json_dict.items()}
        roots = [k for k in json_dict if parents.get(k) is None]

        lines = []

        def get_all_children(node_id: str) -> list:
            """取得某個節點的所有子節點名稱"""
            if node_id not in json_dict:
                return []
            return json_dict[node_id].get("children", [])

        def add_node_to_lines(node_id: str):
            """將節點加入輸出列表，處理新特徵標記"""
            node_name = json_dict[node_id]["name"]
            if new_features and node_name in new_features:
                return f"{node_name}*"
            return node_name

        def dfs(node_id: str, depth: int, visited: set):
            if node_id in visited:
                lines.append("\t"*depth + f"{json_dict[node_id]['name']} (cycle detected)")
                return
            
            visited.add(node_id)
            
            # 處理當前層級節點 (不包括 leaf)
            lines.append("\t"*depth + add_node_to_lines(node_id))

            if depth < self.MAX_DEPTH - 1:
                # 遞迴處理子節點
                children = get_all_children(node_id)
                for child in children:
                    if child not in json_dict:
                        lines.append("\t"*(depth+1) + f"{child} (missing)")
                    else:
                        dfs(child, depth+1, visited)
            else:
                # 到達 max_depth - 1，將所有子節點以逗號分隔輸出
                children = get_all_children(node_id)
                if children:
                    child_names = [add_node_to_lines(child) for child in children if child in json_dict]
                    if child_names:
                        lines.append("\t"*(depth+1) + ", ".join(child_names))
            
            # visited.remove(node_id)  # 允許重複 node 在不同分支中出現

        # 從根節點開始遍歷
        for root in roots:
            dfs(root, 0, set())

        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def read_txt(self, path: Path) -> bool:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            vlog(f"read_txt: cannot read file: {e}")
            return False

        # Parse lines -> (depth, name, remove_flag)
        parsed = []
        line_no = 0

        def split_flags(name: str) -> Tuple[str, bool, Optional[str]]:
            s = name.strip()
            low = s.lower()
            # remove
            if low.endswith("(remove)"):
                base = s[: -len("(remove)")].rstrip()
                return base, True, None
            # rename
            m = re.search(r"\(rename\s*[:=]\s*(.+?)\)\s*$", s, flags=re.IGNORECASE)
            if m:
                base = s[: m.start()].rstrip()
                new_name = m.group(1).strip()
                return base, False, new_name
            return s, False, None

        for raw in text.splitlines():
            line_no += 1
            line = raw.rstrip("\n\r")
            if not line.strip():
                continue  # ignore empty lines

            # Count leading tabs
            i = 0
            while i < len(line) and line[i] == "\t":
                i += 1
            depth = i
            if depth < self.MAX_DEPTH:
                # 處理第一、二層節點
                name_raw = line[i:].replace("*", "").strip()
                if not name_raw:
                    vlog(f"read_txt: empty name at line {line_no}")
                    return False

                name, rm, rename_to = split_flags(name_raw)
                parsed.append((depth, name, rm, rename_to))
            else:
                # 處理第三層（逗號分隔）節點
                items = [item.strip() for item in line[i:].split(",")]
                for item in items:
                    if not item:  # 跳過空項目
                        continue
                    name_raw = item.replace("*", "").strip()
                    if not name_raw:
                        continue
                    
                    name, rm, rename_to = split_flags(name_raw)
                    parsed.append((depth, name, rm, rename_to))

        if not parsed:
            vlog("read_txt: file is empty after parsing.")
            return False

        # 先依「名稱（不含 remove 標記）」檢查重複
        names_in_file = [name for _, name, _, _ in parsed]
        seen, dups = set(), set()
        for n in names_in_file:
            if n in seen:
                dups.add(n)
            seen.add(n)
        if dups:
            vlog(f"read_txt: duplicated node(s) in file: {sorted(dups)}")
            return False

        # 展開 remove 子樹：遇到 (remove) 就把該節點與其後續更深的子孫一起標記為刪除
        removed = set()
        kept_lines = []
        i = 0
        while i < len(parsed):
            depth, name, rm, rename_to = parsed[i]

            if rm and rename_to:
                vlog(f"read_txt: line renames and removes the same node '{name}' — not allowed.")
                return False

            if rm:
                removed.add(name)
                i += 1
                # 收掉整個子樹
                while i < len(parsed) and parsed[i][0] > depth:
                    removed.add(parsed[i][1])
                    i += 1
            else:
                kept_lines.append((depth, name, rename_to))
                i += 1

        node_names = set(self.nodes.keys())
        kept_names = [n for _, n, _ in kept_lines]

        # Unknown 檢查（保留與刪除的節點都必須存在於 ontology）
        unknown_all = (set(kept_names) | removed) - node_names
        if unknown_all:
            vlog(f"read_txt: unknown node(s) in file (kept/removed): {sorted(unknown_all)}")
            return False

        # 缺漏檢查：未出現在檔案、且也沒被標記 remove 的節點 => 錯誤
        missing = node_names - set(kept_names) - removed
        if missing:
            vlog(f"read_txt: nodes missing from file (and not marked remove): {sorted(missing)}")
            return False

        # 僅對「保留的行」檢查縮排跳躍與父子鏈
        parent_of_new = {name: None for name in kept_names}
        stack = []  # (depth, name)

        for depth, name, _ in kept_lines:
            if depth > len(stack):
                vlog(
                    f"read_txt: invalid indentation jump for '{name}' "
                    f"(depth {depth} after stack depth {len(stack)})."
                )
                return False

            while len(stack) > depth:
                stack.pop()

            if depth > 0:
                if not stack:
                    vlog(f"read_txt: no parent for '{name}' at depth {depth}.")
                    return False
                parent_of_new[name] = stack[-1][1]

            stack.append((depth, name))


        # 收集改名請求（old -> new）
        rename_map = {}
        for _, old, new in kept_lines:
            if new:
                old_clean = clean_phrase(old)
                new_clean = clean_phrase(new)
                rename_map[old_clean] = new_clean

        # 改名衝突檢查
        # 1. 目標重複
        targets = [t for t in rename_map.values() if t not in (None, "")]
        dups_targets = {x for x in targets if targets.count(x) > 1}
        if dups_targets:
            vlog(f"read_txt: duplicate rename targets: {sorted(dups_targets)}")
            return False
        
        # 2. 目標與現有/未刪除節點衝突（允許 no-op: old==new）
        for old, new in rename_map.items():
            if old == new:
                continue
            if (new in self.nodes) and (new != old):
                vlog(f"read_txt: rename target '{new}' already exists.")
                return False
            if new in removed:
                vlog(f"read_txt: rename target '{new}' is being removed.")
                return False

        # 通過驗證：套用更新（原子性）
        # 先記錄舊父節點用於 logging
        old_parent_name = {
            n.name: (n.parent.name if n.parent else None) for n in self.nodes.values()
        }

        # 移除節點
        if removed:
            vlog(f"read_txt: removed {len(removed)} node(s): {sorted(removed)}")
        try:
            for r in removed:
                self.nodes.pop(r, None)
        except Exception as e:
            vlog(f"read_txt: failed to remove nodes: {e}")
            return False
        
        # 改名節點
        try:
            for old, new in rename_map.items():
                if old == new:
                    continue  # no-op
                if old not in self.nodes:
                    vlog(f"read_txt: rename source '{old}' not found (maybe removed?)")
                    return False
                node = self.nodes.pop(old)
                #node.aliases.add(old)   # 舊名保留成 alias (optional)
                node.name = new
                node.node_id = new 
                self.nodes[new] = node
        except Exception as e:
            vlog(f"read_txt: failed to apply renames: {e}")
            return False


        # 重新連線（先清空）
        for node in self.nodes.values():
            node.parent = None
            node.children.clear()

        def _m(name: Optional[str]) -> Optional[str]:
            if name is None:
                return None
            return rename_map.get(name, name)
        
        # 重建父子
        try:
            for child_name, p_name in parent_of_new.items():
                child_name = _m(child_name)
                p_name = _m(p_name)

                child = self.nodes[child_name]
                if p_name is None:
                    child.parent = None
                else:
                    parent = self.nodes[p_name]
                    child.parent = parent
                    parent.children[child_name] = child
        except Exception as e:
            vlog(f"read_txt: failed to apply relationships: {e}")
            return False

        # 移動紀錄：舊父 != 新父
        for name in kept_names:
            old_p = old_parent_name.get(name, None)
            new_p = parent_of_new.get(name, None)
            if old_p != new_p:
                vlog(f"read_txt: change parent node for '{name}': {old_p} -> {new_p}")

        return True

def extract_feature_mentions(text: str, ontology: Ontology = None, dset = None, model="openai") -> List[Tuple[str, float]]:
    hint_str = ', '.join(ontology.feature_hints(text)) if ontology else []

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

If relevant, align your phrasing with "Known Features". Otherwise, invent a reasonable new phrase.
### Known Features
{hint_str}

### Review
{text.strip()}

### Output format:
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

def human_in_the_loop_update(new_since_refine: int, new_features: List[str], update_cnt: int, ontology: Ontology = None) -> None:
    ontology.save_txt(Path(f"cache/ontology_human_in_the_loop_{update_cnt}.txt"), new_features)
    print(f"\n>>> 已新增 {new_since_refine} 個新 feature：{new_features}\n>>> 請打開 Ontology {update_cnt} 進行微調（標記 * 為新增 feature），完成後按 Enter 繼續...")
    input()
    valid = ontology.read_txt(Path(f"cache/ontology_human_in_the_loop_{update_cnt}.txt"))
    while valid == False:
        print(f"\n>>> 微調出錯請重試，完成後按 Enter 繼續...")
        input()
        valid = ontology.read_txt(Path(f"cache/ontology_human_in_the_loop_{update_cnt}.txt"))

### --- Main Ontology Building Function --- ###
def build_ontology_by_reviews(args, reviews: List[Dict], K: int = 10) -> Ontology:
    review_cnt = len(reviews)

    node_cnt = []
    new_since_refine = 0
    new_features = []
    update_cnt = 0

    ontology = Ontology()
    # Food Quality 相關特徵
    ontology.added_as_root("food quality", "root", 0, "Overall quality assessment of foods")
    ontology.added_as_child("food quality", "freshness", "child", 0, "Freshness of ingredients and prepared items")
    ontology.added_as_child("food quality", "flavor", "child", 0, "Taste and overall flavor profile")
    ontology.added_as_child("food quality", "temperature", "child", 0, "Appropriate serving temperature")
    ontology.added_as_child("food quality", "texture", "child", 0, "Food texture and consistency")
    ontology.added_as_child("food quality", "presentation", "child", 0, "Visual appearance and plating")

    # Price 相關特徵
    ontology.added_as_root("price", "root", 0, "All pricing and value-related aspects")
    ontology.added_as_child("price", "value", "child", 0, "Price to quality ratio")
    ontology.added_as_child("price", "portion value", "child", 0, "Amount of food for the price")
    ontology.added_as_child("price", "pricing policy", "child", 0, "Pricing structure and additional charges")
    ontology.added_as_child("price", "payment options", "child", 0, "Available payment methods")

    # Environment 相關特徵
    ontology.added_as_root("environment", "root", 0, "Physical and ambient characteristics of the establishment")
    ontology.added_as_child("environment", "cleanliness", "child", 0, "Overall cleanliness and hygiene")
    ontology.added_as_child("environment", "ambiance", "child", 0, "Atmosphere and mood of the place")
    ontology.added_as_child("environment", "noise level", "child", 0, "Sound level and acoustics")
    ontology.added_as_child("environment", "lighting", "child", 0, "Quality and appropriateness of lighting")

    # Service 相關特徵
    ontology.added_as_root("service", "root", 0, "All service-related experiences and interactions")
    ontology.added_as_child("service", "staff attitude", "child", 0, "Staff friendliness and professionalism")
    ontology.added_as_child("service", "responsiveness", "child", 0, "Speed and quality of service response")
    ontology.added_as_child("service", "accuracy", "child", 0, "Accuracy of orders and service")
    ontology.added_as_child("service", "knowledge", "child", 0, "Staff knowledge about menu and service")

    # Variety 相關特徵
    ontology.added_as_root("variety", "root", 0, "Range and diversity of offerings")
    ontology.added_as_child("variety", "menu options", "child", 0, "Diversity of menu items")
    ontology.added_as_child("variety", "dietary options", "child", 0, "Availability of different dietary choices")
    ontology.added_as_child("variety", "beverage selection", "child", 0, "Range of drink options")
    ontology.added_as_child("variety", "special items", "child", 0, "Unique or special menu items")

    # Convenience 相關特徵
    ontology.added_as_root("convenience", "root", 0, "Ease of access and use of facilities")
    ontology.added_as_child("convenience", "location", "child", 0, "Accessibility of location")
    ontology.added_as_child("convenience", "hours", "child", 0, "Operating hours convenience")
    ontology.added_as_child("convenience", "wait time", "child", 0, "Time spent waiting")
    ontology.added_as_child("convenience", "ordering process", "child", 0, "Ease of ordering")

    # Comfort 相關特徵
    ontology.added_as_root("comfort", "root", 0, "Physical and emotional comfort aspects")
    ontology.added_as_child("comfort", "seating", "child", 0, "Comfort of seating arrangements")
    ontology.added_as_child("comfort", "space", "child", 0, "Amount of personal space")
    ontology.added_as_child("comfort", "temperature", "child", 0, "Indoor temperature comfort")
    ontology.added_as_child("comfort", "accessibility", "child", 0, "Ease of movement and access")

    # Experience 相關特徵
    ontology.added_as_root("experience", "root", 0, "Overall satisfaction and subjective aspects of the dining experience including atmosphere, occasion suitability, and emotional response")
    ontology.added_as_child("experience", "overall satisfaction", "child", 0, "General satisfaction level")
    ontology.added_as_child("experience", "authenticity", "child", 0, "Authenticity of dining experience")
    ontology.added_as_child("experience", "atmosphere", "child", 0, "Overall dining atmosphere")
    ontology.added_as_child("experience", "return intention", "child", 0, "Likelihood to return")

    for i, r in tqdm(enumerate(reviews), total=review_cnt, desc="Process reviews"):
        feature_scores = extract_feature_mentions(r["text"], ontology, args.dset)
        for phrase, desc, score in feature_scores:
            is_new = ontology.add_or_update_node(r["review_id"], clean_phrase(phrase), desc, score)
            if is_new:
                new_since_refine += 1
                new_features.append(clean_phrase(phrase))

        # 每累積到 K 個新節點，就暫停並提示：  
        if new_since_refine >= K or i == review_cnt - 1:
            human_in_the_loop_update(new_since_refine, new_features, update_cnt, ontology)
            new_since_refine = 0
            new_features = []
            update_cnt += 1
        
        node_cnt.append(len(ontology.nodes))
    
    # 最後存檔
    ontology.save_json(Path("cache/ontology_human_in_the_loop.json"))
    with open("node_cnt_human_in_the_loop.json", "w") as fp:
        json.dump(node_cnt, fp)
    return ontology
