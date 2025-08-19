from typing import List, Tuple, Dict, Optional
from collections import defaultdict, OrderedDict
from pathlib import Path
import numpy as np
import json
import logging
import statistics
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as csim
from utils import readf, vprint, pause_if
from functools import partial
from tqdm import tqdm
import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"

from llm import query_llm
from debug import check

### --- Flags --- ###
VERBOSE = True
PAUSE = False
vlog = partial(vprint, flag=VERBOSE)
ppause = partial(pause_if, flag=PAUSE)

### --- Embedding Setup --- ###

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed(text: str) -> np.ndarray:
    return embed_model.encode(text, normalize_embeddings=True)

def cosine(a, b):
    return float(csim(a.reshape(1, -1), b.reshape(1, -1))[0][0])

def clean_phrase(phrase: str) -> str:
    return phrase.lower().strip("* ").split(". ")[-1].strip()

### --- Ontology Node and Ontology Structure --- ###

class OntologyNode:
    def __init__(self, node_id, name, description, embedding, type_id_pairs):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.aliases = {name}
        self.embedding = embedding
        self.children = {}
        self.parent = None
        self.user2score = defaultdict(dict)
        self.item2score = defaultdict(dict)
        for type_, id in type_id_pairs:
            if type_ == "USER": self.user2score[id] = {'scores': [], 'avg': -100}
            if type_ == "ITEM": self.item2score[id] = {'scores': [], 'avg': -100}

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
    def __init__(self, type_id_pairs):
        self.nodes = {}
        self.type_id_pairs = type_id_pairs
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
    
    def add_node_score(self, node_name, type_, id, score):
        if type_ ==  "USER":
            # if id not in self.nodes[node_name].user2score:
            #     self.nodes[node_name].user2score[id] = {'scores': [], 'avg': 0.0}
            self.nodes[node_name].user2score[id]['scores'].append(score)
        elif type_ ==  "ITEM":
            # if id not in self.nodes[node_name].item2score:
            #     self.nodes[node_name].item2score[id] = {'scores': [], 'avg': 0.0}
            self.nodes[node_name].item2score[id]['scores'].append(score)
        else:
            vlog("add_or_update_node: unkown type")

    def add_or_update_node(self, review_id, type_, id, phrase, description, score) -> bool:
        """
        回傳 True = 新增了一個 node (包括 CHILD / PARENT 分支)，
        False = 只是加了 alias 或回傳到既有 node。
        """
        cleaned = clean_phrase(phrase)
        # 1) alias match
        for node in self.nodes.values():
            if cleaned in node.aliases:
                self.add_node_score(node.name, type_, id, score)
                self.review2node_id_score[review_id].append((node.node_id, score))
                return False

        # 2) LLM 判斷
        top_candidates = self.search_top_ten(description, top_k=60)
        if top_candidates:
            candidates_text = "\n".join(f"{n.name}: {n.description}" for _, n in top_candidates)
            prompt = f"""
A new feature has been extracted from a review:

New Feature Name: {cleaned}
New Feature Definition: {description}

Below are existing features:
{candidates_text}

Decide the best relationship for the new feature and follow the output format:
- If it's a near synonym or alternative wording of an existing one, output: "ALIAS: <existing name>"
- If it's a more specific case of an existing feature, output: "CHILD: <existing name>"
- If it's a more general feature that should subsume an existing one, output: "PARENT: <existing name>"
- If no relationship, output: "NEW"
"""
            decision = query_llm(prompt).strip()
            if decision != "NEW" and \
                  decision.replace("ALIAS: ", "").replace("CHILD: ", "").replace("PARENT: ", "") not in self.nodes:
                vlog(f"add_or_update_node: invalid decision \'{decision}\'")
                return False

            if decision.startswith("ALIAS:"):
                target = decision.split(":",1)[1].strip()
                self.nodes[target].update(cleaned)
                self.add_node_score(target, type_, id, score)
                self.review2node_id_score[review_id].append((target, score))
                vlog(f"'{cleaned}' added as ALIAS to '{target}'")
                return False

            elif decision.startswith("CHILD:"):
                parent = decision.split(":",1)[1].strip()
                new_id = cleaned
                new_node = OntologyNode(new_id, cleaned, description, embed(cleaned), self.type_id_pairs)
                new_node.parent = self.nodes[parent]
                self.nodes[parent].children[new_id] = new_node
                self.nodes[new_id] = new_node
                self.add_node_score(cleaned, type_, id, score)
                self.review2node_id_score[review_id].append((new_id, score))
                vlog(f"'{cleaned}' added as CHILD to '{parent}'")
                return True

            elif decision.startswith("PARENT:"):
                child = decision.split(":",1)[1].strip()
                new_id = cleaned
                new_node = OntologyNode(new_id, cleaned, description, embed(cleaned), self.type_id_pairs)
                new_node.children[child] = self.nodes[child]
                self.nodes[child].parent = new_node
                self.nodes[new_id] = new_node
                self.add_node_score(cleaned, type_, id, score)
                self.review2node_id_score[review_id].append((new_id, score))
                vlog(f"'{cleaned}' added as PARENT to '{child}'")
                return True

        # 3) 全新 node
        node_id = cleaned
        self.nodes[node_id] = OntologyNode(node_id, cleaned, description, embed(cleaned), self.type_id_pairs)
        self.add_node_score(cleaned, type_, id, score)
        self.review2node_id_score[review_id].append((node_id, score))
        return True

    def cal_node_scores_mean(self):
        for node in self.nodes.values():
            for uid in node.user2score:
                if node.user2score[uid]['scores']: node.user2score[uid]['avg'] = statistics.mean(node.user2score[uid]['scores'])
            for iid in node.item2score:
                if node.item2score[iid]['scores']: node.item2score[iid]['avg'] = statistics.mean(node.item2score[iid]['scores'])

    def save_json(self, path: Path):
        json_dict = {
            name: {
                "name": n.name,
                "description": n.description,
                "aliases": list(n.aliases),
                "children": list(n.children.keys()),
                "parent": ("None" if n.parent == None else n.parent.name),
                "user2score": n.user2score,
                "item2score": n.item2score,
            } for name, n in self.nodes.items()
        }
        with open(path, "w") as f:
            json.dump(json_dict, f, indent=2)
    
    def save_txt(self, path: Path):
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

        def dfs(node_id: str, depth: int, stack: set):
            if node_id in stack:
                lines.append("\t"*depth + f"{json_dict[node_id].get('name', node_id)} (cycle detected)")
                return
            stack = set(stack) | {node_id}
            lines.append("\t"*depth + json_dict[node_id].get("name", node_id))
            for child in (json_dict[node_id].get("children") or []):
                if child in json_dict:
                    dfs(child, depth+1, stack)
                else:
                    lines.append("\t"*(depth+1) + f"{child} (missing)")

        # Write known roots first
        for r in roots:
            dfs(r, 0, set())

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

        def split_remove_flag(name: str) -> Tuple[str, bool]:
            s = name.strip()
            low = s.lower()
            if low.endswith("(remove)"):
                base = s[: -len("(remove)")].rstrip()
                return base, True
            return s, False

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
            name_raw = line[i:].strip()
            if not name_raw:
                vlog(f"read_txt: empty name at line {line_no}")
                return False

            name, rm = split_remove_flag(name_raw)
            parsed.append((depth, name, rm))

        if not parsed:
            vlog("read_txt: file is empty after parsing.")
            return False

        # 先依「名稱（不含 remove 標記）」檢查重複
        names_in_file = [name for _, name, _ in parsed]
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
            depth, name, rm = parsed[i]
            if rm:
                removed.add(name)
                i += 1
                # 收掉整個子樹
                while i < len(parsed) and parsed[i][0] > depth:
                    removed.add(parsed[i][1])
                    i += 1
            else:
                kept_lines.append((depth, name))
                i += 1

        node_names = set(self.nodes.keys())
        kept_names = [n for _, n in kept_lines]

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

        for depth, name in kept_lines:
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

        # 重新連線（先清空）
        for node in self.nodes.values():
            node.parent = None
            node.children.clear()

        # 重建父子
        try:
            for child_name, p_name in parent_of_new.items():
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

def extract_feature_mentions(text: str, ontology: Ontology = None, dset = None, model="openai") -> List[Tuple[str, str, float]]:
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

def human_in_the_loop_update(new_since_refine: int, new_features: List[str], ontology: Ontology = None) -> None:
    ontology.save_txt(Path("cache/ontology_human_in_the_loop.txt"))
    print(f"\n>>> 已新增 {new_since_refine} 個新 feature：{new_features}\n>>> 請打開 Ontology 進行微調，完成後按 Enter 繼續...")
    input()
    valid = ontology.read_txt(Path("cache/ontology_human_in_the_loop.txt"))
    while valid == False:
        print(f"\n>>> 微調出錯請重試，完成後按 Enter 繼續...")
        input()
        valid = ontology.read_txt(Path("cache/ontology_human_in_the_loop.txt"))

### --- Main Ontology Building Function --- ###
def build_ontology_by_reviews(args, review_type_id_pairs: List[Tuple[Dict, str, str]], K: int = 10) -> Ontology:
    review_cnt = len(review_type_id_pairs)

    node_cnt = []
    new_since_refine = 0
    new_features = []

    type_id_pairs = [(type_, id) for _, type_, id in review_type_id_pairs]
    ontology = Ontology(type_id_pairs)

    for i, (r, type_, id) in tqdm(enumerate(review_type_id_pairs), total=review_cnt, desc="Process reviews"):
        feature_scores = extract_feature_mentions(r["text"], ontology, args.dset)
        for phrase, desc, score in feature_scores:
            is_new = ontology.add_or_update_node(r["review_id"], type_, id, clean_phrase(phrase), desc, score)
            if is_new:
                new_since_refine += 1
                new_features.append(clean_phrase(phrase))

        # 每累積到 K 個新節點，就暫停並提示：  
        if new_since_refine >= K or i == review_cnt - 1:
            human_in_the_loop_update(new_since_refine, new_features, ontology)
            new_since_refine = 0
            new_features = []
        
        node_cnt.append(len(ontology.nodes))
    
    ontology.cal_node_scores_mean()

    # 最後存檔
    ontology.save_json(Path("cache/ontology_human_in_the_loop.json"))
    print("new ontology is saved to \"cache/ontology_human_in_the_loop.json\"")
    with open("node_cnt_human_in_the_loop.json", "w") as fp:
        json.dump(node_cnt, fp)
    return ontology
