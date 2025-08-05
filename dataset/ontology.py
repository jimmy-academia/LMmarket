from collections import defaultdict
from utils import vlog, ppause
from utils import clean_phrase
from utils import dumpj

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

        self.unprocessed_nodes = []
        self.Ulim = 20

    def add_or_update_node(self, review_id, phrase, description, score):
        cleaned = clean_phrase(phrase)
        node = self.find_or_create_node(cleaned)

        if node is not None:
            self.review2name_score[review_id].append((node.name, score))
        else:
            self.unprocessed_nodes.append((cleaned, description, review_id, score))
            if len(self.unprocessed_nodes) >= self.Ulim:
                self.human_in_loop_operate()

    def find_or_create_node(self, cleaned):
        for node in self.nodes:
            if cleaned in node.aliases:
                vlog(f"Feature '{cleaned}' matched existing alias in node '{node.name}'")
                return node
        return None

    # '''
    def human_in_loop_operate(self):
        # 1. Write to file
        filename = "human_in_loop_ontology_edit.txt"

        with open(filename, "w", encoding="utf-8") as f:
            seen = set()
            for cleaned, _, _, _ in self.unprocessed_nodes:
                if cleaned not in seen:
                    f.write(f"{cleaned}\n")
                    seen.add(cleaned)
            f.write("====\n")
            for root in self.roots:
                self._write_subtree(root, f, depth=0)

        input(f"\n📄 Please edit `{filename}` to rename or organize nodes. Press ENTER when done...")

        # 2. Read edited file and build map: raw_name → new_name
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()

        raw_to_name = {}
        for line in lines:
            line = line.strip()
            if line == "====":
                break
            parts = line.split("\t")
            raw = parts[0]
            new = parts[1] if len(parts) > 1 else raw
            raw_to_name[raw] = new

        # 3. Update ontology nodes
        for raw, new in raw_to_name.items():
            existing_node = next((n for n in self.nodes if n.name == new), None)
            if existing_node:
                if raw not in existing_node.aliases:
                    existing_node.aliases.append(raw)
            else:
                new_node = OntologyNode(name=new, aliases=[raw], description="")
                self.nodes.append(new_node)
                self.roots.append(new_node)

        # 4. Reconnect deferred review mappings
        for raw, _, review_id, score in self.unprocessed_nodes:
            if raw not in raw_to_name:
                vlog(f"⚠️ '{raw}' not resolved by human input — skipping")
                continue
            final_name = raw_to_name[raw]
            node = next(n for n in self.nodes if n.name == final_name)
            self.review2name_score[review_id].append((node.name, score))

        self.unprocessed_nodes = []


    def _write_subtree(self, node, f, depth):
        indent = '\t' * depth
        child_names = ", ".join(child.name for child in node.children)
        line = f"{indent}{node.name}"
        if child_names:
            line += f", {child_names}"
        f.write(line + "\n")
        for child in node.children:
            self._write_subtree(child, f, depth + 1)
    
    # '''

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

    def save(self, path):
        dumpj(path, self.to_dict())
        vlog(f"[Ontology saved to {path}]")

    @classmethod
    def load(cls, path):
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


