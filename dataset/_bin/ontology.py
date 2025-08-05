# find_or_create_node

        # root_choice = self.choose_root(cleaned, description)
        # vlog(f"root_choice decision: {root_choice}")
        # if root_choice.lower() == "new":
        #     new_node = OntologyNode(cleaned, description)
        #     self.nodes.append(new_node)
        #     self.roots.append(new_node)
        #     vlog("created new root")
        #     return new_node
        # else:
        #     return self.add_to_root(cleaned, description, root_choice)


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