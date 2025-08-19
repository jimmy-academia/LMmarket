
import json
from collections import defaultdict
from pathlib import Path
from utils import loadj, dumpj

def load_data(data_dir: Path):
    """載入所有需要的 JSON 檔案"""
    print("Loading data...")
    yelp_data = loadj(data_dir / "yelp_data.json")
    ontology_data = loadj(data_dir / "ontology.json")
    review_to_features = loadj(data_dir / "review_to_features.json")
    return yelp_data, ontology_data, review_to_features

def get_ontology_structure(ontology_data):
    """
    解析 ontology，獲取父子對應關係、節點深度和 alias 對應表。
    """
    parent_to_children_map = defaultdict(list)
    all_nodes = set(ontology_data.keys())
    nodes_with_parent = set()
    alias_to_main_node_map = {}

    for name, details in ontology_data.items():
        parent = details.get('parent')
        if parent:
            parent_to_children_map[parent].append(name)
            nodes_with_parent.add(name)
        
        # 建立別名對應表
        aliases = details.get('aliases', [])
        for alias in aliases:
            alias_to_main_node_map[alias] = name
    
    roots = all_nodes - nodes_with_parent
    
    node_depths = {}
    nodes_by_depth = defaultdict(list)
    queue = [(root, 0) for root in roots]
    visited = set()

    while queue:
        node_name, depth = queue.pop(0)
        if node_name in visited:
            continue
        visited.add(node_name)

        node_depths[node_name] = depth
        nodes_by_depth[depth].append(node_name)

        for child in parent_to_children_map.get(node_name, []):
            if child not in visited:
                queue.append((child, depth + 1))

    max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0
    
    print(f"Ontology structure parsed. Max depth: {max_depth}, Root nodes: {len(roots)}, Aliases: {len(alias_to_main_node_map)}")
    return parent_to_children_map, nodes_by_depth, max_depth, alias_to_main_node_map

def build_node_scores(entity_to_reviews, review_to_features, ontology_structure, entity_type="USER"):
    """
    為每個 ontology node 計算所有 entity 的分數結構
    回傳格式: {node_name: {entity_id: {'scores': [score1, score2, ...], 'avg': avg_score}}}
    """
    parent_to_children_map, nodes_by_depth, max_depth, alias_to_main_node_map = ontology_structure
    
    # 初始化每個 node 的分數結構
    node_scores = defaultdict(lambda: defaultdict(lambda: {'scores': [], 'avg': -100}))
    
    print(f"Building {entity_type} scores for ontology nodes...")
    
    for entity_id, review_ids in entity_to_reviews.items():
        for review_id in review_ids:
            if review_id in review_to_features:
                for feature_name, score in review_to_features[review_id]:
                    # alias 解析：如果 feature_name 是別名，轉換為主節點名稱
                    main_node_name = alias_to_main_node_map.get(feature_name, feature_name)
                    node_scores[main_node_name][entity_id]['scores'].append(score)
    
    # 計算平均分數
    import statistics
    for node_name in node_scores:
        for entity_id in node_scores[node_name]:
            scores = node_scores[node_name][entity_id]['scores']
            if scores:
                node_scores[node_name][entity_id]['avg'] = round(statistics.mean(scores), 2)
    
    return dict(node_scores)

def build_profiles(entity_to_reviews, review_to_features, ontology_structure, decay_factor=0.8):
    """
    為所有實體 (使用者或物品) 計算最終的階層式分數輪廓。
    """
    parent_to_children_map, nodes_by_depth, max_depth, alias_to_main_node_map = ontology_structure
    all_entity_profiles = {}
    total_entities = len(entity_to_reviews)
    count = 0

    for entity_id, review_ids in entity_to_reviews.items():
        count += 1
        if count % 5000 == 0:
            print(f"  - Processing entity {count}/{total_entities}...")

        # 步驟 A: 收集並計算每個 feature 的直接平均分（包含別名解析）
        direct_scores = defaultdict(list)
        for review_id in review_ids:
            if review_id in review_to_features:
                for feature_name, score in review_to_features[review_id]:
                    # 別名解析：如果 feature_name 是別名，轉換為主節點名稱
                    main_node_name = alias_to_main_node_map.get(feature_name, feature_name)
                    direct_scores[main_node_name].append(score)
        
        if not direct_scores:
            continue

        avg_direct_scores = {name: round(sum(scores) / len(scores), 2)
                             for name, scores in direct_scores.items()}

        # 步驟 B: 由下至上逐層計算最終分數
        final_scores = {}
        for d in range(max_depth, -1, -1):  # 從最深的層級開始
            for node_name in nodes_by_depth.get(d, []):
                
                # 獲取此節點的直接分數
                direct_score_val = avg_direct_scores.get(node_name)
                
                # 獲取來自所有子節點的貢獻分數
                children = parent_to_children_map.get(node_name, [])
                children_scores = [final_scores[child] * decay_factor 
                                   for child in children if child in final_scores]
                
                # 結合所有分數來源
                all_contributions = []
                if direct_score_val is not None:
                    all_contributions.append(direct_score_val)
                if children_scores:
                    all_contributions.extend(children_scores)
                
                # 計算加權平均分並存儲
                if all_contributions:
                    final_scores[node_name] = round(sum(all_contributions) / len(all_contributions), 2)

        all_entity_profiles[entity_id] = final_scores

    return all_entity_profiles

def main():
    """主執行函式"""
    DATA_DIR = Path("cache")
    DATA_DIR.mkdir(exist_ok=True)

    yelp_data, ontology_data, review_to_features = load_data(DATA_DIR)
    
    print(f"Loaded Yelp data (users: {len(yelp_data.get('USERS', []))}, items: {len(yelp_data.get('ITEMS', []))})")
    print(f"Loaded ontology data (nodes: {len(ontology_data)})")
    print(f"Loaded review to features mapping (reviews: {len(review_to_features)})")

    ontology_structure = get_ontology_structure(ontology_data)

    # --- Build User Profiles ---
    print("\nBuilding User Profiles...")
    users_list = yelp_data.get('USERS', [])
    users_dict = {user['user_id']: user['review_ids'] for user in users_list if 'user_id' in user and 'review_ids' in user}
    
    user_profiles = build_profiles(
        entity_to_reviews=users_dict,
        review_to_features=review_to_features,
        ontology_structure=ontology_structure
    )
    
    # Build User Scores for Ontology Nodes
    user_node_scores = build_node_scores(
        entity_to_reviews=users_dict,
        review_to_features=review_to_features,
        ontology_structure=ontology_structure,
        entity_type="USER"
    )
    
    user_profiles_path = DATA_DIR / "user_profiles.json"
    dumpj(user_profiles_path, user_profiles)
    print(f"Generated profiles for {len(user_profiles)} users.")
    print(f"User profiles saved to {user_profiles_path}")

    # --- Build Item Profiles ---
    print("\nBuilding Item Profiles...")
    items_list = yelp_data.get('ITEMS', [])
    items_dict = {item['business_id']: item['review_ids'] for item in items_list if 'business_id' in item and 'review_ids' in item}
    
    item_profiles = build_profiles(
        entity_to_reviews=items_dict,
        review_to_features=review_to_features,
        ontology_structure=ontology_structure
    )

    # Build Item Scores for Ontology Nodes
    item_node_scores = build_node_scores(
        entity_to_reviews=items_dict,
        review_to_features=review_to_features,
        ontology_structure=ontology_structure,
        entity_type="ITEM"
    )

    item_profiles_path = DATA_DIR / "item_profiles.json"
    dumpj(item_profiles_path, item_profiles)
    print(f"Generated profiles for {len(item_profiles)} items.")
    print(f"Item profiles saved to {item_profiles_path}")

    # --- Save User Node Scores ---
    user_node_scores_path = DATA_DIR / "user_node_scores.json"
    dumpj(user_node_scores_path, user_node_scores)
    print(f"User node scores saved to {user_node_scores_path}")

    # --- Save Item Node Scores ---
    item_node_scores_path = DATA_DIR / "item_node_scores.json"
    dumpj(item_node_scores_path, item_node_scores)
    print(f"Item node scores saved to {item_node_scores_path}")

    print(f"Total nodes with user scores: {sum(1 for node in user_node_scores.values() if node)}")
    print(f"Total nodes with item scores: {sum(1 for node in item_node_scores.values() if node)}")

if __name__ == "__main__":
    main()
