from dataset.ontology import Ontology, OntologyNode, clean_phrase
from pathlib import Path
import pytest

def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def test_basic_ontology_operations():
    """Test basic node addition and alias handling"""
    ontology = Ontology()
    
    # Test adding a root node
    review_id = "test_review_1"
    ontology.add_or_update_node(
        review_id=review_id,
        phrase="food quality",
        description="The overall quality of food items, including taste, freshness, and presentation",
        score=0.8
    )
    
    assert "food quality" in ontology.nodes
    assert len(ontology.nodes) == 1
    assert len(ontology.review2node_id_score[review_id]) == 1
    
    # Test adding a child node
    ontology.add_or_update_node(
        review_id=review_id,
        phrase="ingredient freshness",
        description="The freshness and quality of ingredients used in food preparation",
        score=0.9
    )
    
    # Test adding an alias
    ontology.add_or_update_node(
        review_id=review_id,
        phrase="food ingredients",
        description="The freshness and quality of ingredients used in food preparation",
        score=0.7
    )

def test_hierarchy_operations():
    """Test hierarchy creation and modification"""
    ontology = Ontology()
    
    # Setup test data
    test_data = [
        ("service quality", "Overall quality of customer service", 0.8),
        ("staff friendliness", "How friendly and welcoming the staff are", 0.9),
        ("staff professionalism", "How professional and competent the staff are", 0.7),
        ("response time", "How quickly staff responds to customer needs", 0.6),
    ]
    
    # Add all test nodes
    for phrase, desc, score in test_data:
        ontology.add_or_update_node("test_review", phrase, desc, score)
    
    # Save hierarchy to file
    test_file = Path("test_hierarchy.txt")
    ontology.save_txt(test_file)
    
    # Modify hierarchy
    with open(test_file, "w") as f:
        f.write("service quality\n")
        f.write("\tstaff friendliness\n")
        f.write("\tstaff professionalism\n")
        f.write("\t\tresponse time\n")
    
    # Read and verify hierarchy
    success = ontology.read_txt(test_file)
    assert success == True
    
    # Verify relationships
    service_node = ontology.nodes["service quality"]
    staff_friendly_node = ontology.nodes["staff friendliness"]
    response_time_node = ontology.nodes["response time"]
    
    assert staff_friendly_node.parent == service_node
    assert "staff friendliness" in service_node.children
    assert response_time_node.parent.parent == service_node
    
    # Clean up test file
    test_file.unlink()

def test_search_functionality():
    """Test search and embedding functionality"""
    ontology = Ontology()
    
    # Add test nodes
    test_data = [
        ("food quality", "Overall quality of food items", 0.8),
        ("service speed", "How quickly orders are served", 0.7),
        ("cleanliness", "Cleanliness of the establishment", 0.9),
        ("price value", "Value for money of items", 0.6),
    ]
    
    for phrase, desc, score in test_data:
        ontology.add_or_update_node("test_review", phrase, desc, score)
    
    # Test search functionality
    query = "How quick was the service and food delivery"
    results = ontology.search_top_ten(query)
    
    # Verify that service speed is among top results
    found_speed = False
    for _, node in results:
        if node.name == "service speed":
            found_speed = True
            break
    
    assert found_speed == True

def test_remove_functionality():
    """Test node removal and hierarchy updates"""
    ontology = Ontology()
    
    # Setup test data
    test_data = [
        ("ambiance", "Restaurant atmosphere and setting", 0.8),
        ("lighting", "Quality and appropriateness of lighting", 0.7),
        ("noise level", "Level of ambient noise", 0.6),
    ]
    
    # Add all test nodes
    for phrase, desc, score in test_data:
        ontology.add_or_update_node("test_review", phrase, desc, score)
    
    # Save current hierarchy
    test_file = Path("test_remove.txt")
    ontology.save_txt(test_file)
    
    # Add remove flag to a node
    with open(test_file, "w") as f:
        f.write("ambiance\n")
        f.write("\tlighting\n")
        f.write("\tnoise level (remove)\n")
    
    # Read modified hierarchy
    success = ontology.read_txt(test_file)
    assert success == True
    
    # Verify node was removed
    assert "noise level" not in ontology.nodes
    assert len(ontology.nodes) == 2
    
    # Clean up test file
    test_file.unlink()



def _build_base_tree(o):
    """
    建立基礎節點並寫入一份「合法層級」TXT（深度<=3）：
    service quality
      ├─ staff friendliness
      └─ staff professionalism
           └─ response time
    """
    data = [
        ("service quality", "Overall quality of customer service", 0.8),
        ("staff friendliness", "How friendly and welcoming the staff are", 0.9),
        ("staff professionalism", "How professional and competent the staff are", 0.7),
        ("response time", "How quickly staff responds to customer needs", 0.6),
    ]
    for p, d, s in data:
        o.add_or_update_node("test_review", p, d, s)

def test_hitl_rename_parent_and_child(tmp_path):
    """
    測試在 HITL 純文字中改名 parent 與 child，並驗證：
    1) read_txt() 回 True
    2) 新名字存在、舊名字不存在
    3) 父子關係正確以「新名」重建
    """
    o = Ontology()
    _build_base_tree(o)

    # 建立合法階層
    base_txt = tmp_path / "base.txt"
    _write_lines(base_txt, [
        "service quality",
        "\tstaff friendliness",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    assert o.read_txt(base_txt) is True

    # 進行改名（行尾語法）
    # - 將 parent: "service quality" -> "service excellence"
    # - 將 child : "staff friendliness" -> "staff warmth"
    edited = tmp_path / "edited_rename.txt"
    _write_lines(edited, [
        "service quality (rename: service excellence)",
        "\tstaff friendliness (rename: staff warmth)",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])

    ok = o.read_txt(edited)
    assert ok is True

    # 驗證新舊名稱
    assert "service excellence" in o.nodes
    assert "service quality" not in o.nodes
    assert "staff warmth" in o.nodes
    assert "staff friendliness" not in o.nodes

    # 驗證父子關係（以新名為準）
    svc = o.nodes["service excellence"]
    staff_warm = o.nodes["staff warmth"]
    staff_prof = o.nodes["staff professionalism"]
    resp = o.nodes["response time"]

    assert staff_warm.parent is svc
    assert "staff warmth" in svc.children
    assert staff_prof.parent is svc
    assert resp.parent is staff_prof

def test_hitl_rename_duplicate_target_rejected(tmp_path):
    """
    兩個不同節點改成同一個新名 -> 應拒絕（read_txt() 回 False），
    且原本結構維持不變。
    """
    o = Ontology()
    _build_base_tree(o)

    base_txt = tmp_path / "base.txt"
    _write_lines(base_txt, [
        "service quality",
        "\tstaff friendliness",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    assert o.read_txt(base_txt) is True

    # 快照原本狀態（名稱集合與部分父子）
    orig_names = set(o.nodes.keys())
    orig_parent = {
        "staff friendliness": o.nodes["staff friendliness"].parent.name,
        "staff professionalism": o.nodes["staff professionalism"].parent.name,
        "response time": o.nodes["response time"].parent.name,
    }

    # 兩行都改到 "dup-name" => 應該被拒絕
    bad_txt = tmp_path / "rename_conflict.txt"
    _write_lines(bad_txt, [
        "service quality",
        "\tstaff friendliness (rename: dup-name)",
        "\tstaff professionalism (rename: dup-name)",
        "\t\tresponse time",
    ])
    ok = o.read_txt(bad_txt)
    assert ok is False, "兩個節點改成同一目標名，應被 read_txt() 拒絕"

    # 驗證結構仍維持原狀
    assert set(o.nodes.keys()) == orig_names
    assert o.nodes["staff friendliness"].parent.name == orig_parent["staff friendliness"]
    assert o.nodes["staff professionalism"].parent.name == orig_parent["staff professionalism"]
    assert o.nodes["response time"].parent.name == orig_parent["response time"]

def test_hitl_rename_and_remove_same_line_rejected(tmp_path):
    """
    同一行同時 (rename: ...) 與 (remove) -> 應拒絕（read_txt() 回 False）
    """
    o = Ontology()
    _build_base_tree(o)

    base_txt = tmp_path / "base.txt"
    _write_lines(base_txt, [
        "service quality",
        "\tstaff friendliness",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    assert o.read_txt(base_txt) is True

    bad_txt = tmp_path / "rename_and_remove.txt"
    _write_lines(bad_txt, [
        "service quality",
        "\tstaff friendliness (rename: staff warmth) (remove)",  # 不允許同時出現
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    ok = o.read_txt(bad_txt)
    assert ok is False, "同一行同時 rename 與 remove 應被拒絕"

def test_hitl_rename_parent_key_mapping_for_children(tmp_path):
    """
    改名一個有子節點的父：確認重建時會以「新父名」接回所有孩子。
    """
    o = Ontology()
    _build_base_tree(o)

    base_txt = tmp_path / "base.txt"
    _write_lines(base_txt, [
        "service quality",
        "\tstaff friendliness",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    assert o.read_txt(base_txt) is True

    # 將父 "service quality" 改為 "svc"
    edited = tmp_path / "rename_parent.txt"
    _write_lines(edited, [
        "service quality (rename: svc)",
        "\tstaff friendliness",
        "\tstaff professionalism",
        "\t\tresponse time",
    ])
    ok = o.read_txt(edited)
    assert ok is True

    assert "svc" in o.nodes and "service quality" not in o.nodes
    svc = o.nodes["svc"]
    # 兩個孩子都應該掛在 "svc" 名下
    assert "staff friendliness" in svc.children
    assert "staff professionalism" in svc.children
    assert o.nodes["staff friendliness"].parent is svc
    assert o.nodes["staff professionalism"].parent is svc
    o.save_txt(tmp_path / "final_ontology.txt")



if __name__ == "__main__":
    # test_hierarchy_operations()
    #test_hitl_rename_parent_and_child(Path("."))
    #test_hitl_rename_and_remove_same_line_rejected(Path("."))
    #test_hitl_rename_duplicate_target_rejected(Path("."))
    test_hitl_rename_parent_key_mapping_for_children(Path("."))
    print("All tests passed successfully!")
