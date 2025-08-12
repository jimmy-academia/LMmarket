from dataset.ontology import Ontology, OntologyNode, clean_phrase
from pathlib import Path
import pytest

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

if __name__ == "__main__":
    test_basic_ontology_operations()
    test_hierarchy_operations()
    test_search_functionality()
    test_remove_functionality()
    print("All tests passed successfully!")
