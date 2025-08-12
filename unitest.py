from dataset.ontology import Ontology, OntologyNode, clean_phrase
from pathlib import Path
import pytest

def test_basic_ontology_operations():
    o = Ontology()
    # Create a simple ontology
    o.add_or_update_node("review1", "meals quality", "Variety of menu items", 0.9)
    o.add_or_update_node("review1", "beverage quality", "Quality of beverages", 0.8)
    o.add_or_update_node("review2", "beverage flavor", "Flavor of beverage", 0.95)
    o.add_or_update_node("review2", "beverage freshness", "Freshness of beverage", 0.85)

    print(o.nodes["meals_quality"].__repr__())
    print("="*10)
    print(o.nodes["beverage_quality"].__repr__())
    print("="*10)
    print(o.nodes["beverage_flavor"].__repr__())
    print("="*10)
    print(o.nodes["beverage_freshness"].__repr__())

    o.save_txt("unitest.txt")

if __name__ == "__main__":
    test_basic_ontology_operations()
    print("All tests passed successfully!")
