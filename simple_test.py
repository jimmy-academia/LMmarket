from dataset.ontology import build_ontology_by_reviews
from pathlib import Path
import argparse

def test_restaurant_features():
    """A minimal complete example of building ontology from restaurant reviews"""
    print("Starting Ontology building test...")
    
    # Test reviews with detailed text descriptions
    reviews = [
        {
            "review_id": "review1",
            "text": """Had an amazing dining experience here! The service was outstanding - 
            our waiter was very attentive and friendly, and the food came out quickly. 
            The dishes were beautifully presented and absolutely delicious. 
            The restaurant atmosphere was lovely and everything was spotlessly clean."""
        },
        {
            "review_id": "review2",
            "text": """The quality of food was excellent, especially the fresh ingredients used. 
            However, the service was a bit slow during peak hours. 
            The portion sizes were generous for the price, making it good value for money. 
            The dining room was comfortable but a little noisy."""
        },
        {
            "review_id": "review3",
            "text": """Great menu variety with many options to choose from. 
            The staff was professional and knowledgeable about the menu. 
            While the food taste was amazing, the plating could be more creative. 
            The restaurant maintains high standards of cleanliness."""
        }
    ]

    # Create args for build_ontology_by_reviews
    args = argparse.Namespace()
    args.dset = 'yelp'  # Using yelp dataset configuration
    
    # Build ontology from reviews with human-in-the-loop refinement
    # Will pause every K=10 new features for manual review
    ont = build_ontology_by_reviews(args, reviews, K=3)
    
    # Test search functionality
    test_queries = [
        "The waiter was very friendly and brought our food quickly",
        "Beautiful presentation of dishes but the taste was average",
        "The restaurant was clean but quite noisy"
    ]
    
    print("\nTesting search functionality:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = ont.search_top_ten(query)
        print("Top 3 related features:")
        for score, node in results[:3]:
            print(f"- {node.name} (similarity: {score:.3f})")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_restaurant_features()
