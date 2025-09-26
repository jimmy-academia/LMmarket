from dataset.build_features import build_ontology_by_reviews
from pathlib import Path
import argparse

def test_restaurant_features():
    """A minimal complete example of building ontology from restaurant reviews"""
    print("Starting Ontology building test...")
    
    # Test reviews with detailed text descriptions
    reviews = [
        {
            "review_id": "review1",
            "text": """This restaurant's menu variety is truly impressive. Their beverage selection stands out - 
            I was particularly impressed with their beverage quality. The whiskey flight quality is exceptional, 
            featuring rare single malts with perfect temperature control. The beverage pricing is reasonable 
            for the quality offered, and the beverage flavor profiles are complex and well-balanced. 
            I noticed the beverage freshness in their craft cocktails, using fresh-pressed juices and house-made syrups. 
            The beverage appearance is also remarkable, with perfect garnishes and elegant glassware presentation.

            The beverage variety is extensive, ranging from craft cocktails to local beers and an impressive wine list. 
            What really impressed me was their menu accuracy - the online menu matched exactly what was available, 
            and the food accuracy was spot-on, with each dish matching its description perfectly."""
        },
        {
            "review_id": "review2",
            "text": """The order accuracy here deserves special mention - every item in our complex group order 
            was correct, including all special requests and modifications. Their takeout convenience is outstanding, 
            with an efficient online ordering system that accurately captures all customization requests. 
            Speaking of which, their food customization options are extensive - they're happy to modify dishes 
            to accommodate preferences and dietary restrictions.

            The beverage selection continues to impress me on every visit. The beverage quality is consistently high - 
            their beverage freshness is evident in every drink, especially in their seasonal cocktails. 
            The beverage flavor combinations are innovative yet balanced, and the beverage appearance shows 
            great attention to detail."""
        },
        {
            "review_id": "review3",
            "text": """The menu variety here is exceptional, with options for every dietary preference. 
            Their beverage selection is particularly noteworthy - the beverage quality is top-notch, 
            especially in their whiskey flight quality. Each whiskey in the flight is carefully selected 
            and served at optimal temperature. The beverage pricing is transparent and fair for the quality received. 
            
            The menu accuracy is impressive - their takeout convenience system is well-organized, 
            with accurate food preparation times and order accuracy that's consistently reliable. 
            The food customization options are extensive, allowing for detailed modifications to any dish. 
            I especially appreciate how their beverage freshness is maintained - they even note the date 
            when wine bottles were opened on their by-the-glass menu."""
        },
        {
            "review_id": "review4",
            "text": """What a fantastic dining experience! The chef's expertise really shines through
            in every dish. The ingredients were top quality and perfectly seasoned.
            The staff went above and beyond with their recommendations and attention to detail.
            The ambiance was perfect for a romantic evening, with thoughtful lighting and elegant decor."""
        },
        {
            "review_id": "review5",
            "text": """Mixed feelings about this place. While the food preparation was excellent,
            with high-quality ingredients and beautiful presentation, the portion sizes were quite small
            for the price point. The service staff was courteous but seemed overwhelmed during the rush hour.
            The restaurant's layout and acoustics made conversation difficult."""
        },
        {
            "review_id": "review6",
            "text": """A truly exceptional culinary experience. The menu offers innovative dishes
            that showcase both traditional and modern cooking techniques. The wine pairing suggestions
            were spot-on. The wait staff was knowledgeable and efficient, though not overly formal.
            The dining space felt intimate despite being quite spacious."""
        },
        {
            "review_id": "review7",
            "text": """The value for money here is outstanding. The portion sizes are generous,
            and the quality of ingredients is consistently high. The service team maintains good timing
            between courses. The restaurant's cleanliness is impeccable, from the dining room to the restrooms.
            The background music creates a pleasant atmosphere without being intrusive."""
        },
        {
            "review_id": "review8",
            "text": """An impressive attention to dietary requirements. The staff was well-versed in
            allergen information and the kitchen was very accommodating with modifications. The presentation
            of each dish was Instagram-worthy. The temperature control in the dining room was perfect,
            and the seating was comfortable even for a lengthy meal."""
        },
        {
            "review_id": "review9",
            "text": """The seasonal menu is a real highlight here. You can taste the freshness of
            locally sourced ingredients in every bite. The cooking techniques are flawless, bringing out
            the natural flavors of each component. The service strikes a perfect balance between
            being attentive and giving diners space. The restaurant's decor complements the culinary experience."""
        }
    ]

    # Create args for build_ontology_by_reviews
    args = argparse.Namespace()
    args.use_feature_cache = False
    args.feature_cache_path = Path("./cache/tmp")
    args.dset = 'yelp'  # Using yelp dataset configuration
    
    # Build ontology from reviews with human-in-the-loop refinement
    # Will pause every K=10 new features for manual review
    ont = build_ontology_by_reviews(args, reviews)
    
    '''
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
    '''
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_restaurant_features()
