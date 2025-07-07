{
  "entity_id": { "type": "string" },
  "entity_type": { "type": "string", "options": ["user", "restaurant"] },
  "profile": {
    "dynamic_potential": {
      "parking_profile": { "type": "string", "options": ["Easy", "Moderate", "Difficult"] },
      "cuisine_specificity": { "type": "list_of_strings" },
      # "price_sensitivity_archetype": { "type": "string", "options": ["Price-Sensitive", "Value-Seeker", "Price-Insensitive"] }
    },
    "static_features": {
      "food_quality": { "type": "float", "range": [0.0, 1.0], "comment": "Score for food quality and taste." },
      "food_variety": { "type": "float", "range": [0.0, 1.0], "comment": "Score for menu variety and uniqueness." },
      "service_attitude": { "type": "float", "range": [0.0, 1.0], "comment": "Score for staff friendliness." },
      "service_speed": { "type": "float", "range": [0.0, 1.0], "comment": "Score for service attentiveness and speed." },
      "ambiance_decor": { "type": "float", "range": [0.0, 1.0], "comment": "Score for decor and design." },
      "ambiance_noise": { "type": "float", "range": [0.0, 1.0], "comment": "Score for noise level (0=very loud, 1=very quiet)." },
      "dietary_accommodation": { "type": "float", "range": [0.0, 1.0], "comment": "Score for availability of dietary options." },
      "reservation_friendliness": { "type": "float", "range": [0.0, 1.0], "comment": "Score for ease of reservations." }
    },
    "price": {
      "value": { "type": "float", "range": [1.0, 4.0], "comment": "For a user, their Budget (B). For a restaurant, its Price (p)." },
      "sensitivity": { "type": "float", "range": "[0.0, ...)", "comment": "For a user, their price sensitivity (β). Not applicable for restaurants." }
    },
    "weights": {
      "comment": "Parallel structure to static_features. All values are normalized.",
      "food_quality": { "type": "float", "range": [0.0, 1.0] },
      "food_variety": { "type": "float", "range": [0.0, 1.0] },
      "service_attitude": { "type": "float", "range": [0.0, 1.0] },
      "service_speed": { "type": "float", "range": [0.0, 1.0] },
      "ambiance_decor": { "type": "float", "range": [0.0, 1.0] },
      "ambiance_noise": { "type": "float", "range": [0.0, 1.0] },
      "dietary_accommodation": { "type": "float", "range": [0.0, 1.0] },
      "reservation_friendliness": { "type": "float", "range": [0.0, 1.0] }
    }
  }
}

import numpy as np
import math
from typing import Dict, Any, Tuple, List

def _profiles_to_vectors(
    user_profile: Dict[str, Any],
    restaurant_profile: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms the profile dictionaries into NumPy vectors for calculation.
    
    This function defines the canonical order of features, ensuring vectors align.
    """
    # Define the ordered list of feature keys to ensure vectors are aligned
    feature_keys = [
        "food_quality", "food_variety", "service_attitude", "service_speed",
        "ambiance_decor", "ambiance_noise", "dietary_accommodation", 
        "reservation_friendliness"
    ]

    # Extract values in the defined order, providing a default of 0 if a key is missing
    user_pref_vec = np.array([user_profile['profile']['static_features'].get(k, 0.0) for k in feature_keys])
    user_weight_vec = np.array([user_profile['profile']['weights'].get(k, 1.0) for k in feature_keys])
    
    item_feat_vec = np.array([restaurant_profile['profile']['static_features'].get(k, 0.0) for k in feature_keys])
    item_weight_vec = np.array([restaurant_profile['profile']['weights'].get(k, 1.0) for k in feature_keys])
    
    return user_pref_vec, user_weight_vec, item_feat_vec, item_weight_vec

def _calculate_preference_utility(
    user_pref_vec: np.ndarray,
    user_weight_vec: np.ndarray,
    item_feat_vec: np.ndarray,
    item_weight_vec: np.ndarray
) -> float:
    """Calculates the alignment of features and preferences using weighted cosine similarity."""
    # Apply weights to the base vectors
    user_weighted_vec = user_pref_vec * user_weight_vec
    item_weighted_vec = item_feat_vec * item_weight_vec

    # Calculate cosine similarity between the two weighted vectors
    dot_product = np.dot(user_weighted_vec, item_weighted_vec)
    user_norm = np.linalg.norm(user_weighted_vec)
    item_norm = np.linalg.norm(item_weighted_vec)

    if user_norm == 0 or item_norm == 0:
        return 0.0
        
    # The result is naturally in the [0, 1] range because features are [0, 1]
    return dot_product / (user_norm * item_norm)

def _calculate_price_utility(
    user_profile: Dict[str, Any],
    restaurant_profile: Dict[str, Any]
) -> float:
    """Calculates the user's utility from price using the provided formula."""
    user_price_info = user_profile['profile']['price']
    restaurant_price_info = restaurant_profile['profile']['price']

    beta = user_price_info.get('sensitivity', 1.0)  # User's price sensitivity
    budget = user_price_info.get('value', 3.0)      # User's budget
    price = restaurant_price_info.get('value', 2.0) # Restaurant's price
    
    # The price must not exceed the budget for the log utility to be positive
    if price > budget:
        # A large negative utility can be returned, but 0 is simpler for combining.
        # This signifies an "unaffordable" choice.
        return 0.0 

    # Utility formula: β * log(B - p + 1)
    utility = beta * math.log(budget - price + 1)
    return utility

def calculate_match_score(
    user_profile: Dict[str, Any], 
    restaurant_profile: Dict[str, Any],
    alpha: float = 0.7
) -> Dict[str, float]:
    """
    Calculates a final match score by combining preference and price utilities.

    Args:
        user_profile (dict): The user's profile object.
        restaurant_profile (dict): The restaurant's profile object.
        alpha (float): The trade-off parameter between preference (alpha) and 
                       price (1-alpha). Ranges from 0.0 to 1.0.

    Returns:
        dict: A dictionary containing the final score and its components.
    """
    # 1. Transform dictionaries into aligned numerical vectors
    user_pref_vec, user_weight_vec, item_feat_vec, item_weight_vec = \
        _profiles_to_vectors(user_profile, restaurant_profile)

    # 2. Calculate the Preference Utility component
    preference_utility = _calculate_preference_utility(
        user_pref_vec, user_weight_vec, item_feat_vec, item_weight_vec
    )

    # 3. Calculate the Price Utility component
    price_utility = _calculate_price_utility(user_profile, restaurant_profile)

    # --- Normalization Step (Crucial for real-world application) ---
    # Before combining, price_utility should be normalized to the same [0, 1] scale
    # as preference_utility. This requires calculating the utility for all possible
    # restaurants to find the min/max range, then scaling.
    # For this example, we'll assume a simplified normalization.
    # Let's assume max possible price utility is log(4)=~1.38
    max_assumed_price_utility = math.log(4) 
    normalized_price_utility = min(price_utility / max_assumed_price_utility, 1.0) if max_assumed_price_utility > 0 else 0.0

    # 4. Combine utilities using the alpha trade-off parameter
    final_score = (alpha * preference_utility) + ((1 - alpha) * normalized_price_utility)
    
    return {
        "final_score": round(final_score, 4),
        "components": {
            "preference_utility": round(preference_utility, 4),
            "price_utility": round(price_utility, 4),
            "normalized_price_utility": round(normalized_price_utility, 4)
        }
    }

# Example Usage:
if __name__ == '__main__':
    # Mock profiles conforming to the schema
    mock_user = {
        'entity_id': 'user123', 'entity_type': 'user', 'profile': {
            'static_features': {'food_quality': 0.9, 'food_variety': 0.8, 'service_attitude': 0.7, 'service_speed': 0.6, 'ambiance_decor': 0.8, 'ambiance_noise': 0.2, 'dietary_accommodation': 0.5, 'reservation_friendliness': 0.8},
            'weights': {'food_quality': 1.0, 'food_variety': 0.8, 'service_attitude': 0.9, 'service_speed': 0.5, 'ambiance_decor': 0.6, 'ambiance_noise': 0.9, 'dietary_accommodation': 0.4, 'reservation_friendliness': 0.7},
            'price': {'value': 3.0, 'sensitivity': 1.5} # Budget=$$$, Sensitivity=1.5
        }
    }
    mock_restaurant = {
        'entity_id': 'rest456', 'entity_type': 'restaurant', 'profile': {
            'static_features': {'food_quality': 0.85, 'food_variety': 0.7, 'service_attitude': 0.75, 'service_speed': 0.8, 'ambiance_decor': 0.9, 'ambiance_noise': 0.3, 'dietary_accommodation': 0.9, 'reservation_friendliness': 0.9},
            'weights': {'food_quality': 1.0, 'food_variety': 0.7, 'service_attitude': 0.8, 'service_speed': 0.9, 'ambiance_decor': 0.7, 'ambiance_noise': 0.8, 'dietary_accommodation': 0.6, 'reservation_friendliness': 0.6},
            'price': {'value': 2.0} # Price=$$
        }
    }
    
    # --- Calculate score with default alpha (more weight on features) ---
    match_result_default = calculate_match_score(mock_user, mock_restaurant, alpha=0.7)
    print(f"Match Score (alpha=0.7): {match_result_default}")
    
    # --- Calculate score with low alpha (more weight on price) ---
    match_result_pricey = calculate_match_score(mock_user, mock_restaurant, alpha=0.2)
    print(f"Match Score (alpha=0.2): {match_result_pricey}")