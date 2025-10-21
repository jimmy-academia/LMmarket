import json
from api import query_llm, user_struct, system_struct, assistant_struct

def infer_aspects_weights(query):
    # Strict JSON Schema (OpenAI json_schema mode requires additionalProperties:false)
    schema_body = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "AspectCardsSchema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "aspects": {
                "type": "array",
                "minItems": 3,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                            "pattern": "^[a-z0-9_]{2,40}$",
                            "description": "Concise aspect token (lowercase, dot/underscore allowed)."
                        },
                        "weight": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "sentence": {
                            "type": "string",
                            "minLength": 8,
                            "maxLength": 220,
                            "description": "ONE sentence (≤ ~25 tokens) to drive dense retrieval."
                        },
                        "positives": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "minLength": 6,
                                "maxLength": 200
                            }
                        },
                        "negatives": {
                            "type": "array",
                            "minItems": 0,
                            "maxItems": 2,
                            "items": {
                                "type": "string",
                                "minLength": 6,
                                "maxLength": 200
                            }
                        }
                    },
                    "required": ["name", "weight", "sentence", "positives", "negatives"]
                }
            }
        },
        "required": ["aspects"]
    }

    response_schema = {
        "name": "AspectCardsSchema",
        "schema": schema_body,
        "strict": True
    }

    sys_prompt = (
        "You convert a natural-language request into aspect cards for dense retrieval.\n"
        "Output JSON only, conforming to the schema.\n\n"
        "Rules:\n"
        "1) Extract 3–10 distinct aspects (tokens like 'quietness', 'natural_light', 'service.speed').\n"
        "2) Assign weights in [0,1] that SUM to 1.0 (relative importance to the user intent).\n"
        "3) For EACH aspect, write ONE compact sentence (≤ ~25 tokens) that captures the retrieval intent.\n"
        "   - Keep it declarative and specific to the aspect.\n"
        "   - Avoid fluff; prefer concrete phrasing useful for semantic search.\n"
        "4) Provide up to 3 positive exemplar snippets and up to 2 negatives (exclusions/anti-examples).\n"
        "   - Keep exemplars short, concrete, sentence-like.\n"
        "5) Use lowercase keys for aspect names; use dot/underscore to indicate hierarchy if helpful."
    )

    # Worked example (restaurant)
    ex_in = (
        "I want a lively restaurant with great seafood, quick service, and outdoor seating, "
        "but it shouldn’t be too expensive."
    )
    ex_out = {
        "aspects": [
            {
                "name": "food_quality.seafood",
                "weight": 0.30,
                "sentence": "A restaurant renowned for fresh, well-executed seafood dishes.",
                "positives": [
                    "oysters are fresh and sweet",
                    "fish cooked perfectly, not overdone",
                    "seafood menu breadth with seasonal specials"
                ],
                "negatives": [
                    "frozen or fishy taste",
                    "limited seafood options"
                ]
            },
            {
                "name": "price.affordability",
                "weight": 0.20,
                "sentence": "Reasonable prices for quality meals without premium markups.",
                "positives": [
                    "good value for portion size",
                    "affordable happy-hour menu"
                ],
                "negatives": [
                    "overpriced for quality"
                ]
            },
            {
                "name": "service.speed",
                "weight": 0.20,
                "sentence": "Prompt service with short wait times from order to table.",
                "positives": [
                    "food arrives quickly even when busy",
                    "attentive servers without delays"
                ],
                "negatives": [
                    "long ticket times"
                ]
            },
            {
                "name": "seating.outdoor",
                "weight": 0.15,
                "sentence": "Comfortable outdoor seating available for regular dining.",
                "positives": [
                    "spacious patio seating",
                    "shade umbrellas on most tables"
                ],
                "negatives": [
                    "outdoor area often closed"
                ]
            },
            {
                "name": "atmosphere.lively",
                "weight": 0.15,
                "sentence": "A lively atmosphere with energetic but pleasant vibe.",
                "positives": [
                    "buzzy dining room energy",
                    "background music complements conversation"
                ],
                "negatives": [
                    "ear-splitting noise levels"
                ]
            }
        ]
    }

    messages = [
        system_struct(sys_prompt),
        user_struct(ex_in),
        assistant_struct(json.dumps(ex_out)),  # seed the pattern with valid JSON
        user_struct(query)
    ]
    return query_llm(messages, json_schema=response_schema, use_json=True)
