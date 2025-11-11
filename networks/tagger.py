import json
import re
from api import user_struct, system_struct, assistant_struct
from api import batch_run_llm, query_llm

_SYSTEM_PROMPT = """You are an expert annotator for consumer reviews.
Extract ASPECT TAGS that actually appear in the text.

For each aspect tag, provide: name (kebab-case-aspect), sentiment (pos|neg), evidence ("exact substring from review"), confidence (0.0_to_1.0)

Rules:
- Output STRICT JSON only (no markdown or commentary) that validates against the provided JSON Schema.
- Use open-vocabulary, but prefer short, canonical, kebab-case aspect names.
- Only include aspects explicitly supported by text; quote exact evidence substrings (no ellipses).
- Merge synonyms into one canonical name; avoid duplicates.
- No neutral/mixed labels: choose pos or neg; skip ambiguous cases.

Aspect types to consider (used for your internal judgment; do NOT output the type):
- Ontological: whether the review asserts the item belongs (or does not belong) to a category (e.g., "sushi-restaurant", "neapolitan-pizza-restaurant", "cafe").
  • pos = membership affirmed; neg = membership denied.
- Functional: whether the review claims the item HAS/DOES a feature and how well it performs (e.g., "service", "wifi", "noise", "price", "portion-size", "cleanliness", "parking", "outdoor-seating", "vegan-options").
  • pos = has/does well or good performance; neg = lacks/does poorly.
- Teleological: whether the review indicates the item is suitable or unsuitable for a purpose (e.g., "remote-work", "date-night", "kids", "large-groups", "quick-lunch", "business-meeting").
  • pos = suitable; neg = unsuitable.
"""

# A tiny, single example to anchor style (kept minimal to avoid overfitting).
_FEWSHOT_EXAMPLE = [
    user_struct("Coffee was excellent and staff were friendly, but the music was loud and the line took 20 minutes."),
    assistant_struct(json.dumps({"tags": [
        {
            "name": "coffee-quality",
            "sentiment": "pos",
            "evidence": "Coffee was excellent",
            "confidence": 0.92
        },
        {
            "name": "service",
            "sentiment": "pos",
            "evidence": "staff were friendly",
            "confidence": 0.86
        },
        {
            "name": "noise",
            "sentiment": "neg",
            "evidence": "the music was loud",
            "confidence": 0.88
        },
        {
            "name": "wait-time",
            "sentiment": "neg",
            "evidence": "the line took 20 minutes",
            "confidence": 0.9
        }
    ]})),
    user_struct("Authentic Neapolitan pizza place with a wood-fired oven. Perfect for date night and large groups."),
    assistant_struct(json.dumps({
    "tags": [
    {
        "name": "neapolitan-pizza-restaurant",  # ontological
        "sentiment": "pos",
        "evidence": "Authentic Neapolitan pizza place",
        "confidence": 0.94
    },
    {
        "name": "wood-fired oven",  # ontological
        "sentiment": "pos",
        "evidence": "with a wood-fired oven",
        "confidence": 0.78
    },
    
    {
        "name": "date-night",  # teleological
        "sentiment": "pos",
        "evidence": "Perfect for date night",
        "confidence": 0.89
    },
    {
        "name": "large-groups",  # teleological
        "sentiment": "pos",
        "evidence": "Perfect for date night and large groups",
        "confidence": 0.86
    }]
  }))
]


_SCHEMA = {
    "name": "aspect_tags_schema",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "minItems": 0,
                "items": {"$ref": "#/definitions/tag"}
            }
        },
        "definitions": {
            "tag": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "sentiment", "evidence", "confidence"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                        "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["pos", "neg"]
                    },
                    "evidence": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        }
    }
}


# def extract_tags(review_text):
#     # Build the messages for the LLM.
#     msgs = [
#         system_struct(_SYSTEM_PROMPT),
#         *_FEWSHOT_EXAMPLE,
#         user_struct(review_text),
#     ]
#     out = query_llm(msgs, json_schema=_SCHEMA)
#     # batch_run_llm may return a dict with "content" or a raw string; handle both.
#     # out = out.get("content", out) if isinstance(out, dict) else out
#     data = json.loads(out)
#     return data

def extract_tags_batch(batch_obj):
    messages_list = []
    for review_text in batch_obj:
        messages = [
            system_struct(_SYSTEM_PROMPT),
            *_FEWSHOT_EXAMPLE,
            user_struct(review_text),
        ]
        messages_list.append(messages)
    results = batch_run_llm(messages_list, json_schema=_SCHEMA, verbose=True)
    results = [json.loads(raw) for raw in results]
    return results