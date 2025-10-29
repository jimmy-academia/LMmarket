# network.helper
import logging
from api import batch_run_llm, query_llm
from api import user_struct, system_struct, assistant_struct

def _decompose_aspect(query):
    messages = [
        system_struct(
            "You are a text decomposition assistant. "
            "Your task is to extract key aspects from a natural-language request and output a comma-separated list. "
            "Follow these rules strictly:\n"
            "1) TYPE vs FEATURES: Identify the domain/type (place/product/service). Emit the type as ONE item; emit each feature as its OWN item. Each list item must express a single idea. If a span contains a type plus modifiers, split them.\n"
            "2) SPLIT CONNECTORS: Break on connectors such as 'with', 'and', 'that', 'which', 'for', 'to', 'near', 'in', 'featuring', 'offering'.\n"
            "3) PURPOSE vs DURATION: Separate purpose/usage (e.g., suitability) from time/duration/length-of-stay into distinct items.\n"
            "4) PHRASE LENGTH: Keep items short (≤ 3 words when possible), but preserve meaningful noun phrases.\n"
            "5) FORMAT: lowercase, no duplicates, no filler words. Output only the comma-separated list—no labels or extra text."),
        user_struct("Find a romantic Italian restaurant with candlelight and outdoor seating, perfect for a date night."),
        assistant_struct("romantic, Italian restaurant, has candlelight, has outdoor seating, perfect for a date night"),
        user_struct("Looking for a family-friendly restaurant that serves vegetarian food and has a play area for kids."),
        assistant_struct("family-friendly, has vegetarian food, has play area for kids"),
        user_struct("Recommend a sushi spot with fast service and reasonable prices"),
        assistant_struct("sushi place, fast service, reasonable prices"),
        user_struct(query)
    ]
    output = query_llm(messages)
    return output


SYSTEM_MESSAGE = (
    "You classify ONE aspect and propose starter keywords for review retrieval.\n"
    "Input arrives as TWO user messages:\n"
    "  • First user message: the FULL ORIGINAL QUERY (context only)\n"
    "  • Second user message: the ASPECT to classify\n\n"
    "Use the full query ONLY to disambiguate domain or meaning if needed.\n"
    "Otherwise, reason directly from the aspect.\n\n"
    "Steps:\n"
    "1) Assign `aspect_type` as one of: 'ontological', 'functional', 'teleological'.\n"
    "   - ontological → what the item IS (category/identity)\n"
    "   - functional → what the item HAS/DOES (features/properties/behaviors)\n"
    "   - teleological → what the item is FOR (purpose/suitability/intended use)\n"
    "2) Suggest 3–6 concise starter keywords (lowercase, natural language).\n"
    "   - Do NOT include the aspect itself (the system will add it later).\n"
    "   - No explanations—return ONLY JSON.\n"
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "aspect_type": {
            "type": "string",
            "enum": ["ontological", "functional", "teleological"]
        },
        "starter_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6
        }
    },
    "required": ["aspect_type", "starter_keywords"],
    "additionalProperties": False
}

def _generate_aspect_info(aspect_list, query):
    messages_batch = []
    for aspect in aspect_list:
        messages_batch.append([
            system_struct(SYSTEM_MESSAGE),
            user_struct(query),                  # context first
            user_struct(f"ASPECT:\n{aspect}")    # explicit aspect second
        ])
        break

    raw_results = batch_run_llm(messages_batch, use_json=True, json_schema=OUTPUT_SCHEMA)
    from debug import check
    check()
    
    aspect_info_list = []
    for aspect, raw in zip(aspect_list, raw_results):
        try:
            result = json.loads(raw)
        except Exception:
            result = {"aspect_type": "functional", "starter_keywords": []}
        result["aspect"] = aspect
        result["starter_keywords"].insert(0, aspect)
        aspect_info_list.append(result)

    return aspect_info_list
