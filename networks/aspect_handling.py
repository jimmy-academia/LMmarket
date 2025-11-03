import json
import logging
from api import batch_run_llm, query_llm, user_struct, system_struct, assistant_struct

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


SYSTEM_MESSAGE_INFO = (
    "You classify ONE aspect and propose synonym-based starter keywords for review retrieval.\n"
    "You will receive two user messages:\n"
    "  • First message: the full original query (context only)\n"
    "  • Second message: the specific aspect to classify\n\n"
    "Use the full query ONLY if it helps disambiguate the aspect meaning.\n"
    "Otherwise, ignore it and reason directly from the aspect.\n\n"
    "Steps:\n"
    "1) Assign `aspect_type` as one of: 'ontological', 'functional', or 'teleological'.\n"
    "   - ontological → what the item IS (category/identity)\n"
    "   - functional → what the item HAS/DOES (features/properties/behaviors)\n"
    "   - teleological → what the item is FOR (purpose/suitability/intended use)\n"
    "2) Suggest 3–6 concise synonyms or near-synonyms that can serve as search keywords for the aspect ITSELF.\n"
    "   - Keywords must be lowercase, natural-language words or short phrases.\n"
    "   - Focus only on synonyms or strongly related expressions of the aspect.\n"
    "   - Do NOT include the aspect itself or irrelevant query context.\n"
    "   - Do NOT use overly generic terms; consider whether the keyword can be used in sparse retrieval to obtain the desired aspect.\n"
    "   - Output ONLY valid JSON matching the schema."
)

OUTPUT_SCHEMA = {
    "name": "aspect_classification",
    "schema": {
        "type": "object",
        "properties": {
            "aspect_type": {
                "type": "string",
                "enum": ["ontological", "functional", "teleological"],
                "description": "The classification of the aspect type."
            },
            "starter_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
                "description": "3–6 concise lowercase keywords describing the aspect itself."
            }
        },
        "required": ["aspect_type", "starter_keywords"],
        "additionalProperties": False
    }
}

def _generate_aspect_info(aspect_list, query):
    messages_batch = []
    for aspect in aspect_list:
        messages_batch.append([
            system_struct(SYSTEM_MESSAGE_INFO),
            user_struct(f"QUERY (context only): {query}"),
            user_struct(f"ASPECT to classify:\n{aspect}")
        ])

    raw_results = batch_run_llm(messages_batch, use_json=True, json_schema=OUTPUT_SCHEMA)
    aspect_info_list = []
    for aspect, raw in zip(aspect_list, raw_results):
        try:
            result = json.loads(raw)
        except Exception:
            logging.error('[_generate_aspect_info] json load failed')

        result["aspect"] = aspect
        result["starter_keywords"].insert(0, aspect)
        aspect_info_list.append(result)

    return aspect_info_list
