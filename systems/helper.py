# network.helper
import json
import logging
from api import batch_run_llm, query_llm
from api import user_struct, system_struct, assistant_struct, developer_struct

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

LLM_JUDGE_SCHEMA = {
    "name": "llm_judge_schema",
    "schema": {
        "type": "object",
        "properties": {
            "aspect_status": {
                "type": "string",
                "enum": ["positive", "negative", "inconclusive"],
                "description": "Overall judgment for the given aspect."
            },
            "is_conclusive": {
                "type": "boolean",
                "description": "True if the review gives clear evidence for or against the aspect."
            },
            "is_positive": {
                "type": "boolean",
                "description": "True if the aspect_status is positive; false otherwise."
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence level of the judgment, from 0.0 to 1.0."
            },
            "reasoning": {
                "type": "string",
                "maxLength": 300,
                "description": "Short natural-language reasoning (≤300 chars) explaining why the decision was made."
            }
        },
        "required": ["aspect_status", "is_conclusive", "is_positive", "confidence", "reasoning"],
        "additionalProperties": False
        }
    }


def _llm_judge_item_prompt(aspect, aspect_type, query, review_text, snippet):
    """
    Evaluate a single review for one aspect.

    Returns dict:
      {
        "aspect_status": str,
        "is_conclusive": bool,
        "is_positive": bool,
        "confidence": float,
        # "new_keywords": str
      }
    """
    rubric = {
        "ontological": "Judge whether the review confirms or denies that the item IS of this category.",
        "functional": "Judge whether the review confirms the item HAS or DOES this feature well.",
        "teleological": "Judge whether the item IS SUITABLE or UNSUITABLE for this purpose."
    }[aspect_type]

    system_prompt = (
        "You are a precise review judge.\n"
        f"This is a {aspect_type.upper()} aspect. {rubric}\n"
        "Decide one of: positive, negative, inconclusive.\n"
        "Provide confidence score and briefly explain your reasoning in plain text (1–2 sentences).\n"
        "Return JSON: {aspect_status, is_conclusive, is_positive, confidence, reasoning}."
    )
    messages = [
        system_struct(system_prompt),
        # user_struct(f"ASPECT: {aspect}\nQUERY: {query}\nSNIPPET:\n{snippet}")
        user_struct(f"ASPECT: {aspect}\nQUERY: {query}\nREVIEW:\n{review_text}")
    ]
    return messages

def _llm_judge_batch(aspect, aspect_type, query, batch_obj):
    messages_list = []
    for obj in batch_obj:
        review_id, score, item_id, text, snippet = obj 
        messages = _llm_judge_item_prompt(aspect, aspect_type, query, text, snippet)
        messages_list.append(messages)
    results = batch_run_llm(messages_list, json_schema=LLM_JUDGE_SCHEMA, verbose=True)
    results = [json.loads(raw) for raw in results]
    return results

SYSTEM_MESSAGE_score = (
    "You are a precise and impartial evaluator. "
    "Score how well a REVIEW TEXT supports one ASPECT of a multi-aspect user query. "
    "Use only explicit textual evidence (allow reasonable synonyms). "
    "Ignore other aspects. Do not assume facts not in the review.\n\n"
    "Scoring (0–1):\n"
    "  1.00  = Strongly positive evidence (clear, emphatic praise)\n"
    "  0.80  = Moderately positive (favorable, specific but brief)\n"
    "  0.60  = Slightly positive / mixed but leans positive\n"
    "  0.50  = Neutral or aspect not mentioned / irrelevant\n"
    "  0.40  = Slightly negative / mixed but leans negative\n"
    "  0.20  = Moderately negative (clear criticism)\n"
    "  0.00  = Strongly negative (explicit complaint or clear failure)\n\n"
    "Output ONLY valid JSON with fields:\n"
    "{ 'score': float, 'explanation': str, 'evidence': str }\n"
    "- 'score' in [0.0, 1.0], rounded to two decimals.\n"
    "- 'evidence' is a short quote (<= 12 words) copied verbatim from the review.\n"
)

USER_TEMPLATE = (
    "Query: {query}\n"
    "Aspect: {aspect}\n"
    "Review text: {review}\n\n"
    "Judge ONLY the given aspect. Return JSON."
)

LLM_SCORE_SCHEMA = {
    "name": "llm_score_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "explanation", "evidence"],
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "multipleOf": 0.1,
                "description": "Normalized score in [0,1]."
            },
            "explanation": {
                "type": "string",
                "minLength": 4,
                "maxLength": 300,
                "description": "Brief rationale tied to the evidence."
            },
            "evidence": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "description": "Short verbatim quote from the review."
            }
        }
    }
}


def _llm_score_prompt(aspect, query, text):
    messages = [
        system_struct(SYSTEM_MESSAGE_score),
        user_struct(USER_TEMPLATE.format(query=query, aspect=aspect, review=text))
    ]
    return messages


def _llm_score_batch(aspect, query, texts):
    messages_list = []
    for text in texts:
        messages = _llm_score_prompt(aspect, query, text)
        messages_list.append(messages)
    raw_results = batch_run_llm(messages_list, json_schema=LLM_SCORE_SCHEMA, verbose=True)
    results = [json.loads(raw) for raw in raw_results]
    # for result in results:
    #     if 'score' not in result:
    #         from debug import check
    #         check()
    return results
