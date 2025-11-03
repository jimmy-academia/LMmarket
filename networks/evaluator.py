import json
import logging
from api import batch_run_llm, user_struct, system_struct

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
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "explanation", "evidence"],
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
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
    return results
