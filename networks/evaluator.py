import json
from api import batch_run_llm, user_struct, system_struct

SYSTEM_MESSAGE_score = (
    "You are a precise review evaluator. "
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
    "Provide performnace score, briefly explain your reasoning in plain text (1–2 sentences), and provide evidence excerpt from review."
    "Return JSON: {score, explanation, evidence}\n"
)

LLM_SCORE_SCHEMA = {
    "name": "llm_score_schema",
    "schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Performance score of review on aspect, from 0.0 to 1.0."
            },
            "explanation": {
                "type": "string",
                "maxLength": 300,
                "description": "Brief rationale tied to the evidence."
            },
            "evidence": {
                "type": "string",
                "maxLength": 200,
                "description": "Short verbatim quote from the review."
            }
        },
        "required": ["score", "explanation", "evidence"],
        "additionalProperties": False
    }
}


def _llm_score_prompt(aspect, query, text):
    messages = [
        system_struct(SYSTEM_MESSAGE_score),
        user_struct(f"ASPECT: {aspect}\nQUERY: {query}\nREVIEW:\n{text}")
    ]
    return messages


def _llm_score_batch(aspect, query, review_texts):
    messages_list = []
    for text in review_texts:
        messages = _llm_score_prompt(aspect, query, text)
        messages_list.append(messages)
    raw_results = batch_run_llm(messages_list, json_schema=LLM_SCORE_SCHEMA, verbose=True)
    results = [json.loads(raw) for raw in raw_results]
    return results
