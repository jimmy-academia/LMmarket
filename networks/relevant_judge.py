
import json
from api import batch_run_llm, user_struct, system_struct

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
