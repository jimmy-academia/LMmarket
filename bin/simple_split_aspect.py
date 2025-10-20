import json
from api import query_llm, user_struct, system_struct, assistant_struct

system_prompt = """You extract concise aspect phrases from a single user request.
Return ONLY strict JSON with the shape: {"aspects": [<string>, ...]}.
Rules:
- Use exact words from the input (you may drop filler like “good”, “perfect for”, “really”, “very”).
- Keep each phrase short (1–4 words).
- Preserve the order in which they appear.
- Split distinct ideas into separate phrases.
- Keep venue/type nouns (e.g., cafe, bakery, bar) as their own phrase.
- Keep negations as written (e.g., “no loud music”).
- Keep durations/time as written (e.g., “a few hours”).
- Do not add, rephrase, or infer anything not in the text.
- Output only the list (no quotes, no brackets, no prose)."""

user_temp = "Text: {request_text}\nReturn only the JSON object."

FEW_SHOT_EXAMPLES = [
    (
        "Looking for an open late ramen spot, cheap, with counter seating; no long wait.",
        {"aspects": ["open late", "ramen spot", "cheap", "counter seating", "no long wait"]},
    ),
    (
        "A quiet bakery with strong wifi and plenty of power outlets; vegan pastries preferred.",
        {"aspects": ["quiet", "bakery", "strong wifi", "plenty of power outlets", "vegan pastries"]},
    ),
    (
        "Find a lively sports bar with big screens, large beer selection, and no smoking indoors.",
        {"aspects": ["lively", "sports bar", "big screens", "large beer selection", "no smoking indoors"]},
    ),
]

_SCHEMA = {
    "name": "AspectList",
    "strict": True,  # force exact shape
    "schema": {
        "type": "object",
        "properties": {
            "aspects": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0
            }
        },
        "required": ["aspects"],
        "additionalProperties": False
    }
}

def aspect_splitter(request):
    messages = [system_struct(system_prompt)]

    for inp, out_json in FEW_SHOT_EXAMPLES:
        messages.append(user_struct(user_temp.format(request_text=inp)))
        messages.append(assistant_struct(json.dumps(out_json, ensure_ascii=False)))


    messages.append(user_struct(user_temp.format(request_text=request)))

    response = query_llm(messages, json_schema=_SCHEMA, use_json=True)
    return response
