from api import query_llm, user_struct, system_struct, assistant_struct

system_prompt = """You extract concise aspect phrases from a single user request.
Output only a comma-separated list of phrases, nothing else.
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

user_temp = """Text: {request_text}
Output only the comma-separated list."""

few_shot_examples = [
    (
        "Looking for an open late ramen spot, cheap, with counter seating; no long wait.",
        "open late, ramen spot, cheap, counter seating, no long wait",
    ),
    (
        "A quiet bakery with strong wifi and plenty of power outlets; vegan pastries preferred.",
        "quiet, bakery, strong wifi, plenty of power outlets, vegan pastries",
    ),
    (
        "Find a lively sports bar with big screens, large beer selection, and no smoking indoors.",
        "lively, sports bar, big screens, large beer selection, no smoking indoors",
    ),
]

ASPECT_SCHEMA = {
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
    message = [system_struct(system_prompt), assistant_struct(few_shot_examples), user_struct(user_temp)]

    for inp, out in few_shot_examples:
        messages.append(user_struct(user_temp.format(request_text=inp)))
        messages.append(assistant_struct(out))

    messages.append(user_struct(user_temp.format(request_text=request)))

    response = query_llm(message, json_schema=ASPECT_SCHEMA, use_json=True)
    return response
