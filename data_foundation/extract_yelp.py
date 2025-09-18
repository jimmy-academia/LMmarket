# foundation/feature_extract.py
import re
from llm import run_llm_batch, safe_json_parse

# ---- config ----
MODEL = "gpt-5-nano"
TEMPERATURE = 0.0
NUM_WORKERS = 8
VERBOSE = True
"""
Notes gpt-5-nano gpt-5-mini
"""

def extract_yelp_features(reviews):
    """
    reviews : list of dict where {'review_id': str, 'text': str}
    output
    - rid_to_fids: dict map rid => list of feature_ids (fid = {rid:name_norm})
    """
    prompts = [build_prompt(r["text"]) for r in reviews]
    raw_outputs = run_llm_batch(prompts, model=MODEL, temperature=TEMPERATURE, num_workers=NUM_WORKERS, verbose=VERBOSE)
    parsed_outputs = [safe_json_parse(out) for out in raw_outputs]
    
    combined_outputs = []
    rid_to_fids = {}
    for r, out in zip(reviews, parsed_outputs):
        rid = r["review_id"]
        names = [normalize_feature_name(f.get("name", "")) for f in out.get("features", [])]
        rid_to_fids[rid] = [f"{rid}:{n}" for n in names if n]

        out['review'] = r['text']
        combined_outputs.append(out)

    return {'rid_to_fids': rid_to_fids, 'results': combined_outputs}


def normalize_feature_name(name):
    """
    Normalize feature name:
    - lowercase
    - strip leading/trailing spaces
    - collapse spaces -> '_'
    - remove punctuation except '_' and '-'
    """
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", "_", s)              # spaces to underscore
    s = re.sub(r"[^\w\-]", "", s)           # drop non-word chars except dash
    return s

### ==== human in the loop prompt optimize ===

def build_prompt(text, domain="restaurant"):
    return f"""You analyze a {domain} review and extract a SMALL set of FEATURES the reviewer evaluated.

INPUT REVIEW:
{text}

Return ONLY JSON in exactly this shape (no extra text). 
Every feature MUST include all fields: name, definition, sentiment, reason, quotes.

Example (single feature shown):

{{
  "features": [
    {{
      "name": "service",
      "definition": "attentiveness and helpfulness of staff; positive means attentive, negative means unhelpful.",
      "sentiment": 0.80,
      "reason": "reviewer praised staff as fast and friendly, showing clear positive service quality.",
      "quotes": ["fast and friendly service"]
    }}
  ]
}}

REQUIREMENTS

1) FEATURE NAMES
- lowercase, space-separated words; neutral and reusable (e.g., "food quality", "service", "price value", "wait time", 
  "portion size", "menu variety", "beverage program", "ambience", "seating", "location", "consistency across visits", 
  "takeout experience").
- Be specific but not overly abstract:
  • If the review clearly refers to a particular dietary restriction, name it directly 
    (e.g., "gluten free option availability", "vegan option availability", "vegetarian option availability").
  • Avoid vague catch-alls like "dietary options" unless the review is truly generic.
- Do NOT create features for single dishes or one-off events (e.g., no "garlic bread", "ponchartrain", "burger showdown"); 
  mention those only in quotes/reason under a broader feature like "food quality" or "menu variety".
- Synonyms DO NOT need to be collapsed. If distinct aspects are clearly discussed (e.g., "service", "service speed", 
  "staff friendliness"), you may include them separately.

2) DEFINITIONS
- One concise sentence describing what the feature refers to and what positive vs negative means. 
- Neutral tone; no second person.

3) SENTIMENT
- Float in [-1.00, 1.00], two decimals. 
- Strong praise ≈ 0.70–1.00; mild praise ≈ 0.20–0.60; neutral ≈ 0.00; mild criticism ≈ -0.20 to -0.60; strong criticism ≈ -0.70 to -1.00.
- If mixed, produce a single averaged score and explain the tradeoff in "reason".

4) REASON
- 1–2 sentences explaining why you chose this feature and score, explicitly referencing what the reviewer said 
  (summarize the evidence and tradeoffs).

5) QUOTES
- Provide 1–3 short verbatim excerpts from the review that best support the score (≤140 chars each). No offsets.

6) SCOPE LIMIT
- Maximum 8 features per review; prioritize the strongest signals. Avoid redundant near-duplicates.

7) NAME STYLE ENFORCEMENT
- Always emit feature names as lowercase, space-separated words in the JSON, even if internal IDs use snake_case.

8) SANITY
- No hallucinations; use only the review text. 
- American spelling. 
- No empty arrays or nulls.
"""



def build_prompt3(text, domain="restaurant"):
    return f"""You analyze a {domain} review and extract a SMALL set of FEATURES the reviewer evaluated.

INPUT REVIEW:
{text}

Return ONLY JSON in exactly this shape (no extra text):

{{
  "features": [
    {{
      "name": "string",
      "definition": "string",
      "sentiment": 0.00,
      "reason": "string",
      "quotes": ["string"]
    }}
  ]
}}

REQUIREMENTS

1) FEATURE NAMES
- lowercase, space-separated words; neutral and reusable (e.g., "food quality", "service", "price value", "wait time", 
  "portion size", "menu variety", "beverage program", "ambience", "seating", "location", "consistency across visits", 
  "takeout experience").
- Be specific but not overly abstract:
  • If the review clearly refers to a particular dietary restriction, name it directly 
    (e.g., "gluten option availability", "vegan option availability", "vegetarian option availability"). 
  • Avoid vague terms like "dietary options".
- Do NOT create features for single dishes or one-off events (e.g., no "garlic bread", "ponchartrain", "burger showdown"); 
  mention those only in quotes/reason under a broader feature like "food quality" or "menu variety".
- Do NOT collapse synonyms; keep distinct aspects separate if the text supports them (e.g., "service", "service speed", 
  "staff friendliness" can all appear if each is clearly discussed).

2) DEFINITIONS
- One concise sentence describing what the feature refers to and what positive vs negative means.

3) SENTIMENT
- Float in [-1.00, 1.00], two decimals. 
- Strong praise ≈ 0.70–1.00; mild praise ≈ 0.20–0.60; neutral ≈ 0.00; mild criticism ≈ -0.20 to -0.60; strong criticism ≈ -0.70 to -1.00.
- If mixed, produce a single averaged score and explain the tradeoff in "reason".

4) REASON
- 1–2 sentences explaining why you chose this feature and score, explicitly referencing the reviewer’s statements.

5) QUOTES
- Provide 1–3 short verbatim excerpts from the review that best support the score (≤140 chars each). No offsets.

6) SCOPE LIMIT
- Maximum 8 features per review; prioritize the strongest signals. Avoid redundant near-duplicates.

7) SANITY
- No hallucinations; use only the review text. 
- American spelling. 
- No empty arrays or nulls.
"""




def build_prompt2(text, domain="restaurant"):
    return f"""You analyze a {domain} review and extract a SMALL set of abstract, reusable FEATURES the reviewer evaluated.

INPUT REVIEW (for offset calculations):
{text}

OUTPUT: Return ONLY JSON in this exact shape (no extra text):

{{
  "features": [
    {{
      "name": "string",
      "definition": "string",
      "sentiment": 0.00,
      "polarity": "positive|negative|neutral|mixed",
      "evidence": [{{"text": "string", "start": 0, "end": 0}}]
    }}
  ]
}}

STRICT RULES

1) FEATURE NAMES
- lowercase, space-separated, letters only (no slashes, underscores, hyphens, punctuation, or emojis).
- abstract, reusable concepts (e.g., "service", "food quality", "price value", "wait time", "portion size", "menu variety", "dietary options", "beverage program", "ambience", "seating", "noise level", "cleanliness", "location", "consistency across visits", "takeout experience").
- DO NOT create features for specific items/dishes, events, or branded programs (e.g., not "ponchartrain", "garlic bread", "burger showdown", "tour availability"). Instead, attach those as evidence for a broader feature:
  - dish/item → "food quality" (and optionally "menu variety" or "specials")
  - live music, tours, events → "ambience" or "experience extras"
- Collapse synonyms into the same canonical name:
  - "price/value", "pricing", "overpriced" → "price value"
  - "service quality" → "service"
  - "staff friendliness" → "service"
  - "service speed" → "wait time" (for order speed) OR "service" (for attentiveness)
  - "gluten-free options", "dietary needs" → "dietary options"

2) DEFINITIONS
- One concise sentence that states what the feature refers to AND how to interpret sentiment (what positive vs negative means). Stay generic; do not restate the review.

3) SENTIMENT
- Float in [-1.00, 1.00], two decimals.
- Map language cues to magnitude: strong praise ~ 0.70–1.00; mild praise ~ 0.20–0.60; neutral/ambiguous ~ 0.00; mild criticism ~ -0.20 to -0.60; strong criticism ~ -0.70 to -1.00.
- If both praise and criticism are present for the same feature, average into a single score and set polarity to "mixed".

4) POLARITY
- "positive" (> 0.05), "negative" (< -0.05), "neutral" (between -0.05 and 0.05 with insufficient signal), or "mixed" (clear conflicting signals).

5) EVIDENCE
- Provide 1–3 direct snippets from the review supporting the score.
- Include character offsets [start, end) for each snippet within the given review text.
- Use the most diagnostic phrases; keep snippets short.

6) SCOPE & LIMIT
- Return at most 8 features per review, prioritizing the strongest signals.
- Always deduplicate and prefer broader features over micro-features.
- If the review references multiple visits, you may include "consistency across visits" only if explicitly discussed.

7) SANITY
- No hallucinations. Do not infer hours, policies, or facts that aren’t stated.
- American spelling. Trim whitespace. No empty arrays or nulls.
"""

def build_prompt1(text, domain="restaurant"):
    return f"""You analyze a {domain} review and extract the distinct feature concepts the user evaluated.

Rules:
- Use short, lowercase, neutral feature names (e.g., "wait time", not "long wait")
- Provide a one-sentence abstract definition (what the feature refers to; note what positive vs negative means)
- Return ONLY JSON in the exact shape below. Do not add extra text.

Exact JSON:
{{
  "features": [
    {{"name": "string", "definition": "string", "sentiment": 0.0}}
  ]
}}

Review:
{text.strip()}"""


