# self.item_aspect_status = {}
#         for item in self.items:
#             item_id = item['item_id']
#             self.item_aspect_status[item_id] = {aspect:{} for aspect in aspect_list}
#         self.review_aspect_labels = {}



LLM_ASPECT_LABEL_PROMPT = """
(2) score its SENTIMENT PERFORMANCE for that aspect.
    - Sentiment PERFORMANCE is how good/bad the aspect seems, not overall review mood.
    - Use only the segment text; do not invent facts.
    - Return STRICT JSON that matches the example structure. No extra text.

Relevance rubric
    - 1 = Mentions or clearly implies the aspect with actionable signal.
    - 0 = Off-topic, generic praise/complaint without aspect tie, or too vague.

Sentiment PERFORMANCE rubric (for relevant = 1)
    - polarity ∈ {-1,0,1}: negative / mixed or neutral / positive.
    - score ∈ [0,1]: strength of the aspect’s performance implied by the segment.
        0.00–0.15: disastrous; 0.16–0.35: poor; 0.36–0.64: mixed/ok;
        0.65–0.84: good; 0.85–1.00: excellent.
    - If mixed (“good lighting but seats uncomfortable”), pick the dominant signal; if truly balanced, polarity=0 and score≈0.5.

Output
    - Return ONLY a JSON object following the example format.
    - No explanations or text outside JSON.

ASPECT: "{aspect_sentence}"

Segments:
{segments_json}

Example output format:
{
    "aspect": "{aspect_sentence}",
    "version": "lmmarket.aspect-v1",
    "results": [
        {
            "id": 101,
            "relevant": 1,
            "sentiment": {
                "polarity": 1,
                "score": 0.9
            },
            "evidence": ["no music", "read for hours"],
            "quality_flags": []
        },
        {
            "id": 102,
            "relevant": 0,
            "sentiment": null,
            "evidence": [],
            "quality_flags": ["low_signal"]
        }
    ]
}
"""


LLM_ASPECT_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "aspect": {"type": "string"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": ["string", "integer"]},
                    "relevant": {"type": "integer", "enum": [0, 1]},
                    "rel_confidence": {
                        "type": ["number", "null"],
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "sentiment": {
                        "type": ["object", "null"],
                        "properties": {
                            "polarity": {"type": "integer", "enum": [-1, 0, 1]},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "sent_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["polarity", "score", "sent_confidence"]
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 2
                    },
                    "quality_flags": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "id", "relevant", "rel_confidence",
                    "sentiment", "evidence", "quality_flags"
                ]
            }
        }
    },
    "required": ["aspect", "results"]
}
