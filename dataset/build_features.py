from collections import defaultdict
from pathlib import Path

from utils import vlog, ppause
from utils import readf
from utils import dumpj, loadj
from utils import clean_phrase
from utils import VERBOSE

from llm import query_llm
from debug import check
from functools import partial

from .ontology import Ontology

M = 10  # Max children under a root before reordering

### --- Feature + Sentiment Extraction --- ###
def extract_feature_mentions(text, ontology=None, dset = None, model="openai"):
    hints = ontology.feature_hints() if ontology else []
    hint_str = f"Known features to look for include: {', '.join(hints)}." if hints else ""

    if dset == 'yelp':
        domain = 'restaurant'
        cares = 'service, food, timing, pricing, experience, etc.'
    elif dset == 'amazon':
        domain = 'product'
        cares = 'build quality, ease of use, packaging, delivery experience, pricing, performance, compatibility, aesthetics, etc.'
    else:
        domain = 'general'
        cares = 'anything.'

    prompt = f"""
You are a careful and accurate reviewer analysis assistant.

You are analyzing a {domain} review to extract key features that the user evaluated. These features must be **generalizable** — they should be applicable to many other {domain}s, not specific to just this one case.

Follow this two-step process:

Step 1: Identify distinct **general feature concepts** described in the review — what broad aspects did the user care about that are likely relevant across most {domain}s? These could relate to {cares} — avoid naming brand-specific, item-specific, or context-specific details.

Step 2: For each general concept, provide:
- A short, lowercase phrase that neutrally names the feature (e.g. "wait time", not "long wait time" or "15-minute delay")
- A one-sentence abstract definition that clearly explains what the feature refers to in general. Briefly describe what a positive and a negative score would mean for that feature (e.g., “short wait times are positive; long delays are negative”)
- A sentiment score between -1.0 and +1.0 indicating how the feature is portrayed in this specific review

If relevant, align your phrasing with known features: {hint_str}
Otherwise, invent a **general-purpose** phrase — avoid overly specific or one-off descriptions.

Be precise. Avoid using the same phrase for unrelated meanings.

Review:
\"\"\"{text.strip()}\"\"\"

Output format (one line per feature):
feature name | definition | score (float between -1.0 and 1.0)
"""
    response = query_llm(prompt, model=model)

    results = []
    for line in response.strip().splitlines():
        if "|" in line:
            try:
                parts = line.split("|")
                phrase = clean_phrase(parts[0])
                definition = parts[1].strip()
                score = float(parts[2].strip())
                results.append((phrase, definition, score))
            except Exception:
                continue
    return results

def process_feature_data(args, ontology: Ontology, reviews):
    feature_data = []
    vlog('ontology class initialized')

    count = 0
    for r in reviews:
        review_id, text = r["review_id"], r["text"]
        vlog("\n" + "="*60)
        vlog(f"Review: {text}")
        feature_scores = extract_feature_mentions(text, ontology, args.dset)
        vlog(f"Extracted Features: {feature_scores}")
        feature_data.append([review_id, feature_scores])

        for phrase, description, score in feature_scores:
            phrase = clean_phrase(phrase)
            ontology.add_or_update_node(review_id, phrase, description, score)

        vlog("\nOntology after processing this review:")
        if VERBOSE:
            print(ontology)


        # ppause('finished 1 review')

        count += 1
        if count == 50:
            break

    return feature_data

def build_ontology_by_reviews(args, reviews):
    feature_cache_path = Path(f'cache/{args.dset}_feature_score.json')
    ontology = Ontology()

    # Already processed features using llm?
    if feature_cache_path.exists():
        feature_data = loadj(feature_cache_path)

        for review_id, feature_scores in feature_data:
            for phrase, description, score in feature_scores:
                phrase = clean_phrase(phrase)
                ontology.add_or_update_node(review_id, phrase, description, score)

    else:
        feature_data = process_feature_data(args, ontology, reviews)
        dumpj(feature_cache_path, feature_data)
      
    output_path = Path("cache") / "ontology.json"
    ontology.save(output_path)
    print(f"\nOntology saved to {output_path} with {len(ontology.nodes)} nodes.")
    return ontology

def build_ontology_by_users_and_items():
    pass