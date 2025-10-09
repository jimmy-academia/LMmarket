from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

def tokenize_for_spell(text):
    if not text:
        return []
    buffer = []
    for ch in str(text):
        if ch.isalpha():
            buffer.append(ch.lower())
        elif ch.isdigit():
            buffer.append(ch)
        else:
            buffer.append(" ")
    return [tok for tok in "".join(buffer).split() if tok]


def build_symspell(reviews):
    if not reviews:
        return None
    counts = {}
    items = reviews.values() if isinstance(reviews, dict) else reviews
    for review in tqdm(items, ncols=88, desc="[symspell] build"):
        text = review.get("text")
        for token in tokenize_for_spell(text):
            counts[token] = counts.get(token, 0) + 1
    if not counts:
        return None
    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    for token, freq in counts.items():
        symspell.create_dictionary_entry(token, freq)
    return symspell


def correct_spelling(symspell, text):
    if not text or not symspell:
        return text
    words = text.split()
    if not words:
        return text
    corrected = []
    for word in words:
        matches = symspell.lookup(word, Verbosity.CLOSEST)
        corrected.append(matches[0].term if matches else word)
    return " ".join(corrected)

def fix_review(symspell, reviews):
    fixed_reviews = {}
    for rid, rev in tqdm(reviews.items(), ncols=88, desc="[base] apply symspell"):
        fixed_reviews[rid] = {**rev, "text": correct_spelling(symspell, rev.get("text"))}
    return fixed_reviews

