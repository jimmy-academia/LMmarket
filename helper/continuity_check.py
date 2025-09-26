from pathlib import Path
from difflib import SequenceMatcher
import re
import unicodedata
from utils import loadj

# --------------------------- CONFIG ---------------------------------
# Allow small typos in contiguous matching before we consider diffs.
# Example: "reasonable" vs "reasoanble" (single transposition) => match.
FUZZY_MAX_REL_ERR = 0.03       # ≤ 3% of excerpt length
FUZZY_MAX_ABS_ERR = 2          # but never less than 2 edits
# --------------------------------------------------------------------


def review_id(review):
    for key in ("review_id", "id", "reviewId"):
        value = review.get(key)
        if value is not None:
            return value
    return None


def normalize_for_fuzzy(s):
    """
    Lightweight normalization for fuzzy matching:
      - Unicode NFKC
      - lowercase
      - collapse all whitespace runs to a single space
    """
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fuzzy_contains(text, excerpt, rel=FUZZY_MAX_REL_ERR, abs_err=FUZZY_MAX_ABS_ERR):
    """
    Damerau-Levenshtein (optimal string alignment) approximate *substring* match.
    Returns (start_norm, end_norm, distance) over the *normalized* text if within threshold,
    else None. This is used only to decide 'continuous' vs not (no spans reported).
    """
    t = normalize_for_fuzzy(text)
    p = normalize_for_fuzzy(excerpt)
    if not p:
        return None

    # Exact match on normalized text? (fast path)
    pos = t.find(p)
    if pos != -1:
        return (pos, pos + len(p), 0)

    max_dist = max(abs_err, int(round(rel * len(p))))
    m, n = len(p), len(t)

    # Edit DP for approximate substring:
    # dp[0][j] = 0 to allow free start anywhere in text (substring).
    prev2 = None
    prev = [0] * (n + 1)

    for i in range(1, m + 1):
        cur = [i] + [0] * n
        pi = p[i - 1]
        # Optional micro-pruning window could be added; omitted for clarity.
        for j in range(1, n + 1):
            cost = 0 if pi == t[j - 1] else 1
            ins = cur[j - 1] + 1         # insertion in pattern
            dele = prev[j] + 1           # deletion from pattern
            sub = prev[j - 1] + cost     # substitution / match
            val = ins if ins < dele and ins < sub else (dele if dele < sub else sub)

            # OSA transposition (counts as 1)
            if prev2 is not None and i > 1 and j > 1 and p[i - 1] == t[j - 2] and p[i - 2] == t[j - 1]:
                trans = prev2[j - 2] + 1
                if trans < val:
                    val = trans

            cur[j] = val
        prev2, prev = prev, cur

    # Best end position for pattern anywhere in text:
    end_j = min(range(n + 1), key=lambda j: prev[j])
    dist = prev[end_j]
    if dist <= max_dist:
        approx_start = max(0, end_j - m)
        return (approx_start, end_j, dist)
    return None


def match_blocks(text, excerpt):
    """
    Align excerpt (a) to text (b) with difflib, collecting equal-span locations.
    Critical fix: disable autojunk for long NL strings to avoid bogus splits.
    """
    matcher = SequenceMatcher(None, excerpt, text, autojunk=False)
    spans = []
    extras = []
    covered = 0
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            spans.append((b0, b1))
            covered += a1 - a0
        else:
            fragment = excerpt[a0:a1].strip()
            if fragment:
                extras.append(fragment)
    return spans, covered, extras


def summarize(text, spans, limit=3):
    notes = []
    for start, stop in spans[:limit]:
        snippet = text[start:stop]
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        notes.append((start, stop, snippet))
    return notes


def gaps_are_whitespace(text, spans):
    """
    Treat as 'continuous' if only whitespace separates equal spans.
    """
    if not spans:
        return False
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        gap = text[e1:s2]
        if gap and not gap.isspace():
            return False
    return True


def find_ws_insensitive(text, excerpt):
    """
    Contiguous check tolerant to whitespace runs.
    Returns (start, end) in original text if found; else None.
    """
    if not excerpt:
        return None
    pat = re.escape(excerpt)
    pat = re.sub(r"\s+", r"\\s+", pat)
    m = re.search(pat, text)
    return m.span() if m else None


def examine(path):
    data = loadj(path)
    stats = {"reviews": 0, "units": 0, "continuous": 0}
    issues = []
    break_examples = []
    extra_examples = []

    for review in data:
        text = review.get("text")
        units = review.get("opinion_units")
        if not isinstance(text, str) or not units:
            continue
        stats["reviews"] += 1

        for idx, unit in enumerate(units):
            excerpt = unit.get("excerpt")
            excerpt_text = excerpt.strip() if isinstance(excerpt, str) else ""
            stats["units"] += 1

            if not excerpt_text:
                issues.append({
                    "review": review_id(review),
                    "index": idx,
                    "status": "empty",
                    "snippets": []
                })
                continue

            # 1) Exact contiguous
            direct = text.find(excerpt_text)
            if direct != -1:
                stats["continuous"] += 1
                continue

            # 2) Whitespace-insensitive contiguous
            ws_span = find_ws_insensitive(text, excerpt_text)
            if ws_span is not None:
                stats["continuous"] += 1
                continue

            # 3) Fuzzy contiguous (typo-tolerant, incl. transpositions)
            fz = fuzzy_contains(text, excerpt_text)
            if fz is not None:
                # Consider it continuous if within fuzzy threshold
                stats["continuous"] += 1
                continue

            # 4) General alignment (diff, extras, partial/discontinuous)
            spans, covered, extras = match_blocks(text, excerpt_text)

            if covered == 0:
                entry = {
                    "review": '\n\n>>>\n' + text + '\n<<<\n\n',
                    "index": idx,
                    "status": "no-match",
                    "excerpt": excerpt_text,
                    "snippets": [],
                    "extras": extras
                }
                issues.append(entry)
                if len(break_examples) < 3:
                    break_examples.append(entry)

                if extras and len(extra_examples) < 3:
                    cleaned = excerpt_text
                    for fragment in extras:
                        cleaned = cleaned.replace(fragment, "", 1)
                    cleaned = cleaned.strip()
                    # Try to locate remaining cleaned text (exact or ws-insensitive)
                    location = text.find(cleaned) if cleaned else -1
                    if location == -1 and cleaned:
                        ws_span2 = find_ws_insensitive(text, cleaned)
                        location = ws_span2[0] if ws_span2 else -1
                    extra_examples.append((entry, cleaned, location))
                continue

            if covered == len(excerpt_text):
                # Fully covered by equal blocks; whitespace-only gaps => continuous
                if len(spans) == 1 or gaps_are_whitespace(text, spans):
                    stats["continuous"] += 1
                    continue
                status = "discontinuous"
            else:
                status = "partial"

            # If nearly full coverage and extras are tiny (e.g., 1–2 chars), suppress noise
            # (safety belt; fuzzy step should already catch true typos)
            if len(excerpt_text) - covered <= 2 and all(len(x) <= 2 for x in extras):
                stats["continuous"] += 1
                continue

            entry = {
                "review": '\n\n>>>\n' + text + '\n<<<\n\n',
                "index": idx,
                "status": status,
                "excerpt": excerpt_text,
                "snippets": summarize(text, spans),
                "extras": extras
            }
            issues.append(entry)
            if len(break_examples) < 3:
                break_examples.append(entry)

            if extras and len(extra_examples) < 3:
                cleaned = excerpt_text
                for fragment in extras:
                    cleaned = cleaned.replace(fragment, "", 1)
                cleaned = cleaned.strip()
                location = text.find(cleaned) if cleaned else -1
                if location == -1 and cleaned:
                    ws_span2 = find_ws_insensitive(text, cleaned)
                    location = ws_span2[0] if ws_span2 else -1
                extra_examples.append((entry, cleaned, location))

    # ---- Reporting ----
    print(f"reviews_with_units: {stats['reviews']}")
    print(f"total_opinion_units: {stats['units']}")
    print(f"continuous_excerpts: {stats['continuous']}")
    print(f"with_breaks: {len(issues)}")

    for entry in issues:
        print(f"- review={entry.get('review')} unit={entry['index']} status={entry['status']}")
        excerpt = entry.get("excerpt")
        if excerpt:
            print(f"  excerpt: {excerpt}")
        for start, stop, snippet in entry["snippets"]:
            print(f"    span=({start}, {stop}) text={snippet}")
        extras_list = entry.get("extras")
        if extras_list:
            joined = "; ".join(extras_list)
            print(f"    extra_words: {joined}")

    if not stats["units"]:
        print("Conclusion: no opinion units available for analysis.")
        return

    breaks = len(issues)
    extras_count = sum(1 for entry in issues if entry.get("extras"))
    if breaks:
        if extras_count:
            print(
                f"Conclusion: {breaks} of {stats['units']} units break continuity; "
                f"{extras_count} contain extra words absent from the source text."
            )
        else:
            print(
                f"Conclusion: {breaks} of {stats['units']} units break continuity "
                "without introducing new words."
            )
    else:
        print(
            "Conclusion: all opinion units are continuous and rely solely on "
            "words present in the reviews."
        )

    if break_examples:
        print("\n\nExamples: break continuity")
        for entry in break_examples:
            print(f"  review={entry.get('review')} unit={entry['index']} status={entry['status']}")
            print(f"    excerpt: {entry.get('excerpt')}")
            for start, stop, snippet in entry["snippets"]:
                print(f"      span=({start}, {stop}) text={snippet}")
        if len(break_examples) < 3:
            print(f"  only {len(break_examples)} example(s) available")

    if extra_examples:
        print("\n\nExamples: extra words absent from the source text")
        for entry, cleaned, location in extra_examples:
            extras_list = entry.get("extras")
            joined = "; ".join(extras_list) if extras_list else ""
            print(f"  review={entry.get('review')} unit={entry['index']} extras={joined}")
            print(f"    excerpt: {entry.get('excerpt')}")
            print(f"    without extras: {cleaned or '[empty]'}")
            if location != -1:
                print(f"    matches_at: {location}")
            else:
                print("    matches_at: not found")
        if len(extra_examples) < 3:
            print(f"  only {len(extra_examples)} example(s) available")

# --- Add this helper near the bottom of the file --------------------------------
def list_five_reviews_with_units(path, k=5):
    """
    Print up to k reviews that contain non-empty 'opinion_units'.
    For each review: print only the raw review text (delimited by >>> / <<<)
    followed by each opinion unit's 'excerpt' (or the unit as compact JSON if no excerpt).
    """
    data = loadj(path)
    shown = 0
    for review in data:
        text = review.get("text")
        units = review.get("opinion_units")
        if not isinstance(text, str) or not units:
            continue

        # Only keep units with some content
        nonempty_units = [u for u in units if isinstance(u, dict) and (u.get("excerpt") or "").strip()]
        if not nonempty_units:
            continue

        print(">>>")
        print(text.strip())
        print("<<<")
        for u in nonempty_units:
            print(u)
            print('---')
            # ex = (u.get("excerpt") or "").strip()
            # if ex:
            #     print(f"- {ex}")
            # else:
            #     # Fallback if no excerpt string is present
            #     try:
            #         import json
            #         print("- " + json.dumps(u, ensure_ascii=False))
            #     except Exception:
            #         print("- " + str(u))
        print()  # blank line between reviews
        shown += 1
        if shown >= k:
            break

    if shown == 0:
        print("No reviews with opinion_units found.")
# ---------------------------------------------------------------------------------


def main():
    # Allow CLI override while preserving your default path.
    import sys
    root = Path(__file__).resolve().parent
    default_target = root / "cache" / "rich_review" / "yelp_saint_louis.json"

    # Parse args: first non-flag is treated as a path; flags start with '-'
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    target = Path(args[0]).resolve() if args else default_target

    if not target.exists():
        print(f"missing: {target}")
        return

    # If you want the quick sampler, call with:  python script.py --list5
    list_five_reviews_with_units(target, k=5)

    # examine(target)


if __name__ == "__main__":
    main()
