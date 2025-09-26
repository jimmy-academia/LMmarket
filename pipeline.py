"""Three-stage pipeline helpers for LMmarket."""

import json
import math
import re
from collections import defaultdict
from statistics import median

import numpy as np

_GLOBAL_RNG = np.random.default_rng(42)


def set_random_seed(seed):
    """Seed the shared RNG used across PROCESS helpers."""
    global _GLOBAL_RNG
    _GLOBAL_RNG = np.random.default_rng(int(seed) if seed is not None else 42)


def prepare_basic(args):
    """STEP-1 = PREPARE: extract Yelp data and derive geo priors."""
    if args.dset != 'yelp':
        raise ValueError('prepare_basic currently supports the Yelp dataset only.')

    root = args.dset_root / 'yelp'
    business_file = root / 'yelp_academic_dataset_business.json'
    review_file = root / 'yelp_academic_dataset_review.json'

    users = defaultdict(dict)
    items = defaultdict(dict)
    info = defaultdict(dict)
    reviews = defaultdict(list)
    geo_tracker = defaultdict(lambda: defaultdict(lambda: {'lons': [], 'lats': []}))
    biz_city = {}

    with open(business_file, encoding='utf-8') as handle:
        for line in handle:
            biz = json.loads(line)
            city = (biz.get('city') or '').strip().lower()
            if not city:
                continue
            biz_id = biz['business_id']
            coords = _ensure_coords(biz.get('longitude'), biz.get('latitude'))
            price = normalize_price(biz.get('attributes'))
            hours = normalize_hours(biz.get('hours'))
            categories = parse_categories(biz.get('categories'))
            item_entry = {
                'name': biz.get('name'),
                'address': biz.get('address'),
                'coords': coords,
                'price': price,
                'hours': hours,
                'categories': categories,
                'stars': biz.get('stars'),
                'review_count': biz.get('review_count'),
                'postal_code': biz.get('postal_code'),
                'state': biz.get('state'),
            }
            items[city][biz_id] = item_entry
            info[city][biz_id] = {
                'coords': coords,
                'price': price,
                'hours': hours,
                'categories': categories,
            }
            biz_city[biz_id] = city

    with open(review_file, encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            biz_id = row.get('business_id')
            if biz_id not in biz_city:
                continue
            city = biz_city[biz_id]
            uid = row.get('user_id')
            rid = row.get('review_id') or f'rev_{idx}'
            stars = row.get('stars')
            text = row.get('text') or ''
            ts = row.get('date')
            review_entry = {
                'review_id': rid,
                'user_id': uid,
                'item_id': biz_id,
                'stars': stars,
                'text': text,
                'ts': ts,
            }
            reviews[city].append(review_entry)
            user_entry = users[city].setdefault(uid, {'reviews': []})
            user_entry['reviews'].append(rid)
            coords = info[city].get(biz_id, {}).get('coords')
            if coords and coords[0] is not None and coords[1] is not None:
                geo_tracker[city][uid]['lons'].append(coords[0])
                geo_tracker[city][uid]['lats'].append(coords[1])

    user_geo = {}
    for city, user_map in geo_tracker.items():
        user_geo[city] = {}
        for uid, coord_map in user_map.items():
            lons = coord_map['lons']
            lats = coord_map['lats']
            if not lons or not lats:
                continue
            center = coord_median(lons, lats)
            distances = [
                haversine_km(center[0], center[1], lon, lat)
                for lon, lat in zip(lons, lats)
            ]
            med = median(distances) if distances else 0.0
            alpha = math.log(2.0) / max(med, 1e-6) if med else 0.0
            user_geo[city][uid] = {
                'center': center,
                'alpha': alpha,
                'n': len(lons),
            }
            if uid in users[city]:
                users[city][uid]['review_count'] = len(users[city][uid]['reviews'])

    for city, user_map in users.items():
        for uid, entry in user_map.items():
            entry['review_count'] = len(entry.get('reviews', []))

    prep = {
        'USERS': {city: dict(u_map) for city, u_map in users.items()},
        'ITEMS': {city: dict(i_map) for city, i_map in items.items()},
        'REVIEWS': {city: list(r_list) for city, r_list in reviews.items()},
        'INFO': {city: dict(i_map) for city, i_map in info.items()},
        'USER_GEO': user_geo,
    }
    return prep


def process_build(args, prep):
    """STEP-2 = PROCESS: segmentation, embeddings, clustering, sentiment."""
    if args.seg_model != 'sat':
        raise ValueError('Only SaT segmentation is implemented for now.')
    set_random_seed(args.seed)
    segs = segment_with_sat(prep)
    embs = embed_segments(segs, args.embed_model)
    aspects = cluster(embs, args.clusterer)
    sent = absa_score(segs, args.absa_model)
    item_aspect = aggregate_item_aspects(segs, aspects, sent, args.topk_opinion_units)
    rich_reviews = pack(segs, aspects, sent)
    labels = {}
    if args.labeler == 'ctfidf':
        labels = label_aspects(segs, aspects)
    proc = {
        'RICH_REVIEWS': rich_reviews,
        'ITEM_ASPECT_PERF': item_aspect,
        'USER_GEO': prep.get('USER_GEO', {}),
        'META': {
            'seg_model': 'sat',
            'embed_model': args.embed_model,
            'clusterer': args.clusterer,
            'absa_model': args.absa_model,
            'labeler': args.labeler,
            'topk_opinion_units': args.topk_opinion_units,
            'version': 'v1',
            'aspect_labels': labels,
        },
    }
    return proc


def segment_with_sat(prep):
    """Segment review text with a SaT-style sentence splitter."""
    segments = []
    info = prep.get('INFO', {})
    unit_id = 0
    sent_finder = re.compile(r'[^.!?]+[.!?]?')
    for city, review_list in prep.get('REVIEWS', {}).items():
        city_info = info.get(city, {})
        for review in review_list:
            text = review.get('text') or ''
            coords = city_info.get(review.get('item_id'), {}).get('coords')
            for match in sent_finder.finditer(text):
                seg_text = match.group().strip()
                if not seg_text:
                    continue
                unit_id += 1
                segments.append({
                    'unit_id': f'unit_{unit_id:08d}',
                    'city': city,
                    'user_id': review.get('user_id'),
                    'item_id': review.get('item_id'),
                    'review_id': review.get('review_id'),
                    'span': [int(match.start()), int(match.end())],
                    'text': seg_text,
                    'coords': coords,
                })
    return segments


def embed_segments(segs, model):
    """Create deterministic embeddings for segments."""
    dims = {
        'gte-large': 1024,
        'bge-large': 1024,
        'trained': 768,
    }
    if model not in dims:
        raise ValueError(f'Unsupported embed_model: {model}')
    dim = dims[model]
    embeddings = {}
    for seg in segs:
        vec = np.zeros(dim, dtype=float)
        tokens = re.findall(r"[A-Za-z0-9']+", seg.get('text', '').lower())
        for tok in tokens:
            idx = abs(hash(tok)) % dim
            vec[idx] += 1.0
        if vec.sum():
            vec /= np.linalg.norm(vec) + 1e-9
        embeddings[seg['unit_id']] = vec
    return embeddings


def cluster(embs, algo):
    """Cluster embeddings into aspect buckets."""
    unit_ids = list(embs.keys())
    if not unit_ids:
        return {}
    matrix = np.vstack([embs[u] for u in unit_ids])
    if algo == 'kmeans':
        labels = _kmeans_labels(matrix)
    elif algo == 'hdbscan':
        labels = _density_labels(matrix)
    else:
        raise ValueError(f'Unsupported clusterer: {algo}')
    aspects = {}
    for idx, unit_id in enumerate(unit_ids):
        aspects[unit_id] = f'asp_{labels[idx]:04d}'
    return aspects


def absa_score(segs, model):
    """Score aspect sentiment with a lightweight lexicon stub."""
    if model != 'pyabsa-restaurants':
        raise ValueError(f'Unsupported absa_model: {model}')
    pos_words = {'amazing', 'awesome', 'best', 'great', 'friendly', 'fresh', 'tasty', 'delicious', 'love'}
    neg_words = {'awful', 'bad', 'cold', 'dirty', 'rude', 'slow', 'worst', 'bland', 'hate'}
    scores = {}
    for seg in segs:
        text = seg.get('text', '').lower()
        pos = sum(1 for word in pos_words if word in text)
        neg = sum(1 for word in neg_words if word in text)
        score = (pos + 1.0) / (pos + neg + 2.0)
        scores[seg['unit_id']] = float(score)
    return scores


def aggregate_item_aspects(segs, aspects, sent, topk):
    """Aggregate per-item aspect sentiment with trimmed means."""
    per_item = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seg in segs:
        unit_id = seg['unit_id']
        aspect_id = aspects.get(unit_id)
        if not aspect_id:
            continue
        city = seg.get('city')
        item_id = seg.get('item_id')
        score = sent.get(unit_id)
        if score is None:
            continue
        per_item[city][item_id][aspect_id].append(float(score))
    result = {}
    for city, item_map in per_item.items():
        result[city] = {}
        for item_id, aspect_map in item_map.items():
            result[city][item_id] = {}
            for aspect_id, values in aspect_map.items():
                limited = values[:topk] if topk and len(values) > topk else values
                limited = sorted(limited)
                trim = max(int(len(limited) * 0.1), 0)
                if trim and len(limited) > 2 * trim:
                    trimmed = limited[trim:-trim]
                elif trim and len(limited) > trim:
                    trimmed = limited[trim:]
                else:
                    trimmed = limited
                mean = sum(trimmed) / len(trimmed) if trimmed else 0.5
                result[city][item_id][aspect_id] = float(mean)
    return result


def pack(segs, aspects, sent):
    """Merge segment metadata, aspect ids, and sentiment."""
    packed = []
    for seg in segs:
        unit_id = seg['unit_id']
        entry = {
            'unit_id': unit_id,
            'city': seg.get('city'),
            'user_id': seg.get('user_id'),
            'item_id': seg.get('item_id'),
            'review_id': seg.get('review_id'),
            'span': seg.get('span'),
            'text': seg.get('text'),
            'coords': seg.get('coords'),
            'aspect_id': aspects.get(unit_id),
            'sentiment': sent.get(unit_id),
            'rel': 1.0,
            'app': None,
        }
        packed.append(entry)
    return packed


def label_aspects(segs, aspects):
    """Generate lightweight aspect labels using token frequency."""
    bags = defaultdict(lambda: defaultdict(int))
    for seg in segs:
        aspect_id = aspects.get(seg['unit_id'])
        if not aspect_id:
            continue
        tokens = re.findall(r"[A-Za-z0-9']+", seg.get('text', '').lower())
        for tok in tokens:
            bags[aspect_id][tok] += 1
    labels = {}
    for aspect_id, counts in bags.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        labels[aspect_id] = [word for word, _ in ranked]
    return labels


def haversine_km(lon1, lat1, lon2, lat2):
    """Compute haversine distance in kilometers."""
    r = 6371.0
    lon1_rad = math.radians(lon1)
    lat1_rad = math.radians(lat1)
    lon2_rad = math.radians(lon2)
    lat2_rad = math.radians(lat2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def coord_median(lons, lats):
    """Return the coordinate-wise median."""
    return [float(median(lons)), float(median(lats))]


def normalize_price(raw):
    """Normalize Yelp price information to integers 1-4 or None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return normalize_price(raw.get('RestaurantsPriceRange2'))
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.startswith('{') and 'RestaurantsPriceRange2' in text:
            try:
                parsed = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                match = re.search(r'RestaurantsPriceRange2[^0-9]*(\d)', text)
                if match:
                    return normalize_price(int(match.group(1)))
                return None
            return normalize_price(parsed.get('RestaurantsPriceRange2'))
        dollar_count = text.count('$')
        if dollar_count:
            return dollar_count
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            value = int(digits)
            return value if value > 0 else None
        return None
    if isinstance(raw, (int, float)):
        value = int(raw)
        return value if value > 0 else None
    return None


def normalize_hours(hours):
    """Normalize Yelp hours into a compact dict."""
    if not hours:
        return {}
    normalized = {}
    for day, span in hours.items():
        day_key = day.strip().lower()[:3]
        parts = []
        if isinstance(span, str):
            chunks = [chunk.strip() for chunk in span.split(';') if chunk.strip()]
            for chunk in chunks:
                if '-' in chunk:
                    start, end = chunk.split('-', 1)
                    parts.append([start.strip(), end.strip()])
        if parts:
            normalized[day_key] = parts
    return normalized


def parse_categories(raw):
    """Parse category strings into a list of normalized tokens."""
    if not raw:
        return []
    if isinstance(raw, list):
        entries = raw
    else:
        entries = str(raw).split(',')
    return [entry.strip().lower() for entry in entries if entry.strip()]


def _ensure_coords(lon, lat):
    if lon is None or lat is None:
        return [None, None]
    return [float(lon), float(lat)]


def _kmeans_labels(matrix):
    n_samples, dim = matrix.shape
    k = max(1, min(int(math.sqrt(n_samples)) or 1, 32))
    indices = _GLOBAL_RNG.choice(n_samples, size=k, replace=False)
    centers = matrix[indices].copy()
    for _ in range(10):
        dists = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = matrix[mask].mean(axis=0)
    return assign


def _density_labels(matrix):
    n_samples = matrix.shape[0]
    labels = np.zeros(n_samples, dtype=int)
    radius = np.median(np.linalg.norm(matrix - matrix.mean(axis=0), axis=1))
    if radius == 0.0:
        return labels
    cluster_centers = []
    for idx, vec in enumerate(matrix):
        assigned = False
        for cid, center in enumerate(cluster_centers):
            if np.linalg.norm(vec - center) <= radius:
                labels[idx] = cid
                cluster_centers[cid] = (center + vec) / 2.0
                assigned = True
                break
        if not assigned:
            cluster_centers.append(vec.copy())
            labels[idx] = len(cluster_centers) - 1
    return labels
