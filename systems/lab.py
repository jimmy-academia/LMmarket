import json
import logging
import numpy as np

from utils import load_or_build, dumpj, loadj
from .base import BaseSystem
from networks.aspect import infer_aspects_weights


class DevMethod(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.experiment_results = {}
        logging.info("[dev] starting retrieval experiments")
        self.run_experiments()

    def run_experiments(self):
        self.experiment_segmentation()
        self.experiment_query_variants()
        self.experiment_result_overlap()
        self.experiment_embedding_shifts()

    def experiment_segmentation(self, sample_reviews=3):
        total_tokens = []
        multi_sentence = 0
        for seg in self.segments:
            text = seg.get('text')
            if not text:
                continue
            sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
            if len(sentences) > 1:
                multi_sentence += 1
            total_tokens.append(len(text.split()))

        ratio = 0.0
        avg_tokens = 0.0
        if total_tokens:
            ratio = multi_sentence / len(total_tokens)
            avg_tokens = sum(total_tokens) / len(total_tokens)

        review_samples = []
        review_ids = list(self.review_segments.keys())[:sample_reviews]
        for review_id in review_ids:
            seg_ids = self.review_segments[review_id]
            sample = []
            for offset, seg_idx in enumerate(seg_ids):
                segment = self.segments[seg_idx]
                text = segment.get('text')
                if not text:
                    continue
                sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
                sample.append({
                    'segment_id': seg_idx,
                    'position': offset,
                    'tokens': len(text.split()),
                    'sentences': len(sentences),
                    'text': text,
                })
            review_samples.append({'review_id': review_id, 'segments': sample})

        payload = {
            'segment_count': len(total_tokens),
            'multi_sentence_ratio': ratio,
            'avg_tokens': avg_tokens,
            'review_samples': review_samples,
        }
        self.experiment_results['segmentation'] = payload
        logging.info('[exp1] total=%d multi_sentence=%.3f avg_tokens=%.2f', len(total_tokens), ratio, avg_tokens)
        for entry in review_samples:
            logging.info('[exp1] review %s has %d segments', entry['review_id'], len(entry['segments']))
            for seg in entry['segments']:
                logging.info('[exp1] seg%05d pos=%d tokens=%d sentences=%d :: %s', seg['segment_id'], seg['position'], seg['tokens'], seg['sentences'], seg['text'])

    def experiment_query_variants(self, aspect_limit=3, topk=8):
        if not self.reviews:
            logging.info('[exp2] no reviews available')
            return
        base_query = self.reviews[0].get('text')
        if not base_query:
            logging.info('[exp2] missing base query text')
            return
        aspect_path = self.args.cache_dir / 'lab_aspects.json'
        raw = load_or_build(aspect_path, dumpj, loadj, infer_aspects_weights, base_query)
        aspect_dict = raw if isinstance(raw, dict) else json.loads(raw)
        aspects = aspect_dict.get('aspects') or []
        variants_output = {}
        for aspect in aspects[:aspect_limit]:
            name = aspect.get('name')
            sentence = aspect.get('sentence')
            positives = aspect.get('positives') or []
            queries = {
                'keyword': name.replace('_', ' ') if name else None,
                'description': sentence,
                'definition': f"{name} focuses on {sentence}" if name and sentence else None,
                'find_sentence': f"Find segments describing: {sentence}" if sentence else None,
                'synonyms': ' '.join(positives) if positives else None,
            }
            aspect_runs = {}
            for variant, text in queries.items():
                if not text:
                    continue
                scores, idxs = self._collect_top_segments(text, topk)
                results = [{'segment_id': idx, 'score': float(score), 'text': self.segments[idx].get('text')} for score, idx in zip(scores, idxs)]
                aspect_runs[variant] = {'query': text, 'results': results}
                logging.info('[exp2] %s :: %s', variant, text)
                for entry in results:
                    seg_text = entry.get('text')
                    logging.info('[exp2] seg%05d score=%.4f %s', entry['segment_id'], entry['score'], seg_text)
            aspect_key = name or f'aspect_{len(variants_output)}'
            variants_output[aspect_key] = aspect_runs
        self.experiment_results['query_variants'] = variants_output

    def experiment_result_overlap(self):
        runs = self.experiment_results.get('query_variants')
        if not runs:
            logging.info('[exp3] no query variants data')
            return
        overlap_payload = {}
        for aspect, variant_map in runs.items():
            ids_map = {}
            for variant, payload in variant_map.items():
                entries = payload.get('results') if payload else []
                ids_map[variant] = {entry['segment_id'] for entry in entries}
            variants = list(ids_map.keys())
            overlaps = {}
            for i, first in enumerate(variants):
                for second in variants[i + 1:]:
                    first_ids = ids_map[first]
                    second_ids = ids_map[second]
                    if not first_ids and not second_ids:
                        score = 0.0
                    else:
                        union = len(first_ids | second_ids)
                        score = len(first_ids & second_ids) / union if union else 0.0
                    overlaps[f'{first}∩{second}'] = score
            unique_segments = {}
            for variant, ids in ids_map.items():
                others = set()
                for other_variant, other_ids in ids_map.items():
                    if other_variant != variant:
                        others |= other_ids
                unique_segments[variant] = list(ids - others)
            overlap_payload[aspect] = {
                'overlaps': overlaps,
                'unique_segments': unique_segments,
            }
            logging.info('[exp3] aspect=%s overlaps=%s', aspect, overlaps)
            logging.info('[exp3] aspect=%s uniques=%s', aspect, unique_segments)
        self.experiment_results['result_overlap'] = overlap_payload

    def experiment_embedding_shifts(self, topk=5, shift_scale=0.1):
        runs = self.experiment_results.get('query_variants')
        if not runs:
            logging.info('[exp4] no query variants data')
            return
        aspect_name = next(iter(runs)) if runs else None
        if not aspect_name:
            logging.info('[exp4] missing aspect selection')
            return
        aspect_data = runs[aspect_name]
        description = aspect_data.get('description')
        if not description:
            logging.info('[exp4] description variant not cached')
            return
        base_text = description.get('query')
        if not base_text:
            logging.info('[exp4] no text found for base segment')
            return
        base_vec = self._encode_query(base_text)
        base_vec = np.asarray(base_vec, dtype=np.float32).reshape(-1)
        if not np.linalg.norm(base_vec):
            logging.info('[exp4] zero base vector')
            return
        positives = []
        for payload in aspect_data.values():
            entries = payload.get('results') if payload else []
            for entry in entries:
                seg_text = entry.get('text')
                if seg_text:
                    positives.append(seg_text)
        positive_vecs = []
        for text in positives[:5]:
            vec = self._encode_query(text)
            positive_vecs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        if not positive_vecs:
            logging.info('[exp4] no positive vectors available')
            return
        direction = np.mean(positive_vecs, axis=0)
        direction /= np.linalg.norm(direction) + 1e-12
        base_norm = base_vec / (np.linalg.norm(base_vec) + 1e-12)
        shifted_vec = base_norm + shift_scale * direction
        shifted_vec /= np.linalg.norm(shifted_vec) + 1e-12
        base_scores, base_idxs = self._get_top_k(base_norm, min(topk, len(self.segments)))
        shift_scores, shift_idxs = self._get_top_k(shifted_vec, min(topk, len(self.segments)))
        similarity = float(base_norm @ shifted_vec)
        payload = {
            'cos_similarity': similarity,
            'base_results': [{'segment_id': idx, 'score': float(score)} for score, idx in zip(base_scores, base_idxs)],
            'shifted_results': [{'segment_id': idx, 'score': float(score)} for score, idx in zip(shift_scores, shift_idxs)],
        }
        self.experiment_results['embedding_shift'] = payload
        logging.info('[exp4] cos_similarity=%.4f', similarity)
        logging.info('[exp4] base=%s', payload['base_results'])
        logging.info('[exp4] shifted=%s', payload['shifted_results'])

    def _collect_top_segments(self, text, topk):
        query_vec = self._encode_query(text)
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(query_vec)
        if norm:
            query_vec /= norm
        scores, idxs = self._get_top_k(query_vec, min(topk, len(self.segments)))
        return scores, idxs

    def recommend(self, query):
        logging.info('[dev] recommend placeholder :: %s', query)

    def rr(self, *args, **kwargs):
        self.retrieve_similar_segments(*args, **kwargs)
