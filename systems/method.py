import os
import torch

from .base import BaseSystem
from networks.model import SegmentEmbeddingModel


class HyperbolicSegmentSystem(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.segment_candidates = getattr(args, "segment_candidate_segments", 50)
        if not isinstance(self.segment_candidates, int) or self.segment_candidates <= 0:
            self.segment_candidates = 50
        self.segment_top_m = getattr(args, "segment_top_m", 3)
        if not isinstance(self.segment_top_m, int) or self.segment_top_m < 0:
            self.segment_top_m = 3
        self.encode_batch_size = getattr(args, "segment_encode_batch_size", 8)
        if not isinstance(self.encode_batch_size, int) or self.encode_batch_size <= 0:
            self.encode_batch_size = 8
        self.training_samples = getattr(args, "segment_train_samples", 64)
        if not isinstance(self.training_samples, int) or self.training_samples < 0:
            self.training_samples = 0
        self.learning_rate = getattr(args, "segment_learning_rate", 2e-5)
        if not isinstance(self.learning_rate, float) and not isinstance(self.learning_rate, int):
            self.learning_rate = 2e-5
        model_config = self._build_model_config(args)
        self.segment_encoder = SegmentEmbeddingModel(model_config)
        self.model_ready = False
        self.city_indexes = {}
        self._load_or_train_model()

    def recommend(self, request, city=None, top_k=None):
        if not request:
            return []
        self._load_or_train_model()
        index = self._ensure_city_index(city)
        if not index:
            return []
        query_text = str(request).strip()
        if not query_text:
            return []
        query_embed = self.segment_encoder.get_aspect_embedding(query_text)
        if query_embed.numel() == 0:
            return []
        query_ball = self.segment_encoder.expmap0(query_embed)
        distances = self.segment_encoder.hyperbolic_distance(query_ball, index["ball"]).squeeze(-1)
        if distances.numel() == 0:
            return []
        scores = 1.0 / (1.0 + distances)
        candidate_limit = min(len(scores), self.segment_candidates)
        if candidate_limit <= 0:
            return []
        values, indices = torch.topk(scores, candidate_limit, largest=True)
        aggregated = {}
        for score, idx in zip(values.tolist(), indices.tolist()):
            item_id = index["items"][idx]
            if not item_id:
                continue
            segment = index["segments"][idx]
            if not isinstance(segment, dict):
                continue
            bucket = aggregated.get(item_id)
            if bucket is None:
                bucket = []
                aggregated[item_id] = bucket
            bucket.append((float(score), float(distances[idx].item()), segment))
        if not aggregated:
            return []
        results = []
        for item_id, entries in aggregated.items():
            entries.sort(key=lambda row: (-row[0], row[1]))
            selected = entries[:self.segment_top_m] if self.segment_top_m > 0 else entries
            total_score = sum(row[0] for row in selected)
            evidence = []
            snippets = []
            for score, dist, segment in selected:
                seg_id = segment.get("segment_id")
                if seg_id:
                    evidence.append(seg_id)
                snippet = self._normalize_text(segment.get("text"))
                if snippet:
                    snippets.append(snippet)
            results.append({
                "item_id": item_id,
                "model_score": float(total_score),
                "evidence": evidence,
                "short_excerpt": self._compose_excerpt(snippets),
                "full_explanation": self._compose_explanation(snippets),
            })
        results.sort(key=lambda entry: (-entry["model_score"], entry["item_id"]))
        cutoff = top_k if isinstance(top_k, int) and top_k > 0 else self.default_top_k
        return results[:cutoff] if cutoff and cutoff > 0 else results

    def _build_model_config(self, args):
        config = {}
        raw = getattr(args, "segment_model_config", None)
        if isinstance(raw, dict):
            config.update(raw)
        mapping = {
            "segment_backbone": "backbone_name",
            "segment_pooling": "pooling",
            "segment_hidden_dim": "hidden_dim",
            "segment_aspect_dim": "aspect_dim",
            "segment_sentiment_dim": "sentiment_dim",
            "segment_lambda_aspect": "lambda_aspect",
            "segment_lambda_sentiment": "lambda_sentiment",
            "segment_aspect_temperature": "aspect_temperature",
            "segment_sentiment_temperature": "sentiment_temperature",
            "segment_sentiment_loss": "sentiment_loss",
            "segment_sentiment_margin": "sentiment_margin",
            "segment_curvature": "curvature",
            "segment_max_length": "max_length",
        }
        for attr, key in mapping.items():
            value = getattr(args, attr, None)
            if value is not None:
                config[key] = value
        device_value = getattr(args, "device", None)
        if device_value is not None and "device" not in config:
            config["device"] = device_value
        return config

    def _load_or_train_model(self):
        if self.model_ready:
            return
        checkpoint = getattr(self.args, "segment_checkpoint", None)
        if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
            state = torch.load(checkpoint, map_location=self.segment_encoder.device)
            if isinstance(state, dict):
                for key in ["model_state_dict", "state_dict", "model"]:
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            if isinstance(state, dict):
                self.segment_encoder.load_state_dict(state, strict=False)
            self.segment_encoder.eval()
            self.model_ready = True
            return
        self._train_with_segments()
        self.segment_encoder.eval()
        self.model_ready = True

    def _train_with_segments(self):
        if self.training_samples <= 0:
            return
        pairs = self._collect_training_pairs(self.training_samples)
        if not pairs:
            return
        anchors = []
        positives = []
        for anchor_seg, positive_seg in pairs:
            anchor_text = self._normalize_text(anchor_seg.get("text"))
            positive_text = self._normalize_text(positive_seg.get("text"))
            if not anchor_text or not positive_text:
                continue
            anchors.append(anchor_text)
            positives.append(positive_text)
        if not anchors:
            return
        negatives = positives[1:] + positives[:1]
        if len(negatives) < len(anchors):
            deficit = len(anchors) - len(negatives)
            negatives.extend(anchors[:deficit])
        negatives = negatives[:len(anchors)]
        tokenizer = self.segment_encoder.tokenizer
        device = self.segment_encoder.device
        self.segment_encoder.train()
        optimizer = torch.optim.Adam(self.segment_encoder.parameters(), lr=float(self.learning_rate))
        anchor_tokens = tokenizer(anchors, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length)
        positive_tokens = tokenizer(positives, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length)
        negative_tokens = tokenizer(negatives, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length)
        anchor_inputs = {key: value.to(device) for key, value in anchor_tokens.items()}
        positive_inputs = {key: value.to(device) for key, value in positive_tokens.items()}
        negative_inputs = {key: value.to(device) for key, value in negative_tokens.items()}
        outputs_anchor = self.segment_encoder.forward(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
        outputs_positive = self.segment_encoder.forward(positive_inputs["input_ids"], positive_inputs["attention_mask"])
        outputs_negative = self.segment_encoder.forward(negative_inputs["input_ids"], negative_inputs["attention_mask"])
        positives_tensor = outputs_positive["z_asp"].unsqueeze(1)
        negatives_tensor = outputs_negative["z_asp"].unsqueeze(1)
        loss = self.segment_encoder.aspect_contrastive_loss(outputs_anchor["z_asp"], positives_tensor, negatives_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _collect_training_pairs(self, limit):
        if limit <= 0:
            return []
        pairs = []
        for item_id in sorted(self.item_segments.keys()):
            segments = self.item_segments.get(item_id)
            if not segments:
                continue
            valid = [segment for segment in segments if isinstance(segment, dict) and segment.get("text")]
            if len(valid) < 2:
                continue
            base = valid[0]
            for other in valid[1:]:
                pairs.append((base, other))
                if len(pairs) >= limit:
                    return pairs
        return pairs

    def _ensure_city_index(self, city=None):
        key = self.get_city_key(city)
        if not key:
            return None
        cached = self.city_indexes.get(key)
        if cached is not None:
            return cached
        payload = self.get_city_data(key)
        if not payload:
            self.city_indexes[key] = None
            return None
        segments = []
        items = payload.get("ITEMS") if isinstance(payload, dict) else None
        if isinstance(items, dict):
            for item_id in items.keys():
                collection = self.item_segments.get(item_id)
                if not collection:
                    continue
                segments.extend(collection)
        reviews = payload.get("REVIEWS") if isinstance(payload, dict) else None
        if isinstance(reviews, list):
            for review in reviews:
                if not isinstance(review, dict):
                    continue
                review_id = review.get("review_id")
                if review_id:
                    segments.extend(self.get_review_segments(review_id))
        cleaned = []
        seen = set()
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            seg_id = segment.get("segment_id")
            if not seg_id or seg_id in seen:
                continue
            text = segment.get("text")
            if not text:
                continue
            seen.add(seg_id)
            cleaned.append(segment)
        if not cleaned:
            self.city_indexes[key] = None
            return None
        texts = [segment.get("text") for segment in cleaned]
        aspects = self.segment_encoder.encode_texts(texts, batch_size=self.encode_batch_size, logmap=True)
        if aspects.numel() == 0:
            self.city_indexes[key] = None
            return None
        with torch.no_grad():
            ball = self.segment_encoder.expmap0(aspects)
        index = {
            "tangent": aspects,
            "ball": ball,
            "segments": cleaned,
            "items": [segment.get("item_id") for segment in cleaned],
        }
        self.city_indexes[key] = index
        return index

    def _normalize_text(self, text):
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        return collapsed.strip()

    def _compose_excerpt(self, summaries):
        if not summaries:
            return ""
        combined = " ".join(summaries)
        max_len = 160
        if len(combined) <= max_len:
            return combined
        trimmed = combined[:max_len].rstrip()
        return f"{trimmed}…"

    def _compose_explanation(self, summaries):
        if not summaries:
            return ""
        parts = []
        for idx, summary in enumerate(summaries, 1):
            parts.append(f"{idx}) {summary}")
        return " ".join(parts)
