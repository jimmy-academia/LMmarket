import json
import math
import torch
from pathlib import Path

from .base import BaseSystem
from networks.model import SegmentEmbeddingModel
from llm import run_llm_batch
from utils import load_or_build


class HyperbolicSegmentSystem(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.segment_candidates = args.segment_candidate_segments
        self.segment_top_m = args.segment_top_m
        self.encode_batch_size = args.segment_encode_batch_size
        self.training_samples = args.segment_train_samples
        self.learning_rate = args.segment_learning_rate
        model_config = self._build_model_config(args)
        self.segment_encoder = SegmentEmbeddingModel(model_config)
        self.cache_dir = args.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        div_value = args.div_name if args.div_name else args.dset
        checkpoint = args.segment_checkpoint
        if checkpoint:
            self.segment_model_path = Path(checkpoint)
        else:
            self.segment_model_path = self.cache_dir / f"segment_encoder_{div_value}.pt"
        self.segment_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_ready = False
        self.city_indexes = {}
        self._ensure_model_ready()

    def recommend(self, request, city=None, top_k=None):
        print('in main method recommend')
        if not request:
            return []
        self._ensure_model_ready()
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
        raw = args.segment_model_config
        if raw:
            config.update(raw)
        if args.segment_backbone is not None:
            config["backbone_name"] = args.segment_backbone
        if args.segment_pooling is not None:
            config["pooling"] = args.segment_pooling
        if args.segment_hidden_dim is not None:
            config["hidden_dim"] = args.segment_hidden_dim
        if args.segment_aspect_dim is not None:
            config["aspect_dim"] = args.segment_aspect_dim
        if args.segment_sentiment_dim is not None:
            config["sentiment_dim"] = args.segment_sentiment_dim
        if args.segment_lambda_aspect is not None:
            config["lambda_aspect"] = args.segment_lambda_aspect
        if args.segment_lambda_sentiment is not None:
            config["lambda_sentiment"] = args.segment_lambda_sentiment
        if args.segment_aspect_temperature is not None:
            config["aspect_temperature"] = args.segment_aspect_temperature
        if args.segment_sentiment_temperature is not None:
            config["sentiment_temperature"] = args.segment_sentiment_temperature
        if args.segment_sentiment_loss is not None:
            config["sentiment_loss"] = args.segment_sentiment_loss
        if args.segment_sentiment_margin is not None:
            config["sentiment_margin"] = args.segment_sentiment_margin
        if args.segment_curvature is not None:
            config["curvature"] = args.segment_curvature
        if args.segment_max_length is not None:
            config["max_length"] = args.segment_max_length
        if args.device is not None and "device" not in config:
            config["device"] = args.device
        return config

    def _ensure_model_ready(self):
        if self.model_ready:
            return

        def save_fn(path, state):
            torch.save(state, path)

        def load_fn(path):
            return torch.load(path, map_location=self.segment_encoder.device)

        def build_fn():
            self._train_with_segments()
            self.segment_encoder.eval()
            return self.segment_encoder.state_dict()

        state = load_or_build(self.segment_model_path, save_fn, load_fn, build_fn)
        if isinstance(state, dict):
            for key in ["model_state_dict", "state_dict", "model"]:
                nested = state.get(key)
                if isinstance(nested, dict):
                    state = nested
                    break
            self.segment_encoder.load_state_dict(state, strict=False)
        self.segment_encoder.eval()
        self.model_ready = True

    def _train_with_segments(self):
        print('method.py train with segments')
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

    def calibrate_threshold_with_llm(self, candidates, provisional_tau=None, sample_size=200, target_precision_lb=0.9, sentiment_conf_min=0.7, double_judge=False, model="gpt-4.1-mini", confidence=0.95, max_iterations=6):
        usable = []
        for entry in candidates or []:
            if not isinstance(entry, dict):
                continue
            score = entry.get("score")
            if score is None:
                score = entry.get("similarity")
            if score is None:
                continue
            if not isinstance(score, (float, int)):
                continue
            anchor = entry.get("anchor") or entry.get("anchor_segment") or entry.get("segment_a") or {}
            neighbor = entry.get("neighbor") or entry.get("neighbor_segment") or entry.get("segment_b") or {}
            anchor_text = self._normalize_text(anchor.get("text"))
            neighbor_text = self._normalize_text(neighbor.get("text"))
            if not anchor_text or not neighbor_text:
                continue
            usable.append({
                "score": float(score),
                "anchor": anchor,
                "neighbor": neighbor,
                "anchor_text": anchor_text,
                "neighbor_text": neighbor_text,
            })
        if not usable:
            fallback_tau = float(provisional_tau) if isinstance(provisional_tau, (float, int)) else 0.0
            return fallback_tau, [], [], [], []
        scores = [entry["score"] for entry in usable]
        min_score = min(scores)
        max_score = max(scores)
        if provisional_tau is None or not isinstance(provisional_tau, (float, int)):
            provisional_tau = sum(scores) / len(scores)
        provisional_tau = min(max(float(provisional_tau), min_score), max_score)
        low = min_score
        high = max_score
        best_tau = None
        audit_cache = {}
        previous_tau = None
        tau = provisional_tau

        #### temp
        max_iterations = 1
        ###
        for iteration in range(max_iterations):
            if iteration > 0:
                tau = (low + high) / 2.0
            if previous_tau is not None and abs(tau - previous_tau) < 1e-4:
                break
            previous_tau = tau
            qualified = [entry for entry in usable if entry["score"] >= tau]
            if not qualified:
                high = tau
                continue
            qualified.sort(key=lambda entry: abs(entry["score"] - tau))
            sample = qualified[:sample_size] if len(qualified) > sample_size else qualified
            audits = self._ensure_llm_audits(sample, audit_cache, model, double_judge)
            positives = sum(1 for key in audits if audits[key]["aspect_same"])
            total = len(audits)
            if total == 0:
                high = tau
                continue
            lb = self._wilson_lower_bound(positives, total, confidence)
            if lb >= target_precision_lb:
                best_tau = tau
                high = tau
            else:
                low = tau
        if best_tau is None:
            best_tau = high
        final_tau = max(min(best_tau, max_score), min_score)
        eligible_pairs = [entry for entry in usable if entry["score"] >= final_tau]
        self._ensure_llm_audits(eligible_pairs, audit_cache, model, double_judge)
        silver_aspect = []
        bronze_aspect = []
        silver_sentiment = {}
        bronze_sentiment = {}
        for entry in eligible_pairs:
            key = self._pair_key(entry)
            audit = audit_cache.get(key)
            anchor = entry["anchor"]
            neighbor = entry["neighbor"]
            anchor_id = self._segment_identifier(anchor, entry["anchor_text"])
            neighbor_id = self._segment_identifier(neighbor, entry["neighbor_text"])
            payload = {
                "anchor_segment": anchor_id,
                "neighbor_segment": neighbor_id,
                "score": entry["score"],
            }
            if audit and audit["aspect_same"]:
                payload["aspect_label"] = audit.get("aspect_label")
                silver_aspect.append(payload)
            else:
                bronze_aspect.append(payload)
            if audit:
                for idx, segment in enumerate((anchor, neighbor)):
                    seg_id = self._segment_identifier(segment, entry["anchor_text"] if idx == 0 else entry["neighbor_text"])
                    sentiment = audit["sentiments"][idx]
                    if not seg_id or sentiment is None:
                        continue
                    container = silver_sentiment if sentiment["confidence"] >= sentiment_conf_min else bronze_sentiment
                    existing = container.get(seg_id)
                    enriched = {
                        "segment_id": seg_id,
                        "label": sentiment["label"],
                        "score": sentiment["score"],
                        "confidence": sentiment["confidence"],
                        "source": "llm",
                    }
                    if existing:
                        enriched = self._merge_sentiments(existing, enriched)
                    container[seg_id] = enriched
        silver_list = list(silver_sentiment.values())
        bronze_list = list(bronze_sentiment.values())
        silver_list.sort(key=lambda row: (-row["confidence"], -abs(row["score"]), row["segment_id"]))
        bronze_list.sort(key=lambda row: (-row["confidence"], -abs(row["score"]), row["segment_id"]))
        silver_aspect.sort(key=lambda row: (-row["score"], row["anchor_segment"], row["neighbor_segment"]))
        bronze_aspect.sort(key=lambda row: (-row["score"], row["anchor_segment"], row["neighbor_segment"]))
        return final_tau, silver_aspect, bronze_aspect, silver_list, bronze_list

    def _merge_sentiments(self, existing, update):
        weight_existing = existing.get("confidence", 0.0)
        weight_update = update.get("confidence", 0.0)
        total = weight_existing + weight_update
        if total <= 0:
            merged_score = 0.5 * (existing.get("score", 0.0) + update.get("score", 0.0))
            merged_conf = max(existing.get("confidence", 0.0), update.get("confidence", 0.0))
        else:
            merged_score = (existing.get("score", 0.0) * weight_existing + update.get("score", 0.0) * weight_update) / total
            merged_conf = total / 2.0
        label = existing.get("label")
        if update.get("confidence", 0.0) >= existing.get("confidence", 0.0):
            label = update.get("label")
        merged = {
            "segment_id": existing.get("segment_id") or update.get("segment_id"),
            "label": label,
            "score": merged_score,
            "confidence": merged_conf,
            "source": "llm",
        }
        return merged

    def _segment_identifier(self, segment, fallback):
        if not isinstance(segment, dict):
            return fallback
        for key in ["segment_id", "id", "segment"]:
            value = segment.get(key)
            if value:
                return value
        item = segment.get("item_id")
        if item:
            text = self._normalize_text(segment.get("text"))
            if text:
                return f"{item}:{text}"
        return fallback

    def _ensure_llm_audits(self, entries, cache, model, double_judge):
        pending = []
        keys = []
        for entry in entries:
            key = self._pair_key(entry)
            if key in cache:
                continue
            prompt = self._build_llm_prompt(entry["anchor_text"], entry["neighbor_text"])
            pending.append(prompt)
            keys.append(key)
        if pending:
            responses = run_llm_batch(pending, "segment_aspect_sentiment_audit", model=model, temperature=0.0, use_json=True)
            parsed_primary = [self._parse_audit_output(text) for text in responses]
            if double_judge:
                responses_secondary = run_llm_batch(pending, "segment_aspect_sentiment_audit_double", model=model, temperature=0.0, use_json=True)
                parsed_secondary = [self._parse_audit_output(text) for text in responses_secondary]
                combined = []
                for first, second in zip(parsed_primary, parsed_secondary):
                    combined.append(self._combine_audits(first, second))
                parsed_primary = combined
            for key, parsed in zip(keys, parsed_primary):
                cache[key] = parsed
        result = {}
        for entry in entries:
            key = self._pair_key(entry)
            if key in cache:
                result[key] = cache[key]
        return result

    def _combine_audits(self, primary, secondary):
        if not primary and not secondary:
            return {"aspect_same": False, "aspect_label": "", "sentiments": [None, None]}
        if not secondary:
            return primary
        if not primary:
            return secondary
        sentiments_primary = primary.get("sentiments") or [None, None]
        sentiments_secondary = secondary.get("sentiments") or [None, None]
        aspect_same = bool(primary.get("aspect_same")) and bool(secondary.get("aspect_same"))
        label = primary.get("aspect_label") if aspect_same else ""
        if not label and aspect_same:
            label = secondary.get("aspect_label") or ""
        sentiments = []
        for idx in range(2):
            sentiments.append(self._merge_sentiment_entries(sentiments_primary[idx], sentiments_secondary[idx]))
        return {"aspect_same": aspect_same, "aspect_label": label, "sentiments": sentiments}

    def _merge_sentiment_entries(self, first, second):
        if first is None and second is None:
            return {"label": "NEU", "score": 0.0, "confidence": 0.0}
        if second is None:
            return first
        if first is None:
            return second
        weight_first = first.get("confidence", 0.0)
        weight_second = second.get("confidence", 0.0)
        total = weight_first + weight_second
        if total <= 0:
            score = 0.5 * (first.get("score", 0.0) + second.get("score", 0.0))
            confidence = max(first.get("confidence", 0.0), second.get("confidence", 0.0))
        else:
            score = (first.get("score", 0.0) * weight_first + second.get("score", 0.0) * weight_second) / total
            confidence = total / 2.0
        label = first.get("label") if weight_first >= weight_second else second.get("label")
        return {"label": label, "score": score, "confidence": confidence}

    def _pair_key(self, entry):
        anchor_id = self._segment_identifier(entry["anchor"], entry["anchor_text"])
        neighbor_id = self._segment_identifier(entry["neighbor"], entry["neighbor_text"])
        return (str(anchor_id), str(neighbor_id))

    def _build_llm_prompt(self, anchor_text, neighbor_text):
        template = [
            "You are auditing two review segments.",
            "For each pair, answer in JSON with keys: aspect_same (YES or NO), aspect_label (short phrase), segment_a, segment_b.",
            "segment_a and segment_b must each contain label (NEG/NEU/POS), score (between -1 and 1), confidence (between 0 and 1).",
            "Decide aspect_same by focusing on the primary aspect described, ignoring sentiment differences.",
            "Use empty string for aspect_label when aspect_same is NO.",
            "Pair:",
            f"Segment A: {self._truncate_for_prompt(anchor_text)}",
            f"Segment B: {self._truncate_for_prompt(neighbor_text)}",
        ]
        return "\n".join(template)

    def _truncate_for_prompt(self, text, limit=480):
        snippet = text.strip()
        if len(snippet) <= limit:
            return snippet
        trimmed = snippet[:limit].rstrip()
        return f"{trimmed}…"

    def _parse_audit_output(self, text):
        if not text:
            return {"aspect_same": False, "aspect_label": "", "sentiments": [None, None]}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            return {"aspect_same": False, "aspect_label": "", "sentiments": [None, None]}
        aspect_flag = str(data.get("aspect_same", "")).strip().upper() == "YES"
        label = self._normalize_text(data.get("aspect_label")) if aspect_flag else ""
        sentiments = []
        for key in ["segment_a", "segment_b"]:
            sentiments.append(self._normalize_sentiment_entry(data.get(key)))
        return {"aspect_same": aspect_flag, "aspect_label": label, "sentiments": sentiments}

    def _normalize_sentiment_entry(self, payload):
        if not isinstance(payload, dict):
            return {"label": "NEU", "score": 0.0, "confidence": 0.0}
        label = str(payload.get("label", "NEU")).strip().upper()
        if label not in {"NEG", "NEU", "POS"}:
            label = "NEU"
        score = payload.get("score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        score = max(min(score, 1.0), -1.0)
        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0
        confidence = max(min(confidence, 1.0), 0.0)
        return {"label": label, "score": score, "confidence": confidence}

    def _wilson_lower_bound(self, success, total, confidence):
        if total <= 0:
            return 0.0
        if success <= 0:
            return 0.0
        p = float(success) / float(total)
        z = self._z_value(confidence)
        denominator = 1.0 + (z ** 2) / total
        centre = p + (z ** 2) / (2.0 * total)
        margin = z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * total)) / total)
        return (centre - margin) / denominator

    def _z_value(self, confidence):
        alpha = 1.0 - confidence
        if alpha <= 0:
            return 0.0
        # approximate inverse error function for normal quantile (two-tailed)
        target = 1.0 - alpha / 2.0
        return math.sqrt(2.0) * self._inv_erf(2.0 * target - 1.0)

    def _inv_erf(self, x):
        clamped = max(min(x, 0.999999), -0.999999)
        a = 0.147
        log_term = math.log(1.0 - clamped * clamped)
        part = 2.0 / (math.pi * a) + log_term / 2.0
        inside = part * part - log_term / a
        sign = 1.0 if x >= 0 else -1.0
        return sign * math.sqrt(max(inside, 0.0))

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
