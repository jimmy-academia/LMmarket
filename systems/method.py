import json
import math
import torch

from .base import BaseSystem
from llm import run_llm_batch

from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


# =========================
# Model
# =========================
class SegmentEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Fixed defaults (no config) ----
        backbone = "bert-base-uncased"
        self.pooling = "cls"
        self.max_length = 160
        self.hidden_dim = 512
        self.aspect_dim = 64
        self.sentiment_dim = 3
        self.lambda_aspect = 1.0
        self.lambda_sentiment = 1.0
        self.aspect_temperature = 0.1
        self.sentiment_temperature = 1.0
        self.sentiment_loss_type = "ce"  # "ce" | "mse" | "margin"
        self.sentiment_margin = 1.0
        self.curvature = 1.0
        self.eps = 1e-5

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        enc_dim = self.encoder.config.hidden_size

        self.trunk = nn.Sequential(
            nn.Linear(enc_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.aspect_head = nn.Linear(self.hidden_dim, self.aspect_dim)
        self.sentiment_head = nn.Linear(self.hidden_dim, self.sentiment_dim)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1)
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            pooled = out[:, 0]
        h = self.trunk(pooled)
        z_asp = self.expmap0(self.aspect_head(h))
        z_sent = self.sentiment_head(h)
        return {"z_asp": z_asp, "z_sent": z_sent}

    # ------- Losses -------
    def aspect_contrastive_loss(self, anchor, positives, negatives=None, temperature=None):
        t = temperature or self.aspect_temperature
        anchor = self.proj(anchor).unsqueeze(1)          # [B,1,D]
        if positives.ndim == 2: positives = positives.unsqueeze(1)
        if negatives is not None and negatives.ndim == 2: negatives = negatives.unsqueeze(1)
        pos_sim = torch.exp(-self.hyperbolic_distance(anchor, positives) / t).sum(1)
        denom = pos_sim
        if negatives is not None:
            neg_sim = torch.exp(-self.hyperbolic_distance(anchor, negatives) / t).sum(1)
            denom = denom + neg_sim
        loss = -torch.log((pos_sim / denom.clamp(min=self.eps)).clamp(min=self.eps))
        return loss.mean()

    def sentiment_loss(self, pred, labels):
        if self.sentiment_loss_type == "mse":
            return F.mse_loss(pred, labels)
        if self.sentiment_loss_type == "margin":
            margin_term = self.sentiment_margin - labels.float() * pred.squeeze(-1)
            return F.relu(margin_term).mean()
        if pred.ndim == 1: pred = pred.unsqueeze(0)
        return F.cross_entropy(pred, labels)

    def compute_multi_task_loss(self, anchor, positives, negatives, sentiment_pred, sentiment_labels, lambda_aspect=None, lambda_sentiment=None):
        la = self.lambda_aspect if lambda_aspect is None else lambda_aspect
        ls = self.lambda_sentiment if lambda_sentiment is None else lambda_sentiment
        return la * self.aspect_contrastive_loss(anchor, positives, negatives) + ls * self.sentiment_loss(sentiment_pred, sentiment_labels)

    # ------- Encoders / helpers -------
    def encode_texts(self, texts, batch_size=None, logmap=True, return_sentiment=False):
        if not texts:
            ea = torch.empty(0, self.aspect_dim)
            if return_sentiment:
                es = torch.empty(0, self.sentiment_dim)
                return ea, es
            return ea

        bs = batch_size or len(texts)
        asp_chunks, sent_chunks = [], []
        self.eval()
        for i in tqdm(range(0, len(texts), bs), ncols=88, desc="encode_texts"):
            batch = texts[i:i + bs]
            toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                o = self.forward(toks["input_ids"], toks["attention_mask"])
            a = self.logmap0(o["z_asp"]) if logmap else o["z_asp"]
            asp_chunks.append(a.cpu())
            if return_sentiment:
                sent_chunks.append(o["z_sent"].cpu())
        A = torch.cat(asp_chunks, 0)
        if return_sentiment:
            S = torch.cat(sent_chunks, 0) if sent_chunks else torch.empty(0, self.sentiment_dim)
            return A, S
        return A

    def get_aspect_embedding(self, text):
        e = self.encode_texts([text], batch_size=1, logmap=True)
        return e[0] if e.numel() else torch.empty(self.aspect_dim)

    def get_sentiment_embedding(self, text):
        _, s = self.encode_texts([text], batch_size=1, logmap=False, return_sentiment=True)
        return s[0] if s.numel() else torch.empty(self.sentiment_dim)

    def score_aspect(self, u, v):
        u = torch.as_tensor(u, dtype=torch.float32)
        v = torch.as_tensor(v, dtype=torch.float32)
        d = self.hyperbolic_distance(self.expmap0(u), self.expmap0(v))
        return d.squeeze().item()

    # ------- Hyperbolic ops (Poincaré ball) -------
    def logmap0(self, x):
        x = self.proj(x)
        n = x.norm(-1, keepdim=True).clamp(min=self.eps)
        sc = math.sqrt(self.curvature)
        return (2.0 * torch.atanh(n * sc) / (sc * n)) * x

    def expmap0(self, v):
        n = v.norm(-1, keepdim=True).clamp(min=self.eps)
        sc = math.sqrt(self.curvature)
        return self.proj(torch.tanh(n * sc / 2.0) * v / (n * sc))

    def hyperbolic_distance(self, u, v):
        u = self.proj(u); v = self.proj(v)
        sqdist = self._sqnorm(u - v)
        un = self._sqnorm(u); vn = self._sqnorm(v)
        denom = (1.0 - un).clamp(min=self.eps) * (1.0 - vn).clamp(min=self.eps)
        arg = (1.0 + 2.0 * sqdist / denom).clamp(min=1.0 + self.eps)
        return torch.acosh(arg).squeeze(-1)

    def mobius_add(self, x, y):
        c = self.curvature
        x2 = self._sqnorm(x); y2 = self._sqnorm(y)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return self.proj(num / den.clamp(min=self.eps))

    def proj(self, x):
        max_norm = 1 - self.eps
        n = x.norm(-1, keepdim=True)
        f = torch.where(n >= max_norm, max_norm / n, torch.ones_like(n))
        return x * f

    def _sqnorm(self, x):
        return (x * x).sum(-1, keepdim=True)

    # ------- Train / Inference -------
    def inference(self, batch, optimizer=None, sentiment_labels=None, aspect_positives=None, aspect_negatives=None, train=False):
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        out = self.forward(ids, mask)
        if not train:
            return out

        anchor = out["z_asp"]
        pos = aspect_positives if aspect_positives is not None else anchor.unsqueeze(1)
        neg = aspect_negatives
        asp_loss = self.aspect_contrastive_loss(anchor, pos, neg)
        sent_loss = self.sentiment_loss(out["z_sent"], sentiment_labels.to(self.device)) if sentiment_labels is not None else torch.tensor(0.0, device=self.device)

        total = self.lambda_aspect * asp_loss + self.lambda_sentiment * sent_loss
        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        return {"loss": total.detach(), "aspect_loss": asp_loss.detach(), "sentiment_loss": sent_loss.detach(), **out}


# =========================
# System
# =========================
class HyperbolicSegmentSystem(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)

        # ---- fixed defaults (args only provides .device) ----
        self.segment_candidates = 500
        self.segment_top_m = 4
        self.encode_batch_size = 64
        self.training_samples = 0
        self.learning_rate = 2e-5
        self.training_epochs = 1

        self.segment_encoder = SegmentEmbeddingModel()
        if args.device:
            dev = torch.device(args.device)
            self.segment_encoder.to(dev)
            self.segment_encoder.device = dev

        self.model_ready = False
        self.index = None
        self.llm_silver_aspect = []
        self.llm_bronze_aspect = []
        self._segment_text_cache = {}
        self._ensure_model_ready()
        self._train_with_segments()

    # ------- Recommend (global index; city arg ignored) -------
    def recommend(self, request, city=None, top_k=None):
        if not request:
            return []
        self._ensure_model_ready()
        index = self._ensure_index()
        if not index:
            return []

        q = str(request).strip()
        if not q:
            return []
        q_emb = self.segment_encoder.get_aspect_embedding(q)
        if not q_emb.numel():
            return []

        q_ball = self.segment_encoder.expmap0(q_emb)
        dists = self.segment_encoder.hyperbolic_distance(q_ball, index["ball"]).squeeze(-1)
        if not dists.numel():
            return []

        scores = 1.0 / (1.0 + dists)
        limit = min(len(scores), self.segment_candidates)
        if limit <= 0:
            return []

        vals, idxs = torch.topk(scores, limit, largest=True)
        grouped = {}
        for sc, i in zip(vals.tolist(), idxs.tolist()):
            it = index["items"][i]
            seg = index["segments"][i]
            grouped.setdefault(it, []).append((float(sc), float(dists[i].item()), seg))

        if not grouped:
            return []

        res = []
        for it, rows in grouped.items():
            rows.sort(key=lambda r: (-r[0], r[1]))
            keep = rows[:self.segment_top_m] if self.segment_top_m > 0 else rows
            total = sum(r[0] for r in keep)
            ev, snips = [], []
            for sc, di, s in keep:
                sid = s.get("segment_id")
                if sid: ev.append(sid)
                sn = self._normalize_text(s.get("text"))
                if sn: snips.append(sn)
            res.append({
                "item_id": it,
                "model_score": float(total),
                "evidence": ev,
                "short_excerpt": self._compose_excerpt(snips),
                "full_explanation": self._compose_explanation(snips),
            })

        res.sort(key=lambda e: (-e["model_score"], e["item_id"]))
        k = top_k or self.default_top_k
        return res[:k] if k and k > 0 else res

    # ------- Minimal readiness (no load/save plumbing) -------
    def _ensure_model_ready(self):
        if self.model_ready:
            return
        self.segment_encoder.eval()
        self.model_ready = True

    # ------- Train (kept minimal; off by default) -------
    def _train_with_segments(self):
        llm_pairs = self._gather_llm_aspect_pairs()
        if self.training_samples <= 0 and not llm_pairs.get("all"):
            return
        pairs = self._collect_training_pairs(self.training_samples) if self.training_samples > 0 else []
        anchors, positives = [], []
        for anchor, positive in pairs:
            a_text = self._normalize_text(anchor.get("text"))
            b_text = self._normalize_text(positive.get("text"))
            if a_text and b_text:
                anchors.append(a_text)
                positives.append(b_text)
        for a_text, b_text in llm_pairs.get("all", []):
            anchors.append(a_text)
            positives.append(b_text)
        if not anchors:
            return

        negatives = positives[1:] + positives[:1]
        negatives = negatives[:len(anchors)]
        tok = self.segment_encoder.tokenizer
        dev = self.segment_encoder.device
        opt = torch.optim.Adam(self.segment_encoder.parameters(), lr=float(self.learning_rate))
        epochs = self.training_epochs if self.training_epochs > 0 else 1

        A = {k: v.to(dev) for k, v in tok(anchors, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length).items()}
        P = {k: v.to(dev) for k, v in tok(positives, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length).items()}
        N = {k: v.to(dev) for k, v in tok(negatives, return_tensors="pt", padding=True, truncation=True, max_length=self.segment_encoder.max_length).items()}

        if not llm_pairs.get("all"):
            print("[train] llm audit pairs unavailable; distance tracking skipped.")

        for epoch in range(epochs):
            before = self._distance_metrics(llm_pairs) if llm_pairs.get("all") else {}
            self.segment_encoder.train()
            oA = self.segment_encoder.forward(A["input_ids"], A["attention_mask"])
            oP = self.segment_encoder.forward(P["input_ids"], P["attention_mask"])
            oN = self.segment_encoder.forward(N["input_ids"], N["attention_mask"])
            loss = self.segment_encoder.aspect_contrastive_loss(oA["z_asp"], oP["z_asp"].unsqueeze(1), oN["z_asp"].unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.segment_encoder.eval()
            after = self._distance_metrics(llm_pairs) if llm_pairs.get("all") else {}
            self._log_distance_progress(epoch, before, after)

        self.segment_encoder.eval()

    def _gather_llm_aspect_pairs(self):
        silver = []
        for entry in self.llm_silver_aspect:
            a_text = self._resolve_segment_text(entry.get("anchor_segment"))
            b_text = self._resolve_segment_text(entry.get("neighbor_segment"))
            if a_text and b_text:
                silver.append((a_text, b_text))
        bronze = []
        for entry in self.llm_bronze_aspect:
            a_text = self._resolve_segment_text(entry.get("anchor_segment"))
            b_text = self._resolve_segment_text(entry.get("neighbor_segment"))
            if a_text and b_text:
                bronze.append((a_text, b_text))
        combined = silver + bronze
        return {"silver": silver, "bronze": bronze, "all": combined}

    def _resolve_segment_text(self, segment_id):
        if not segment_id:
            return ""
        cached = self._segment_text_cache.get(segment_id)
        if cached is not None:
            return cached
        text = ""
        lookup = self.segment_lookup.get(segment_id)
        if lookup:
            text = self._normalize_text(lookup.get("text"))
        if not text:
            for group in self.item_segments.values():
                for segment in group:
                    if segment.get("segment_id") == segment_id:
                        text = self._normalize_text(segment.get("text"))
                        if text:
                            break
                if text:
                    break
        self._segment_text_cache[segment_id] = text
        return text

    def _distance_metrics(self, llm_pairs):
        metrics = {}
        pool = {}
        ordered = []
        for tier in ("silver", "bronze"):
            for anchor, positive in llm_pairs.get(tier, []):
                if anchor and anchor not in pool:
                    pool[anchor] = len(ordered)
                    ordered.append(anchor)
                if positive and positive not in pool:
                    pool[positive] = len(ordered)
                    ordered.append(positive)
        if not ordered:
            return metrics
        embeddings = self.segment_encoder.encode_texts(ordered, batch_size=self.encode_batch_size, logmap=False)
        if not embeddings.numel():
            return metrics
        for tier, pairs in llm_pairs.items():
            if not pairs:
                continue
            anchor_idx = torch.tensor([pool[a] for a, _ in pairs], dtype=torch.long)
            positive_idx = torch.tensor([pool[b] for _, b in pairs], dtype=torch.long)
            anchor_vecs = embeddings.index_select(0, anchor_idx)
            positive_vecs = embeddings.index_select(0, positive_idx)
            dists = self.segment_encoder.hyperbolic_distance(anchor_vecs, positive_vecs)
            if dists.numel():
                metrics[tier] = float(dists.mean().item())
        return metrics

    def _log_distance_progress(self, epoch, before, after):
        if not before and not after:
            return
        pos = epoch + 1
        for tier in ("silver", "bronze", "all"):
            b_val = before.get(tier) if before else None
            a_val = after.get(tier) if after else None
            if b_val is None and a_val is None:
                continue
            parts = []
            if b_val is not None:
                parts.append(f"before={b_val:.4f}")
            if a_val is not None:
                parts.append(f"after={a_val:.4f}")
            if b_val is not None and a_val is not None:
                parts.append(f"delta={a_val - b_val:.4f}")
            print(f"[train] epoch {pos} {tier} poincare {' '.join(parts)}")

    # ------- LLM thresholding (unchanged semantics) -------
    def calibrate_threshold_with_llm(self, candidates, provisional_tau=None, sample_size=200, target_precision_lb=0.9, sentiment_conf_min=0.7, double_judge=False, model="gpt-4.1-mini", confidence=0.95, max_iterations=6):
        usable = []
        for e in candidates or []:
            s = e.get("score")
            a = e.get("anchor")
            b = e.get("neighbor")
            at = self._normalize_text(a.get("text")) if a else ""
            bt = self._normalize_text(b.get("text")) if b else ""
            if s is None or not at or not bt:
                continue
            usable.append({"score": float(s), "anchor": a, "neighbor": b, "anchor_text": at, "neighbor_text": bt})

        if not usable:
            return float(provisional_tau or 0.0), [], [], [], []

        scores = [u["score"] for u in usable]
        lo, hi = min(scores), max(scores)
        tau = float(provisional_tau) if provisional_tau is not None else sum(scores) / len(scores)
        tau = min(max(tau, lo), hi)

        best = None
        cache = {}
        prev = None

        for _ in range(max_iterations):
            if prev is not None and abs(tau - prev) < 1e-4:
                break
            prev = tau

            qualified = [u for u in usable if u["score"] >= tau]
            if not qualified:
                hi = tau; tau = (lo + hi) / 2.0; continue

            qualified.sort(key=lambda x: abs(x["score"] - tau))
            sample = qualified[:sample_size] if len(qualified) > sample_size else qualified
            audits = self._ensure_llm_audits(sample, cache, model, double_judge)

            pos = sum(1 for k in audits if audits[k]["aspect_same"])
            tot = len(audits)
            if tot == 0:
                hi = tau; tau = (lo + hi) / 2.0; continue

            lb = self._wilson_lower_bound(pos, tot, confidence)
            if lb >= target_precision_lb:
                best = tau; hi = tau
            else:
                lo = tau
            tau = (lo + hi) / 2.0

        final_tau = max(min(best if best is not None else hi, hi), lo)
        elig = [u for u in usable if u["score"] >= final_tau]
        self._ensure_llm_audits(elig, cache, model, double_judge)

        silver_aspect, bronze_aspect = [], []
        silver_sent, bronze_sent = {}, {}

        for e in elig:
            key = self._pair_key(e)
            audit = cache.get(key)
            a, b = e["anchor"], e["neighbor"]
            aid = self._segment_identifier(a, e["anchor_text"])
            bid = self._segment_identifier(b, e["neighbor_text"])
            rec = {"anchor_segment": aid, "neighbor_segment": bid, "score": e["score"]}

            if audit and audit["aspect_same"]:
                rec["aspect_label"] = audit.get("aspect_label")
                silver_aspect.append(rec)
            else:
                bronze_aspect.append(rec)

            if audit:
                sa, sb = audit["sentiments"]
                if sa:
                    cont = silver_sent if sa["confidence"] >= sentiment_conf_min else bronze_sent
                    cont[aid] = self._merge_sentiments(cont.get(aid, {"segment_id": aid, "label": "NEU", "score": 0.0, "confidence": 0.0}), {"segment_id": aid, **sa})
                if sb:
                    cont = silver_sent if sb["confidence"] >= sentiment_conf_min else bronze_sent
                    cont[bid] = self._merge_sentiments(cont.get(bid, {"segment_id": bid, "label": "NEU", "score": 0.0, "confidence": 0.0}), {"segment_id": bid, **sb})

        silver_list = sorted(silver_sent.values(), key=lambda r: (-r["confidence"], -abs(r["score"]), r["segment_id"]))
        bronze_list = sorted(bronze_sent.values(), key=lambda r: (-r["confidence"], -abs(r["score"]), r["segment_id"]))
        silver_aspect.sort(key=lambda r: (-r["score"], r["anchor_segment"], r["neighbor_segment"]))
        bronze_aspect.sort(key=lambda r: (-r["score"], r["anchor_segment"], r["neighbor_segment"]))
        self.llm_silver_aspect = list(silver_aspect)
        self.llm_bronze_aspect = list(bronze_aspect)
        self._segment_text_cache = {}
        self._train_with_segments()
        return final_tau, silver_aspect, bronze_aspect, silver_list, bronze_list

    def _merge_sentiments(self, a, b):
        wa, wb = a.get("confidence", 0.0), b.get("confidence", 0.0)
        tot = wa + wb
        score = (a.get("score", 0.0) * wa + b.get("score", 0.0) * wb) / tot if tot > 0 else 0.5 * (a.get("score", 0.0) + b.get("score", 0.0))
        conf = tot / 2.0 if tot > 0 else max(a.get("confidence", 0.0), b.get("confidence", 0.0))
        label = b.get("label") if wb >= wa else a.get("label")
        return {"segment_id": a.get("segment_id") or b.get("segment_id"), "label": label, "score": score, "confidence": conf, "source": "llm"}

    def _segment_identifier(self, segment, fallback):
        sid = segment.get("segment_id")
        if sid: return sid
        alt = segment.get("id") or segment.get("segment")
        if alt: return alt
        item = segment.get("item_id")
        if item:
            txt = self._normalize_text(segment.get("text"))
            if txt: return f"{item}:{txt}"
        return fallback

    def _ensure_llm_audits(self, entries, cache, model, double_judge):
        prompts, keys = [], []
        for e in entries:
            k = self._pair_key(e)
            if k in cache: continue
            prompts.append(self._build_llm_prompt(e["anchor_text"], e["neighbor_text"]))
            keys.append(k)

        if prompts:
            r1 = run_llm_batch(prompts, "segment_aspect_sentiment_audit", model=model, temperature=0.0, use_json=True)
            p1 = [self._parse_audit_output(t) for t in r1]
            if double_judge:
                r2 = run_llm_batch(prompts, "segment_aspect_sentiment_audit_double", model=model, temperature=0.0, use_json=True)
                p2 = [self._parse_audit_output(t) for t in r2]
                p1 = [self._combine_audits(a, b) for a, b in zip(p1, p2)]
            for k, v in zip(keys, p1):
                cache[k] = v

        return {self._pair_key(e): cache[self._pair_key(e)] for e in entries if self._pair_key(e) in cache}

    def _combine_audits(self, a, b):
        if not a and not b:
            return {"aspect_same": False, "aspect_label": "", "sentiments": [None, None]}
        if not a: return b
        if not b: return a
        same = bool(a.get("aspect_same")) and bool(b.get("aspect_same"))
        label = a.get("aspect_label") if same else ""
        if not label and same: label = b.get("aspect_label") or ""
        s1, s2 = a.get("sentiments") or [None, None], b.get("sentiments") or [None, None]
        return {"aspect_same": same, "aspect_label": label, "sentiments": [self._merge_sentiment_entries(s1[0], s2[0]), self._merge_sentiment_entries(s1[1], s2[1])]}

    def _merge_sentiment_entries(self, x, y):
        if x is None and y is None: return {"label": "NEU", "score": 0.0, "confidence": 0.0}
        if x is None: return y
        if y is None: return x
        wx, wy = x.get("confidence", 0.0), y.get("confidence", 0.0)
        tot = wx + wy
        score = (x.get("score", 0.0) * wx + y.get("score", 0.0) * wy) / tot if tot > 0 else 0.5 * (x.get("score", 0.0) + y.get("score", 0.0))
        conf = tot / 2.0 if tot > 0 else max(x.get("confidence", 0.0), y.get("confidence", 0.0))
        label = x.get("label") if wx >= wy else y.get("label")
        return {"label": label, "score": score, "confidence": conf}

    def _pair_key(self, e):
        return (str(self._segment_identifier(e["anchor"], e["anchor_text"])), str(self._segment_identifier(e["neighbor"], e["neighbor_text"])))

    def _build_llm_prompt(self, a, b):
        return "\n".join([
            "You are auditing two review segments.",
            "For each pair, answer in JSON with keys: aspect_same (YES or NO), aspect_label (short phrase), segment_a, segment_b.",
            "segment_a and segment_b must each contain label (NEG/NEU/POS), score (between -1 and 1), confidence (between 0 and 1).",
            "Decide aspect_same by focusing on the primary aspect described, ignoring sentiment differences.",
            "Use empty string for aspect_label when aspect_same is NO.",
            f"Segment A: {self._truncate_for_prompt(a)}",
            f"Segment B: {self._truncate_for_prompt(b)}",
        ])

    def _truncate_for_prompt(self, text, limit=480):
        t = text.strip()
        return t if len(t) <= limit else f"{t[:limit].rstrip()}…"

    def _parse_audit_output(self, text):
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"): s = s[4:].strip()
        data = json.loads(s)
        flag = str(data.get("aspect_same", "")).strip().upper() == "YES"
        label = self._normalize_text(data.get("aspect_label")) if flag else ""
        a = self._normalize_sentiment_entry(data.get("segment_a"))
        b = self._normalize_sentiment_entry(data.get("segment_b"))
        return {"aspect_same": flag, "aspect_label": label, "sentiments": [a, b]}

    def _normalize_sentiment_entry(self, p):
        lab = str(p.get("label", "NEU")).strip().upper()
        if lab not in {"NEG", "NEU", "POS"}: lab = "NEU"
        sc = float(p.get("score", 0.0))
        cf = float(p.get("confidence", 0.0))
        sc = max(min(sc, 1.0), -1.0)
        cf = max(min(cf, 1.0), 0.0)
        return {"label": lab, "score": sc, "confidence": cf}

    # ------- Data prep -------
    def _collect_training_pairs(self, limit):
        if limit <= 0:
            return []
        pairs = []
        for item_id in sorted(self.item_segments.keys()):
            segs = [s for s in self.item_segments[item_id] if s.get("text")]
            if len(segs) < 2: continue
            base = segs[0]
            for other in segs[1:]:
                pairs.append((base, other))
                if len(pairs) >= limit: return pairs
        return pairs

    # ------- Global index (replaces _ensure_city_index) -------
    def _ensure_index(self):
        if self.index is not None:
            return self.index

        segs = []
        for it in self.item_segments.keys():
            segs.extend(self.item_segments[it])

        seen, cleaned = set(), []
        for s in segs:
            sid = s.get("segment_id")
            t = s.get("text")
            if sid and t and sid not in seen:
                seen.add(sid)
                cleaned.append(s)

        if not cleaned:
            self.index = None
            return None

        texts = [s.get("text") for s in cleaned]
        aspects = self.segment_encoder.encode_texts(texts, batch_size=self.encode_batch_size, logmap=True)
        if not aspects.numel():
            self.index = None
            return None

        with torch.no_grad():
            ball = self.segment_encoder.expmap0(aspects)

        self.index = {"tangent": aspects, "ball": ball, "segments": cleaned, "items": [s.get("item_id") for s in cleaned]}
        return self.index
