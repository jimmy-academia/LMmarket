import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

class SegmentEmbeddingModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        print('in main model init')

        if config is None:
            config = {}
        backbone_name = config.get("backbone_name")
        if not backbone_name:
            backbone_name = "bert-base-uncased"
        self.pooling = config.get("pooling", "cls")
        self.max_length = config.get("max_length", 160)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.aspect_dim = config.get("aspect_dim", 64)
        self.sentiment_dim = config.get("sentiment_dim", 3)
        self.lambda_aspect = config.get("lambda_aspect", 1.0)
        self.lambda_sentiment = config.get("lambda_sentiment", 1.0)
        self.aspect_temperature = config.get("aspect_temperature", 0.1)
        self.sentiment_temperature = config.get("sentiment_temperature", 1.0)
        self.sentiment_loss_type = config.get("sentiment_loss", "ce")
        self.sentiment_margin = config.get("sentiment_margin", 1.0)
        self.curvature = config.get("curvature", 1.0)
        self.eps = 1e-5
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.encoder = AutoModel.from_pretrained(backbone_name)
        encoder_dim = self.encoder.config.hidden_size
        layers = [
            nn.Linear(encoder_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ]
        self.trunk = nn.Sequential(*layers)
        self.aspect_head = nn.Linear(self.hidden_dim, self.aspect_dim)
        self.sentiment_head = nn.Linear(self.hidden_dim, self.sentiment_dim)
        device = config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        if self.pooling == "mean":
            expanded_mask = attention_mask.unsqueeze(-1)
            masked = token_embeddings * expanded_mask
            summed = masked.sum(dim=1)
            counts = expanded_mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts
        else:
            pooled = token_embeddings[:, 0]
        trunk_output = self.trunk(pooled)
        tangent = self.aspect_head(trunk_output)
        z_asp = self.expmap0(tangent)
        z_sent = self.sentiment_head(trunk_output)
        return {"z_asp": z_asp, "z_sent": z_sent}

    def aspect_contrastive_loss(self, anchor, positives, negatives=None, temperature=None):
        temp = temperature if temperature is not None else self.aspect_temperature
        anchor = self.proj(anchor)
        positives = self._prepare_candidates(positives)
        negatives = self._prepare_candidates(negatives)
        anchor_expanded = anchor.unsqueeze(1)
        pos_dist = self.hyperbolic_distance(anchor_expanded, positives)
        pos_sim = torch.exp(-pos_dist / temp)
        pos_sum = pos_sim.sum(dim=1)
        denom = pos_sum
        if negatives is not None:
            neg_dist = self.hyperbolic_distance(anchor_expanded, negatives)
            neg_sim = torch.exp(-neg_dist / temp)
            denom = denom + neg_sim.sum(dim=1)
        ratio = pos_sum / denom.clamp(min=self.eps)
        loss = -torch.log(ratio.clamp(min=self.eps))
        return loss.mean()

    def sentiment_loss(self, pred, labels):
        if self.sentiment_loss_type == "mse":
            return F.mse_loss(pred, labels)
        if self.sentiment_loss_type == "margin":
            signed = labels.float()
            margin_term = self.sentiment_margin - signed * pred.squeeze(-1)
            return F.relu(margin_term).mean()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        return F.cross_entropy(pred, labels)

    def compute_multi_task_loss(self, anchor, positives, negatives, sentiment_pred, sentiment_labels, lambda_aspect=None, lambda_sentiment=None):
        asp_weight = lambda_aspect if lambda_aspect is not None else self.lambda_aspect
        sent_weight = lambda_sentiment if lambda_sentiment is not None else self.lambda_sentiment
        aspect_loss = self.aspect_contrastive_loss(anchor, positives, negatives)
        sentiment_loss = self.sentiment_loss(sentiment_pred, sentiment_labels)
        return asp_weight * aspect_loss + sent_weight * sentiment_loss

    def encode_texts(self, texts, batch_size=None, logmap=True, return_sentiment=False):
        if not texts:
            empty_aspect = torch.empty(0, self.aspect_dim)
            if return_sentiment:
                empty_sent = torch.empty(0, self.sentiment_dim)
                return empty_aspect, empty_sent
            return empty_aspect
        if batch_size is None or batch_size <= 0:
            batch_size = len(texts)
        aspects = []
        sentiments = []
        self.eval()
        for start in tqdm(range(0, len(texts), batch_size), ncols=88,desc="encode_texts"):
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.no_grad():
                outputs = self.forward(tokens["input_ids"], tokens["attention_mask"])
            asp = outputs["z_asp"]
            if logmap:
                asp = self.logmap0(asp)
            aspects.append(asp.cpu())
            if return_sentiment:
                sentiments.append(outputs["z_sent"].cpu())
        aspect_tensor = torch.cat(aspects, dim=0)
        if return_sentiment:
            sentiment_tensor = torch.cat(sentiments, dim=0) if sentiments else torch.empty(0, self.sentiment_dim)
            return aspect_tensor, sentiment_tensor
        return aspect_tensor

    def get_aspect_embedding(self, segment_text):
        embeddings = self.encode_texts([segment_text], batch_size=1, logmap=True)
        if embeddings.numel() == 0:
            return torch.empty(self.aspect_dim)
        return embeddings[0]

    def get_sentiment_embedding(self, segment_text):
        _, sentiments = self.encode_texts([segment_text], batch_size=1, logmap=False, return_sentiment=True)
        if sentiments.numel() == 0:
            return torch.empty(self.sentiment_dim)
        return sentiments[0]

    def score_aspect(self, u, v):
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=torch.float32)
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32)
        u_ball = self.expmap0(u)
        v_ball = self.expmap0(v)
        distance = self.hyperbolic_distance(u_ball, v_ball)
        return distance.squeeze().item()

    def logmap0(self, x):
        x = self.proj(x)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = math.sqrt(self.curvature)
        scale = torch.atanh(norm * sqrt_c) * 2.0 / (sqrt_c * norm)
        return scale * x

    def expmap0(self, v):
        norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        sqrt_c = math.sqrt(self.curvature)
        scaled = torch.tanh(norm * sqrt_c / 2.0) * v / (norm * sqrt_c)
        projected = self.proj(scaled)
        return projected

    def hyperbolic_distance(self, u, v):
        u = self.proj(u)
        v = self.proj(v)
        sqdist = self._sqnorm(u - v)
        u_norm = self._sqnorm(u)
        v_norm = self._sqnorm(v)
        denom = (1.0 - u_norm).clamp(min=self.eps) * (1.0 - v_norm).clamp(min=self.eps)
        argument = 1.0 + 2.0 * sqdist / denom
        argument = argument.clamp(min=1.0 + self.eps)
        return torch.acosh(argument).squeeze(-1)

    def mobius_add(self, x, y):
        c = self.curvature
        x2 = self._sqnorm(x)
        y2 = self._sqnorm(y)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c * c * x2 * y2
        result = numerator / denominator.clamp(min=self.eps)
        return self.proj(result)

    def proj(self, x):
        max_norm = (1 - self.eps)
        norm = x.norm(dim=-1, keepdim=True)
        factor = torch.where(norm >= max_norm, max_norm / norm, torch.ones_like(norm))
        return x * factor

    def _sqnorm(self, x):
        return (x * x).sum(dim=-1, keepdim=True)

    def _prepare_candidates(self, value):
        if value is None:
            return None
        if isinstance(value, list):
            tensors = []
            for entry in value:
                if entry is None:
                    continue
                if entry.dim() == 1:
                    tensors.append(entry.unsqueeze(0))
                else:
                    tensors.append(entry)
            if not tensors:
                return None
            value = torch.stack(tensors, dim=1)
        if value.dim() == 2:
            value = value.unsqueeze(1)
        return self.proj(value)

    def inference(self, batch, optimizer=None, sentiment_labels=None, aspect_positives=None, aspect_negatives=None, train=False):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.forward(input_ids, attention_mask)
        if not train:
            return outputs
        anchor = outputs["z_asp"]
        positives = aspect_positives if aspect_positives is not None else anchor.unsqueeze(1)
        negatives = aspect_negatives
        aspect_loss = self.aspect_contrastive_loss(anchor, positives, negatives)
        sentiment_loss = 0.0
        if sentiment_labels is not None:
            sentiment_loss = self.sentiment_loss(outputs["z_sent"], sentiment_labels.to(self.device))
        total = self.lambda_aspect * aspect_loss + self.lambda_sentiment * sentiment_loss
        if optimizer is not None:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
        sentiment_value = sentiment_loss if isinstance(sentiment_loss, float) else sentiment_loss.detach()
        return {"loss": total.detach(), "aspect_loss": aspect_loss.detach(), "sentiment_loss": sentiment_value, **outputs}
