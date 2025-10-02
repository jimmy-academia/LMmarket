from .base import BaseSystem
from .sparse import BM25Baseline
from .dense import DenseRetrieverBaseline
from llm import query_llm, safe_json_parse

KEYWORD_PROMPT = """You are an expert retrieval strategist helping with restaurant review search.\nUser request: {request_text}\nSummary of previous retrieval insights: {previous_summary}\nPropose focused search keywords that can retrieve better supporting evidence.\nReturn strict JSON with keys: keywords (array of 1-4 short keywords or phrases) and reasoning (one sentence describing why they were chosen)."""

REVIEW_PROMPT = """You are analyzing retrieval results for an iterative search refinement process.\nUser request: {request_text}\nSearch query that was just executed: {search_query}\nTop retrieval results:\n{result_block}\nWrite a short reflection capturing what these results reveal and how to adjust the next search direction.\nReturn strict JSON with keys: reasoning (2-3 sentences summarizing takeaways) and guidance (one sentence describing what to focus on next)."""

KEYWORD_JSON_SCHEMA = {
    "name": "react_keyword_plan",
    "schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["keywords", "reasoning"],
        "additionalProperties": False,
    },
}

REVIEW_JSON_SCHEMA = {
    "name": "react_review_feedback",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "guidance": {"type": "string"},
        },
        "required": ["reasoning", "guidance"],
        "additionalProperties": False,
    },
}


class ReactRetrievalBaseline(BaseSystem):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.react_iterations = 5
        model_name = getattr(args, "react_model", "gpt-4.1-mini")
        temperature = getattr(args, "react_temperature", 0)
        self.react_model = model_name
        self.react_temperature = temperature
        retriever = getattr(args, "react_retriever", "bm25")
        if isinstance(retriever, str):
            retriever = retriever.strip().lower()
        else:
            retriever = "bm25"
        if retriever not in {"bm25", "dense"}:
            retriever = "bm25"
        self.react_retriever = retriever
        summary_k = getattr(args, "react_summary_k", 3)
        if not isinstance(summary_k, int) or summary_k <= 0:
            summary_k = 3
        self.react_summary_k = summary_k
        self._bm25_helper = None
        self._dense_helper = None

    def recommend(self, request, city=None, top_k=None):
        query_text = self._extract_query(request)
        if not query_text:
            return []
        limit = top_k if isinstance(top_k, int) and top_k > 0 else getattr(self.args, "top_k", self.default_top_k)
        if not isinstance(limit, int) or limit <= 0:
            limit = self.default_top_k
        previous_summary = "No prior retrieval. Start from the core intent."
        last_results = []
        for idx in range(self.react_iterations):
            keywords, planning_reasoning = self._propose_keywords(query_text, previous_summary)
            search_query = self._compose_search_query(query_text, keywords)
            results = self._run_retrieval(search_query, city, limit)
            last_results = results
            display = results[:self.react_summary_k]
            print(f"[ReAct][Iter {idx + 1}] keywords: {keywords}")
            print(f"[ReAct][Iter {idx + 1}] retriever={self.react_retriever} query=\"{search_query}\"")
            print(f"[ReAct][Iter {idx + 1}] retrieved: {self._summarize_results(display)}")
            review = self._review_results(query_text, search_query, display)
            reasoning_text = review.get("reasoning") if isinstance(review, dict) else None
            guidance_text = review.get("guidance") if isinstance(review, dict) else None
            if reasoning_text:
                print(f"[ReAct][Iter {idx + 1}] reasoning: {reasoning_text}")
            else:
                print(f"[ReAct][Iter {idx + 1}] reasoning: {planning_reasoning}")
            if guidance_text:
                previous_summary = guidance_text
            elif reasoning_text:
                previous_summary = reasoning_text
            else:
                previous_summary = planning_reasoning
        return last_results

    def _extract_query(self, request):
        if isinstance(request, dict):
            value = request.get("query")
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    return cleaned
            return ""
        if isinstance(request, str):
            stripped = request.strip()
            if stripped:
                return stripped
        return ""

    def _propose_keywords(self, request_text, previous_summary):
        prompt = KEYWORD_PROMPT.format(request_text=request_text, previous_summary=previous_summary)
        raw = query_llm(
            prompt,
            model=self.react_model,
            temperature=self.react_temperature,
            json_schema=KEYWORD_JSON_SCHEMA,
            use_json=True,
        )
        parsed = safe_json_parse(raw)
        keywords = []
        reasoning = ""
        if isinstance(parsed, dict):
            raw_keywords = parsed.get("keywords")
            if isinstance(raw_keywords, list):
                keywords = [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()]
            reason_value = parsed.get("reasoning")
            if isinstance(reason_value, str):
                reasoning = reason_value.strip()
        if not keywords:
            keywords = self._fallback_keywords(request_text)
        if not reasoning:
            reasoning = "Focused keywords derived from the core request."
        return keywords, reasoning

    def _compose_search_query(self, request_text, keywords):
        unique = []
        seen = set()
        for word in keywords:
            lowered = word.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(word)
        joined = " ".join(unique)
        if joined:
            return f"{request_text} {joined}".strip()
        return request_text

    def _run_retrieval(self, search_query, city, limit):
        if not search_query:
            return []
        if self.react_retriever == "dense":
            helper = self._ensure_dense_helper()
            if helper:
                return helper.recommend(search_query, city=city, top_k=limit)
            return []
        helper = self._ensure_bm25_helper()
        if helper:
            return helper.recommend(search_query, city=city, top_k=limit)
        return []

    def _ensure_bm25_helper(self):
        if self._bm25_helper is None:
            self._bm25_helper = BM25Baseline(self.args, self.data)
        return self._bm25_helper

    def _ensure_dense_helper(self):
        if self._dense_helper is None:
            self._dense_helper = DenseRetrieverBaseline(self.args, self.data)
        return self._dense_helper

    def _summarize_results(self, results):
        if not results:
            return "no results"
        lines = []
        for idx, entry in enumerate(results, 1):
            item_id = entry.get("item_id")
            score = entry.get("model_score")
            excerpt = entry.get("short_excerpt")
            if not excerpt:
                alt = entry.get("full_explanation")
                if alt:
                    excerpt = alt
                else:
                    excerpt = ""
            snippet = self._shorten_excerpt(excerpt)
            if score is None:
                lines.append(f"{idx}. {item_id} - {snippet}")
            else:
                lines.append(f"{idx}. {item_id} (score={float(score):.4f}) - {snippet}")
        return " | ".join(lines)

    def _shorten_excerpt(self, text):
        if not text:
            return ""
        collapsed = " ".join(str(text).split())
        if len(collapsed) <= 120:
            return collapsed
        return f"{collapsed[:117].rstrip()}…"

    def _review_results(self, request_text, search_query, results):
        block_lines = []
        if results:
            for idx, entry in enumerate(results, 1):
                item_id = entry.get("item_id")
                excerpt = entry.get("short_excerpt")
                if not excerpt:
                    alt = entry.get("full_explanation")
                    if alt:
                        excerpt = alt
                    else:
                        excerpt = ""
                snippet = self._shorten_excerpt(excerpt)
                block_lines.append(f"{idx}. item_id={item_id} :: {snippet}")
        else:
            block_lines.append("(no results found)")
        block = "\n".join(block_lines)
        prompt = REVIEW_PROMPT.format(
            request_text=request_text,
            search_query=search_query,
            result_block=block,
        )
        raw = query_llm(
            prompt,
            model=self.react_model,
            temperature=self.react_temperature,
            json_schema=REVIEW_JSON_SCHEMA,
            use_json=True,
        )
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"reasoning": "Results reviewed but no structured feedback returned.", "guidance": "Continue refining with descriptive cuisine or location terms."}

    def _fallback_keywords(self, request_text):
        tokens = request_text.split()
        unique = []
        seen = set()
        for token in tokens:
            cleaned = token.strip().lower()
            if not cleaned:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(token.strip())
            if len(unique) >= 3:
                break
        if unique:
            return unique
        return [request_text]
