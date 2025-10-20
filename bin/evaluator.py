    # def _normalize_text(self, text):
    #     if not text:
    #         return ""
    #     collapsed = " ".join(str(text).split())
    #     return collapsed.strip()

    # def _compose_excerpt(self, summaries):
    #     if not summaries:
    #         return ""
    #     combined = " ".join(summaries)
    #     max_len = 160
    #     if len(combined) <= max_len:
    #         return combined
    #     trimmed = combined[:max_len].rstrip()
    #     return f"{trimmed}…"

    # def _compose_explanation(self, summaries):
    #     if not summaries:
    #         return ""
    #     parts = []
    #     for idx, summary in enumerate(summaries, 1):
    #         parts.append(f"{idx}) {summary}")
    #     return " ".join(parts)

    def _normalize_request(self, entry, index, group):
        base_id = index + 1
        prefix = group if group else "request"
        request_id = f"{prefix}_{base_id}"
        if isinstance(entry, str):
            query = entry.strip()
            if not query:
                return None
            query = self._correct_spelling(query)
            result = {
                "request_id": request_id,
                "query": query,
            }
            if group:
                result["group"] = group
            return result
        if isinstance(entry, dict):
            request = dict(entry)
            existing_id = request.get("request_id")
            if not existing_id:
                request["request_id"] = request_id
            if group and "group" not in request:
                request["group"] = group
            query = request.get("query")
            if isinstance(query, str):
                cleaned = query.strip()
                if cleaned:
                    request["query"] = self._correct_spelling(cleaned)
                else:
                    request.pop("query", None)
            query = request.get("query")
            if not query:
                return None
            topk = request.get("topk")
            if isinstance(topk, int) and topk > 0:
                request["topk"] = topk
            elif "topk" in request:
                request.pop("topk")
            return request
        return None

    def _prepare_requests(self):
        requests = []
        tests = self.test
        if isinstance(tests, list):
            for index, entry in enumerate(tests):
                normalized = self._normalize_request(entry, index, None)
                if normalized:
                    requests.append(normalized)
        elif isinstance(tests, dict):
            for group, entries in tests.items():
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    normalized = self._normalize_request(entry, index, group)
                    if normalized:
                        requests.append(normalized)
        return requests

    def evaluate(self, city=None, top_k=None):
        requests = self._prepare_requests()
        if not requests:
            print("[eval] no test requests available.")
            return
        if not hasattr(self, "recommend"):
            print("[eval] recommend() not implemented for this system.")
            return
        resolved_city = self.get_city_key(city)
        if not resolved_city:
            print("[eval] no city data available for evaluation.")
            return
        cutoff = top_k if isinstance(top_k, int) and top_k > 0 else self.default_top_k
        for request in requests:
            print('base.py', request)
            request_id = request.get("request_id")
            query = request.get("query")
            if not query:
                continue
            per_request_top_k = request.get("topk")
            if not isinstance(per_request_top_k, int) or per_request_top_k <= 0:
                per_request_top_k = cutoff
            results = self.recommend(query, city=resolved_city, top_k=per_request_top_k)
            if results is None:
                results = []
            packaged = []
            for entry in results:
                if isinstance(entry, dict):
                    item_id = entry.get("item_id")
                    if not item_id:
                        continue
                    score = entry.get("model_score")
                    evidence = entry.get("evidence")
                    if evidence is None:
                        evidence = []
                    short_excerpt = entry.get("short_excerpt")
                    full_explanation = entry.get("full_explanation")
                    packaged.append({
                        "item_id": item_id,
                        "model_score": score,
                        "evidence": evidence,
                        "short_excerpt": short_excerpt,
                        "full_explanation": full_explanation,
                    })
                else:
                    packaged.append({
                        "item_id": entry,
                        "model_score": None,
                        "evidence": [],
                        "short_excerpt": "",
                        "full_explanation": "",
                    })
            self.result[request_id] = {
                "request": {
                    "query": query,
                    "city": resolved_city,
                    "topk": per_request_top_k,
                },
                "candidates": packaged,
            }
            print(f"[eval] Request '{request_id}' in {resolved_city}")
            print(query)
            if not packaged:
                print("  (no candidates)")
                continue
            for rank, candidate in enumerate(packaged, 1):
                score = candidate.get("model_score")
                line = f"  {rank}. {candidate['item_id']}"
                if score is not None:
                    line += f" (score={float(score):.4f})"
                short_excerpt = candidate.get("short_excerpt")
                if short_excerpt:
                    line += f" — {short_excerpt}"
                print(line)

            print('pause after 1')
            break
