# llm.py
import re
import json
import logging
import asyncio
import openai
from openai import AsyncOpenAI
from utils import readf

# ---- clients ----
openai_client = None
async_openai_client = None

def get_openai_client():
    """Return a singleton OpenAI client (sync)."""
    global openai_client
    if openai_client is None:
        openai_client = openai.OpenAI(api_key=readf(".openaiapi_key"))
    return openai_client

def get_async_openai_client():
    """Return a singleton OpenAI client (async)."""
    global async_openai_client
    if async_openai_client is None:
        async_openai_client = AsyncOpenAI(api_key=readf(".openaiapi_key"))
    return async_openai_client

# --- pricing -----------------------------------------------------------------
# USD per token (input/output). Keep this map up to date with OpenAI pricing.
PRICING_PER_TOKEN = {
    # GPT-5 family (if you use them)
    "gpt-5":        {"in": 1.25/1_000_000, "out": 10.00/1_000_000},
    "gpt-5-mini":   {"in": 0.25/1_000_000, "out":  2.00/1_000_000},
    "gpt-5-nano":   {"in": 0.05/1_000_000, "out":  0.40/1_000_000},

    # GPT-4 family (common current SKUs)
    "gpt-4.1":      {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4.1-mini": {"in": 0.40/1_000_000, "out":  1.60/1_000_000},
    "gpt-4o":       {"in": 5.00/1_000_000, "out": 15.00/1_000_000},
    "gpt-4o-mini":  {"in": 0.50/1_000_000, "out":  1.50/1_000_000},
    "gpt-4-turbo":  {"in": 10.00/1_000_000,"out": 30.00/1_000_000},
}

def _resolve_rates(model: str):
    m = model.lower()
    if m in PRICING_PER_TOKEN:
        return PRICING_PER_TOKEN[m]
    # try family prefix match (e.g., gpt-4o-2024-xx)
    for k in PRICING_PER_TOKEN:
        if m.startswith(k):
            return PRICING_PER_TOKEN[k]
    return None

def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _resolve_rates(model)
    if not rates:
        return 0.0
    return prompt_tokens * rates["in"] + completion_tokens * rates["out"]

def _extract_usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    # OpenAI SDK may expose as attrs or dict-like
    pt = getattr(u, "prompt_tokens", None) or (u.get("prompt_tokens") if isinstance(u, dict) else 0) or 0
    ct = getattr(u, "completion_tokens", None) or (u.get("completion_tokens") if isinstance(u, dict) else 0) or 0
    return int(pt), int(ct)

# ---- query ----

def query_llm(prompt, model="gpt-4.1-mini", temperature=0.1, verbose=False):
    client = get_openai_client()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "messages": messages}
    if not model.lower().startswith("gpt-5"):
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content

    if verbose:
        pt, ct = _extract_usage(resp)
        cost = _estimate_cost_usd(model, pt, ct)
        print(f"[LLM] model={model} prompt_tokens={pt} completion_tokens={ct} est_cost=${cost:.6f}")

    return content


async def query_llm_async(prompt, model="gpt-4.1-mini", temperature=0.1, sem=None, verbose=False):
    client = get_async_openai_client()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    async with sem:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": model, "messages": messages}
        if not model.lower().startswith("gpt-5"):
            kwargs["temperature"] = temperature

        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        if verbose:
            pt, ct = _extract_usage(resp)
            cost = _estimate_cost_usd(model, pt, ct)
            print(f"[LLM] model={model} prompt_tokens={pt} completion_tokens={ct} est_cost=${cost:.6f}")

        return content

# ---- batch ----
def run_llm_batch(prompts, model="gpt-4.1-mini", temperature=0.1, num_workers=8, verbose=False):
    async def _runner():
        sem = asyncio.Semaphore(num_workers)
        totals = {"pt": 0, "ct": 0, "cost": 0.0}
        tasks = [query_llm_async(p, model=model, temperature=temperature, sem=sem, verbose=verbose) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outs = []
        for r in results:
            if isinstance(r, Exception):
                logging.error(f"LLM query failed: {r}")
                outs.append("{}")
            else:
                outs.append(r)
        # re-query usage for totals (only if verbose and model supports usage in responses)
        if verbose and outs:
            # lightweight second pass: run a tiny accumulator by re-calling _estimate on each task’s usage
            # Note: we already printed per-call; to avoid extra API calls, just inform totals unavailable unless tracked inline.
            print("[LLM] batch complete (totals shown above per-call).")
        return outs

    return asyncio.run(_runner())



def safe_json_parse(output_str):
    """
    Parse LLM output into a Python object.
    - Strips code fences / stray text.
    - Returns {} on failure.
    - Does NOT enforce a fixed schema; keeps whatever keys are present.
    """
    if not output_str:
        return {}

    s = output_str.strip()

    # remove code fences if present
    s = re.sub(r"^```(?:json)?", "", s)
    s = re.sub(r"```$", "", s).strip()

    try:
        data = json.loads(s)
    except Exception as e:
        logging.warning(f"JSON parse failed: {e} | text={s[:100]}...")
        return {}

    # only accept dicts/lists; otherwise return {}
    if isinstance(data, (dict, list)):
        return data
    return {}

