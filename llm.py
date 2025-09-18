# llm.py
import re
import json
import logging
import asyncio
import openai
from openai import AsyncOpenAI
from utils import readf
from tqdm.asyncio import tqdm as tqdm_asyncio

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

# ---- query (sync) ----
def query_llm(prompt, model="gpt-5-nano", temperature=0.1, verbose=False):
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

# ---- query (async) ----
async def query_llm_async(prompt, model="gpt-5-nano", temperature=0.1, sem=None, verbose=False, return_usage=False):
    """
    If return_usage=True, returns (content, pt, ct); else returns content.
    """
    client = get_async_openai_client()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # allow None semaphore
    if sem is None:
        sem = asyncio.Semaphore(999999)

    async with sem:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": model, "messages": messages}
        if not model.lower().startswith("gpt-5"):
            kwargs["temperature"] = temperature

        resp = await client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        pt, ct = _extract_usage(resp)
        if verbose:
            cost = _estimate_cost_usd(model, pt, ct)
            print(f"[LLM] model={model} prompt_tokens={pt} completion_tokens={ct} est_cost=${cost:.6f}")

        return (content, pt, ct) if return_usage else content

# ---- batch ----
def run_llm_batch(prompts, model="gpt-4.1-mini", temperature=0.1, num_workers=8, verbose=False):
    """
    - When verbose=False: fast path, returns list[str] contents.
    - When verbose=True: shows tqdm progress bar and prints final totals; returns list[str] contents.
    """
    async def _runner():
        sem = asyncio.Semaphore(num_workers)

        async def one(idx, p, with_usage):
            try:
                result = await query_llm_async(
                    p,
                    model=model,
                    temperature=temperature,
                    sem=sem,
                    verbose=False,
                    return_usage=with_usage,
                )
            except Exception as e:
                logging.error(f"LLM query failed: {e}")
                result = ("{}", 0, 0) if with_usage else "{}"

            if with_usage:
                content, pt, ct = result
                return idx, content, pt, ct
            return idx, result

        if verbose:
            tasks = [one(i, p, True) for i, p in enumerate(prompts)]
            outs = [None] * len(prompts)
            total_pt = 0
            total_ct = 0
            for fut in tqdm_asyncio.as_completed(tasks, total=len(prompts), desc="LLM batch", ncols=88):
                idx, content, pt, ct = await fut
                outs[idx] = content
                total_pt += pt
                total_ct += ct
            total_cost = _estimate_cost_usd(model, total_pt, total_ct)
            print(
                f"[LLM] batch complete. prompt_tokens={total_pt} "
                f"completion_tokens={total_ct} est_cost=${total_cost:.6f}"
            )
            return outs
        
        tasks = [one(i, p, False) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda item: item[0])
        return [content for _, content in results]

    return asyncio.run(_runner())

# ---- utils ----
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
