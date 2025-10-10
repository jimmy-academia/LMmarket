# api.py
import io
import re
import time
import json
import logging
import asyncio
import openai
from openai import AsyncOpenAI
from utils import readf, dumpj 
from tqdm.asyncio import tqdm as tqdm_asyncio

from pathlib import Path

user_struct = lambda x: {"role": "user", "content": x}
system_struct = lambda x: {"role": "system", "content": x}
assistant_struct = lambda x: {"role": "assistant", "content": x}

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

def prep_msg(prompt):
    if type(prompt) == str:
        messages = [user_struct(prompt)]
    elif (
        isinstance(prompt, list)
        and prompt
        and all(isinstance(x, dict) and {"role", "content"} <= x.keys() for x in prompt)
    ):
        return prompt
    else:
        raise ValueError("prompt must be a string or a list of {'role', 'content'} dicts")
    return messages
    
# ---- query (sync) ----
def query_llm(prompt, model="gpt-5-nano", temperature=0.1, verbose=False, json_schema=None, use_json=False):
    client = get_openai_client()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    messages = prep_msg(prompt)
    kwargs = {"model": model, "messages": messages}
    if use_json:
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        else:
            kwargs["response_format"] = {"type": "json_object"}
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
async def query_llm_async(prompt, model="gpt-5-nano", temperature=0.1, sem=None, verbose=False, return_usage=False, json_schema=None, use_json=False):
    """
    If return_usage=True, returns (content, pt, ct); else returns content.
    """
    client = get_async_openai_client()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # allow None semaphore
    if sem is None:
        sem = asyncio.Semaphore(999999)

    async with sem:
        messages = [user_struct(prompt)]
        kwargs = {"model": model, "messages": messages}
        if use_json:
            if json_schema:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
            else:
                kwargs["response_format"] = {"type": "json_object"}
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
def run_llm_batch(prompts, task_name, model="gpt-4.1-mini", temperature=0.1, num_workers=8, verbose=False, json_schema=None, use_json=False):
    """
    - When verbose=False: fast path, returns list[str] contents.
    - When verbose=True: shows tqdm progress bar and prints final totals; returns list[str] contents.
    """
    print(f'[run_llm_batch] global task name: {task_name}')
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
                    json_schema=json_schema,
                    use_json=use_json,
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
    Parse LLM output into a Python object (dict or list).
    - Strips code fences / stray whitespace
    - Normalizes smart quotes
    - Fixes common LLM JSON glitches:
        * semicolons between objects -> commas
        * missing comma between `}{` -> `},{`
        * trailing commas before } or ]
    - Truncates to the last closing } or ] (to ignore "..."/logs after JSON)
    - Returns {} on failure
    """
    if output_str is None:
        return {}
    if isinstance(output_str, (dict, list)):
        return output_str

    s = str(output_str).strip()

    # remove leading/trailing code fences + surrounding whitespace
    s = re.sub(r"^\s*```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s).strip()

    # normalize smart quotes
    s = s.translate(str.maketrans({
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
    }))

    # common fixes
    # 1) semicolon between objects -> comma
    s = re.sub(r"}\s*;\s*{", "},{", s)

    # 2) missing comma between objects: `}{` -> `},{`
    s = re.sub(r"}\s*{", "},{", s)

    # 3) remove trailing commas before } or ]
    s = re.sub(r",\s*(?=[}\]])", "", s)

    # 4) keep only up to the last closing bracket/brace (drop logs like "...")
    last_close = max(s.rfind("]"), s.rfind("}"))
    if last_close != -1:
        s = s[:last_close + 1]

    try:
        data = json.loads(s)
        return data if isinstance(data, (dict, list)) else {}
    except Exception as e:
        # show a short context around the error to help debugging
        logging.warning(f"JSON parse failed: {e} | text={s}")
        return {}


def run_llm_batch_api(prompts, model="gpt-4.1-mini", temperature=0.1, verbose=False, poll_interval=5):
    """
    OpenAI Batch API: memory-only (no temp files). Returns list[str] in input order.
    Prints token totals and estimated cost when verbose=True.
    """
    client = get_openai_client()

    # Build JSONL (in-memory)
    lines = []
    m_is_gpt5 = model.lower().startswith("gpt-5")
    for i, p in enumerate(prompts):
        body = {"model": model, "messages": [user_struct(p)]}
        if not m_is_gpt5:
            body["temperature"] = temperature
        lines.append(json.dumps({
            "custom_id": f"prompt-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    # Upload input JSONL from memory
    buf = io.BytesIO(payload)
    buf.name = "batch.jsonl"  # some SDKs use this for filename
    uploaded = client.files.create(file=buf, purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    if verbose:
        print(f"[LLM batch] id={batch.id} status={getattr(batch,'status',None)}")

    # Poll
    status = getattr(batch, "status", None)
    while status in {"validating", "queued", "in_progress", "finalizing"}:
        time.sleep(max(1, poll_interval))
        batch = client.batches.retrieve(batch.id)
        status = getattr(batch, "status", None)
        if verbose:
            print(f"[LLM batch] status={status}")

    if status != "completed":
        logging.error(f"[LLM batch] ended with status={status}")
        return ["{}" for _ in prompts]

    # Fetch output (fall back to error file if present)
    fid = getattr(batch, "output_file_id", None) or getattr(batch, "error_file_id", None)
    if not fid:
        logging.error("[LLM batch] no output_file_id/error_file_id")
        return ["{}" for _ in prompts]

    content = client.files.content(fid)
    if hasattr(content, "text") and isinstance(content.text, str):
        raw = content.text
    elif hasattr(content, "read"):
        data = content.read()
        raw = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
    elif isinstance(content, (bytes, bytearray)):
        raw = content.decode("utf-8")
    else:
        raw = str(content)

    # Parse responses
    outputs, total_pt, total_ct = {}, 0, 0
    for line in raw.splitlines():
        if not line.strip(): 
            continue
        try:
            entry = json.loads(line)
        except Exception as e:
            logging.error(f"[LLM batch] parse error: {e}")
            continue

        cid = entry.get("custom_id")
        resp = (entry.get("response") or {})
        if resp.get("status_code") != 200:
            logging.error(f"[LLM batch] item {cid} status={resp.get('status_code')}")
            outputs[cid] = "{}"
            continue

        body = resp.get("body") or {}
        ch = (body.get("choices") or [])
        outputs[cid] = ch[0].get("message", {}).get("content", "") if ch else ""

        usage = body.get("usage") or {}
        total_pt += int(usage.get("prompt_tokens") or 0)
        total_ct += int(usage.get("completion_tokens") or 0)

    if verbose:
        cost = _estimate_cost_usd(model, total_pt, total_ct)
        print(f"[LLM batch] prompt_tokens={total_pt} completion_tokens={total_ct} est_cost=${cost:.6f}")

    # Align to input order
    return [outputs.get(f"prompt-{i}", "") for i in range(len(prompts))]
