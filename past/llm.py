import openai
import logging
from utils import readf
from debug import check

openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = openai.OpenAI(api_key=readf('.openaiapi_key'))
    return openai_client


def query_llm(prompt, model="openai", temperature=0.1):

    if model == "openai":
        client = get_openai_client()

        # Optional: suppress httpx info logs
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)

        # Create the message format expected by chat completion
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4.1-mini", #"gpt-4o-mini", 
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif model == "gemini":
        import google.generativeai as genai
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"Unsupported LLM model: {model}")

import json
import re

def extract_json_block(text):
    """
    Extract the first JSON object or array from the LLM output using regex.
    Supports optional wrapping with markdown fences.
    """
    # Try to match ```json ... ``` blocks first
    md_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if md_match:
        return md_match.group(1)

    # Fallback: try to extract first top-level JSON object or array
    json_match = re.search(r"({.*})", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Still nothing? Return the original text
    return text.strip()

def safe_json_extract(prompt, model="openai"):
    response = query_llm(prompt, model=model)

    # Clean it
    cleaned = extract_json_block(response)

    try:
        return json.loads(cleaned)
    except Exception as e:
        print(f"JSON parse error: {e}")
        print("---- RAW RESPONSE ----")
        print(repr(cleaned))
        print("----------------------")
        check()
        return {}
