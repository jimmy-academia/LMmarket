
def query_llm(prompt, model="openai"):
    if model == "openai":
        from openai import ChatCompletion
        response = ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response['choices'][0]['message']['content']
    
    elif model == "gemini":
        import google.generativeai as genai
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"Unsupported LLM model: {model}")


def safe_json_extract(prompt, model="openai"):
    try:
        response = query_llm(prompt, model=model)
        return json.loads(response)
    except Exception as e:
        print(f"JSON parse error: {e}")
        return {}
