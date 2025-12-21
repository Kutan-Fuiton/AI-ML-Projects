from llm.llm import get_llm
MAX_TOKEN = 300

llm = get_llm(MAX_TOKEN)

def summarize_file(code: str):
    prompt = f"""
You are a senior software engineer.
Explain what the following code does in simple terms:

{code}
"""
    return llm.invoke(prompt)