from llm.llm import get_llm
MAX_TOKEN = 400

llm = get_llm(MAX_TOKEN)

def detect_bugs(code: str, filename: str):
    prompt = f"""
You are a senior software engineer performing a professional code review.

Analyze the following file and identify:
- Bugs
- Logical errors
- Edge cases
- Bad practices

File name: {filename}

Code:
{code}

Respond in bullet points.
"""

    return llm.invoke(prompt)
