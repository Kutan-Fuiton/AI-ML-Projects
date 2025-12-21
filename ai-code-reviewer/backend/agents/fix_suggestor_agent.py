from llm.llm import get_llm
MAX_TOKEN = 400

llm = get_llm(MAX_TOKEN)

def suggest_fixes(code: str, filename: str):
    prompt = f"""
You are a senior software engineer.

Review the following code and:
- Suggest fixes for bugs
- Improve readability and maintainability
- Apply best practices
- Suggest safer alternatives where needed

File name: {filename}

Code:
{code}

Provide:
1. A short explanation
2. A corrected or improved code snippet
"""
    return llm.invoke(prompt)
