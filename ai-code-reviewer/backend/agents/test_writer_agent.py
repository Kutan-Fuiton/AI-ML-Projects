from llm.llm import get_llm
MAX_TOKEN = 1500

llm = get_llm(MAX_TOKEN)

def write_tests(code: str, filename: str):
    prompt = f"""
You are a software engineer writing unit tests.

Given the following code:
- Identify the main functions
- Write unit tests using pytest
- Cover normal cases and edge cases

File name: {filename}

Code:
{code}

Output only the test code.
"""
    return llm.invoke(prompt)
