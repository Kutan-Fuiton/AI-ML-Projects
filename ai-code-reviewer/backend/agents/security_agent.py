from llm.llm import get_llm
MAX_TOKEN = 400

llm = get_llm(MAX_TOKEN)

def analyze_security(code: str, filename: str):
    prompt = f"""
You are a security-focused software engineer.

Analyze the following code for security vulnerabilities, including but not limited to:
- SQL Injection
- Command Injection
- Insecure deserialization
- Hardcoded credentials or secrets
- Unsafe use of eval/exec
- Improper input validation
- Path traversal
- Authentication or authorization issues

File name: {filename}

Code:
{code}

Respond in bullet points.
If no security issues are found, explicitly say so.
"""
    return llm.invoke(prompt)
