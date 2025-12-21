# bug_finder_agent test

# from agents.bug_finder_agent import detect_bugs

# code = """
# def divide(a, b):
#     return a / b

# print(divide(10, 0))
# """

# result = detect_bugs(code, "test.py")
# print(result)




# summary_agent test

from agents.summary_agent import summarize_file

code = """
def divide(a, b):
    return a / b

print(divide(10, 0))
"""

result = summarize_file(code)
print(result)




# security_agent test

# from agents.security_agent import analyze_security

# code = """
# import os

# password = "admin123"

# def run(cmd):
#     os.system(cmd)
# """

# result = analyze_security(code, "danger.py")
# print(result)




# fix_suggestor_agent test

# from agents.fix_suggestor_agent import suggest_fixes

# code = """
# def divide(a, b):
#     return a / b
# """

# print(suggest_fixes(code, "math_utils.py"))




# test_writer_agent test

# from agents.test_writer_agent import write_tests

# code = """
# def add(a, b):
#     return a + b
# """

# print(write_tests(code, "calc.py"))

