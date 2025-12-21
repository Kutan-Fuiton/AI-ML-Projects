from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="deepseek-ai/deepseek-coder-1.3b-instruct",
    max_new_tokens=200
)

print(pipe("Write a Python function to reverse a string")[0]["generated_text"])
