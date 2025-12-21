from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
TEMPERATURE = 0.01

_llm_cache = {}

def get_llm(max_new_tokens: int):
    """
    Returns an LLM instance with a specific token limit.
    Model is reused, only generation config changes.
    """
    global _llm_cache

    if max_new_tokens not in _llm_cache:
        hf_pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE
        )

        _llm_cache[max_new_tokens] = HuggingFacePipeline(
            pipeline=hf_pipeline
        )

    return _llm_cache[max_new_tokens]