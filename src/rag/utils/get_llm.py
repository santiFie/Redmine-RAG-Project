import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

_LLM_REGISTRY = {
    "groq": lambda: Groq(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        api_key=os.getenv("GROQ_API_KEY", ""),
    ),
    "nvidia": lambda: __import__("llama_index.llms.nvidia", fromlist=["NVIDIA"]).NVIDIA(
        model=os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
        api_key=os.getenv("NVIDIA_API_KEY", ""),
    ),
    "openrouter": lambda: __import__("llama_index.llms.openai_like", fromlist=["OpenAILike"]).OpenAILike(
        model=os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        api_base="https://openrouter.ai/api/v1",
        is_chat_model=True,
    ),
}