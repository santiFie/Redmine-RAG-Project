from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import SecretStr
import os


def get_llm(
    provider: str = "groq",
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0,
):
    provider = provider.lower()

    _REGISTRY = {
        "groq": lambda: ChatGroq(model=model, temperature=temperature, timeout=120),
        "nvidia": lambda: ChatNVIDIA(
            model=model,
            temperature=temperature,
            nvidia_api_key=str(os.getenv("NVIDIA_API_KEY")),
            timeout=120,
        ),
        "openrouter": lambda: ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            api_key=SecretStr(str(os.getenv("OPENROUTER_API_KEY", ""))),
        ),
    }

    factory = _REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Provider {provider} not supported")

    return factory()