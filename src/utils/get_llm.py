from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import SecretStr
import os


def get_llm(
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
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
        "huggingface": lambda: ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                model=model,
                temperature=temperature,
                huggingfacehub_api_token=str(os.getenv("HUGGINGFACE_API_KEY", "")),
                task="text-generation",
            ),
        ),
        "gemini": lambda: ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=str(os.getenv("GOOGLE_API_KEY")),
            timeout=120,
        ),
    }

    factory = _REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Provider {provider} not supported")

    return factory()