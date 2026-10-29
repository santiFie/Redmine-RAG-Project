import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_typesafe import TypeSafeClassifier
from pydantic import SecretStr


def get_llm(
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0,
    **kwargs: Any,
):
    provider = provider.lower()

    _REGISTRY = {
        "groq": lambda: ChatGroq(model=model, temperature=temperature, timeout=120, **kwargs),
        "nvidia": lambda: ChatOpenAI(
            model=model,
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=SecretStr(str(os.getenv("NVIDIA_API_KEY", ""))),
            temperature=temperature,
            timeout=120,
            **kwargs,
        ),
        "openrouter": lambda: ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            api_key=SecretStr(str(os.getenv("OPENROUTER_API_KEY", ""))),
            **kwargs,
        ),
        "huggingface": lambda: ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                model=model,
                temperature=temperature,
                huggingfacehub_api_token=str(os.getenv("HUGGINGFACE_API_KEY", "")),
                task="text-generation",
                **kwargs,
            ),
        ),
        "gemini": lambda: ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=str(os.getenv("GOOGLE_API_KEY")),
            timeout=120,
            **kwargs,
        ),
    }

    factory = _REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Provider {provider} not supported")

    return factory()


def get_typesafe_classifier(
    model: str = "openai/gpt-6-luna-decisions",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> TypeSafeClassifier:
    """
    Factory para instanciar TypeSafeClassifier compatible con OpenRouter y TypeSafe AI.

    Para OpenRouter, los endpoints de decisiones (System One) se resuelven bajo
    `/v1/systemone` a partir del base_url proporcionado (por defecto: `https://openrouter.ai/api`).
    """
    resolved_base_url = base_url or "https://openrouter.ai/api"
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY") or "dummy-key-for-mock"
    return TypeSafeClassifier(
        model=model,
        base_url=resolved_base_url,
        api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
        timeout=timeout,
    )
