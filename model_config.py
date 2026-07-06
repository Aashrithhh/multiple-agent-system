"""
Central model configuration for the coding-agent lessons.

Supports two providers (configured via .env):
  1. Azure OpenAI  — set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_CHAT_API_KEY
  2. Google Gemini — set GEMINI_API_KEY (used as fallback or primary)

Set MODEL_PROVIDER=gemini or MODEL_PROVIDER=azure in .env to choose.
Defaults to gemini if both are configured.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Copy .env.example to .env and configure it."
        )
    return value


def _get_gemini_model():
    """Create a Gemini chat model via langchain-google-genai."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        google_api_key=_required("GEMINI_API_KEY"),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0")),
        max_output_tokens=int(os.getenv("CHAT_MAX_TOKENS", "6000")),
    )


def _get_azure_model():
    """Create an Azure OpenAI chat model."""
    from langchain_openai import AzureChatOpenAI

    endpoint = _required("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = _required("AZURE_OPENAI_CHAT_API_KEY")

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=_required("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        api_version="2024-12-01-preview",
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "6000")),
    )


def get_chat_model():
    """Return the configured chat model based on MODEL_PROVIDER env var."""
    provider = os.getenv("MODEL_PROVIDER", "").lower()

    # Explicit choice
    if provider == "azure":
        return _get_azure_model()
    if provider == "gemini":
        return _get_gemini_model()

    # Auto-detect: prefer Gemini (simpler setup), fall back to Azure
    if os.getenv("GEMINI_API_KEY"):
        return _get_gemini_model()
    if os.getenv("AZURE_OPENAI_CHAT_API_KEY"):
        return _get_azure_model()

    raise RuntimeError(
        "No model configured. Set GEMINI_API_KEY or AZURE_OPENAI_CHAT_API_KEY in .env"
    )
