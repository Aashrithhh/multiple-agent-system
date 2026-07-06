"""Central model configuration for the coding-agent lessons."""

import os
import ssl
from pathlib import Path

import certifi
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv()


def _required(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Copy .env.example to .env and configure it."
        )
    return value


def get_chat_model() -> ChatOpenAI:
    """Create the Azure-hosted OpenAI chat model used by the project."""
    endpoint = _required("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = _required("AZURE_OPENAI_CHAT_API_KEY")
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    # Corporate antivirus/proxy products may publish their trusted root here.
    # Add it to the normal CA bundle without weakening TLS verification.
    extra_ca = os.getenv("NODE_EXTRA_CA_CERTS")
    if extra_ca and Path(extra_ca).is_file():
        ssl_context.load_verify_locations(cafile=extra_ca)
        # Some locally installed proxy roots predate OpenSSL's strict RFC 5280
        # checks. Keep certificate verification enabled while allowing that root.
        ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT

    return ChatOpenAI(
        api_key=SecretStr(api_key),
        base_url=f"{endpoint}/openai/v1/",
        default_headers={"api-key": api_key},
        model=_required("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0")),
        max_completion_tokens=int(os.getenv("CHAT_MAX_TOKENS", "6000")),
        http_client=httpx.Client(verify=ssl_context),
        http_async_client=httpx.AsyncClient(verify=ssl_context),
    )
