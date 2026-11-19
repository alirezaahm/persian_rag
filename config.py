"""
configuration management for the Persian RAG application.
loads local and cloud API settings from environment variables.
API credentials are never written to the vector index.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


# load variables from the .env file in the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")

load_dotenv(ENV_FILE)

@dataclass
class APIConfig:
    #configuration for one openai compatible api service
    api_key: str
    base_url: str
    model: str
    timeout: int = 60


@dataclass
class AppConfig:
    #configuration for the complete RAG application."""

    # llm settings
    llm_provider: str
    llm_api: APIConfig
    # embedding settings
    embedding_provider: str
    embedding_api: APIConfig
    # retrieval settings
    top_k: int = 5
    # index settings
    index_dir: str = "data/persian_rag_index"


def load_config():
    """
    read configuration from environment variables.
    the llm and embedding API settings are independent.
    """

    llm_api = APIConfig(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        model=os.getenv("LLM_MODEL", ""),
        timeout=int(os.getenv("LLM_TIMEOUT", "120"))
    )

    embedding_api = APIConfig(
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        model=os.getenv("EMBEDDING_MODEL", ""),
        timeout=int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    )

    config = AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "local"),
        llm_api=llm_api,

        embedding_provider=os.getenv("EMBEDDING_PROVIDER","local"),
        embedding_api=embedding_api,

        top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        index_dir=os.getenv("INDEX_DIR","data/persian_rag_index")
    )
    return config

def validate_cloud_config(provider, api_config, service_name):
    """
    check whether the required settings exist for a cloud API.

    this checks configuration only. It does not test the connection.
    """

    if provider.lower() != "cloud":
        return

    missing = []

    if not api_config.api_key:
        missing.append("API key")

    if not api_config.base_url:
        missing.append("base URL")

    if not api_config.model:
        missing.append("model name")

    if missing:
        raise ValueError(
            f"{service_name} cloud configuration is incomplete. "
            f"Missing: {', '.join(missing)}")