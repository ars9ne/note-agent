import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).with_name(".env")


class Settings:
    def __init__(
        self,
        openrouter_api_key,
        openrouter_model,
        openrouter_classifier_model,
        openrouter_base_url,
    ):
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.openrouter_classifier_model = openrouter_classifier_model
        self.openrouter_base_url = openrouter_base_url


def get_settings():
    load_dotenv(ENV_PATH, override=True)

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Create .env from .env.example."
        )

    model = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.7-plus").strip()
    classifier_model = os.getenv(
        "OPENROUTER_CLASSIFIER_MODEL",
        "google/gemma-4-31b-it",
    ).strip()
    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ).strip()

    return Settings(api_key, model, classifier_model, base_url)
