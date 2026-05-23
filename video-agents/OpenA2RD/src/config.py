"""Configuration loaded from .env and CLI args."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # API keys
    gemini_api_key: str = ""
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"

    # Model backends
    ti2i_backend: str = "nano_banana"  # nano_banana | local_sdxl
    ti2v_backend: str = "veo"  # veo | local_ltx | local_wan

    # Output
    output_dir: Path = Path("./outputs")
    log_level: str = "INFO"

    # HITS performance settings
    max_refinement_iterations_frame: int = 2
    max_refinement_iterations_video: int = 2
    max_candidates_per_frame: int = 3
    max_candidates_per_video: int = 3
    early_stop_threshold: float = 9.0

    # MVMem limits
    mvmem_schema_limit: int = 100  # max window size for S and image lists as MLLM inputs

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
