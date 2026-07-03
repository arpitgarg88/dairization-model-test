"""Load environment variables from .env if present."""

from __future__ import annotations

from pathlib import Path


def load_dotenv() -> None:
    """Load `.env` from project root (no-op if missing or python-dotenv not installed)."""
    try:
        from dotenv import load_dotenv as _load_dotenv
    except ImportError:
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        _load_dotenv(env_path)
