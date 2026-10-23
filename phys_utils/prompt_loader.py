"""Runtime template configuration."""
from functools import lru_cache
import json
import os
from pathlib import Path


@lru_cache(maxsize=8)
def _read_prompts(path: str) -> dict[str, str]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in prompts.items()
    ):
        raise ValueError("Configuration must be a JSON object mapping names to strings.")
    return prompts


def load_prompt(name: str) -> str:
    """Read a named template from the configured JSON object."""
    custom = os.environ.get("PHYSREASONING_PROMPTS_FILE")
    if not custom:
        raise RuntimeError("Configure PHYSREASONING_PROMPTS_FILE before launching Python.")
    source = Path(custom).expanduser().resolve()
    if source.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Configuration file location is not allowed.")
    try:
        templates = _read_prompts(str(source))
    except (OSError, json.JSONDecodeError):
        raise ValueError("Cannot read configuration JSON; check the path and format.") from None
    if name not in templates:
        raise KeyError(f"Configuration key {name!r} is required.")
    template = templates[name]
    if not template.strip():
        raise ValueError(f"Configuration value {name!r} must not be empty.")
    return template
