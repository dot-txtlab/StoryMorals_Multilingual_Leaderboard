"""Load and validate the YAML configs (models, languages) and data paths."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from . import CONFIG, DATA


@dataclass
class ModelSpec:
    id: str
    provider: str
    display: str
    enabled: bool = True
    max_tokens: int | None = None
    temperature: float | None = None

    def column(self) -> str:
        """Stable identifier used as a `source` value in long-format tables."""
        return self.id


def load_models(path=None) -> tuple[list[ModelSpec], dict, dict]:
    """Return (enabled model specs, providers dict, defaults dict)."""
    path = path or (CONFIG / "models.yaml")
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {})
    providers = cfg["providers"]
    specs: list[ModelSpec] = []
    for m in cfg["models"]:
        if not m.get("enabled", True):
            continue
        if m["provider"] not in providers:
            raise ValueError(f"Model {m['id']} uses unknown provider {m['provider']!r}")
        specs.append(
            ModelSpec(
                id=m["id"],
                provider=m["provider"],
                display=m.get("display", m["id"]),
                enabled=True,
                max_tokens=m.get("max_tokens", defaults.get("max_tokens", 1024)),
                temperature=m.get("temperature", defaults.get("temperature")),
            )
        )
    return specs, providers, defaults


def load_languages(path=None) -> dict[str, dict]:
    """language code -> {'name':..., 'country':...}."""
    path = path or (CONFIG / "languages.yaml")
    return yaml.safe_load(path.read_text())


def load_prompt_template(path=None) -> str:
    path = path or (DATA / "prompt_template.txt")
    return path.read_text()


# Canonical input data paths (user may update passages.csv; human_morals.csv is fixed).
HUMAN_MORALS_CSV = DATA / "human_morals.csv"
PASSAGES_CSV = DATA / "passages.csv"
