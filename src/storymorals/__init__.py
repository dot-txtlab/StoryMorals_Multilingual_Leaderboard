"""StoryMorals multilingual leaderboard — replication of Wu & Piper (Figs 3 & 4)."""
from pathlib import Path

# Repo root = three levels up from this file (src/storymorals/__init__.py).
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
DOCS = ROOT / "docs"

__all__ = ["ROOT", "CONFIG", "DATA", "OUTPUT", "DOCS"]
