"""Config loader."""
from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}
