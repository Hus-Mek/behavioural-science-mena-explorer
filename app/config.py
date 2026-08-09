"""Config file access and API-key resolution (config.json first, env second)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
CONFIG_FILE = ROOT / "config.json"


def load_config():
    if CONFIG_FILE.exists():
        try:
            return json.load(open(CONFIG_FILE))
        except Exception:
            pass
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

def get_api_key():
    cfg = load_config()
    return cfg.get("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", ""))

def get_gemini_api_key():
    cfg = load_config()
    return cfg.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", ""))
