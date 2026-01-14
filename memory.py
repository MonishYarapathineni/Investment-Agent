# memory.py 
import json, os
from pathlib import Path

MEMO_DIR = Path("data/memory")
MEMO_DIR.mkdir(parents=True, exist_ok=True)

def load_notes(symbol: str):
    p = MEMO_DIR / f"{symbol.upper()}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_notes(symbol: str, notes: dict):
    p = MEMO_DIR / f"{symbol.upper()}.json"
    p.write_text(json.dumps(notes, indent=2))
