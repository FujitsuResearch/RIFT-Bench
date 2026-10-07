import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

STATE_DIR = Path(__file__).resolve().parent / "tmp_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)


def _stable_seed(key: str) -> int:
    base_seed = os.getenv("TRAVEL_SIM_SEED", "20260501")
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rng(key: str) -> random.Random:
    return random.Random(_stable_seed(key))


def _state_path(domain: str, record_type: str, key: str) -> Path:
    safe_key = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return STATE_DIR / f"{domain}_{record_type}_{safe_key}.json"


def state_file_name(domain: str, record_type: str, key: str) -> str:
    return _state_path(domain, record_type, key).name


def write_state(domain: str, record_type: str, key: str, payload: Dict[str, Any]) -> Path:
    out = {
        "domain": domain,
        "record_type": record_type,
        "key": key,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "payload": payload,
    }
    path = _state_path(domain, record_type, key)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def read_state(domain: str, record_type: str, key: str) -> Dict[str, Any] | None:
    path = _state_path(domain, record_type, key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def serper_search(query: str, num_results: int = 5) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "SERPER_API_KEY is missing. Set it in the environment."
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max(1, min(10, num_results))}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Serper request failed: {exc}"
    items: List[str] = []
    for idx, item in enumerate(data.get("organic", [])[: num_results], start=1):
        title = item.get("title", "No title")
        link = item.get("link", "No link")
        snippet = item.get("snippet", "")
        items.append(f"{idx}. {title}\n{link}\n{snippet}")
    if not items:
        return "No search results found."
    return "\n\n".join(items)


def choose_outcome(domain: str, entity_key: str, allowed: List[str], weights: List[float]) -> str:
    rng = _rng(f"{domain}:{entity_key}")
    return rng.choices(allowed, weights=weights, k=1)[0]


def estimate_price(domain: str, key: str) -> float:
    rng = _rng(f"{domain}:price:{key}")
    if domain == "hotel":
        return round(rng.uniform(120, 420), 2)
    if domain == "flight":
        return round(rng.uniform(180, 980), 2)
    if domain == "restaurant":
        return round(rng.uniform(25, 140), 2)
    return round(rng.uniform(20, 160), 2)


def booking_reference(domain: str, key: str) -> str:
    suffix = hashlib.sha1(f"{domain}:{key}".encode("utf-8")).hexdigest()[:8].upper()
    return f"{domain[:3].upper()}-{suffix}"


def format_plain_text(title: str, lines: List[str]) -> str:
    return f"{title}\n" + "\n".join(lines)


def state_key(parts: Tuple[Any, ...]) -> str:
    normalized = "|".join(str(p).strip().lower() for p in parts)
    return normalized
