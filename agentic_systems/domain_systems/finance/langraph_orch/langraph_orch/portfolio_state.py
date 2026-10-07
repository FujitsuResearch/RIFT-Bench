import hashlib
import json
import os
import random
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent / "tmp_state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
TICKER_UNIVERSE = ["AAPL", "GOOGL", "NVDA", "TSLA"]
PORTFOLIO_RUN_KEY_ENV = "FINANCE_PORTFOLIO_RUN_KEY"


def _seeded_rng(seed_input: str) -> random.Random:
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def format_plain_text(title: str, lines: list[str]) -> str:
    return "\n".join([title, *lines])


def deterministic_price(ticker: str) -> float:
    rng = _seeded_rng(f"price:{ticker.upper()}")
    return round(rng.uniform(12.0, 480.0), 2)


def _default_portfolio() -> dict:
    rng = _seeded_rng("finance_portfolio_state")
    picks = rng.sample(TICKER_UNIVERSE, 4)
    holdings = {}
    for t in picks:
        holdings[t] = {
            "quantity": rng.randint(3, 40),
            "average_buy_price": round(deterministic_price(t) * rng.uniform(0.8, 1.2), 2),
        }
    return {"cash_balance": round(rng.uniform(2500, 25000), 2), "holdings": holdings, "order_history": []}


def _active_run_key() -> str:
    raw = os.getenv(PORTFOLIO_RUN_KEY_ENV, "default")
    return "".join(ch for ch in raw.lower().strip() if ch.isalnum() or ch in ("_", "-")) or "default"


def portfolio_file_path() -> Path:
    return STATE_DIR / f"portfolio_state_{_active_run_key()}.json"


def initialize_portfolio_state() -> None:
    path = portfolio_file_path()
    if not path.exists():
        path.write_text(json.dumps(_default_portfolio(), indent=2), encoding="utf-8")


def read_portfolio() -> dict:
    initialize_portfolio_state()
    return json.loads(portfolio_file_path().read_text(encoding="utf-8"))


def write_portfolio(state: dict) -> None:
    portfolio_file_path().write_text(json.dumps(state, indent=2), encoding="utf-8")


def normalize_ticker(ticker: str) -> str:
    return (ticker or "").upper().strip()


def is_allowed_ticker(ticker: str) -> bool:
    return normalize_ticker(ticker) in TICKER_UNIVERSE
