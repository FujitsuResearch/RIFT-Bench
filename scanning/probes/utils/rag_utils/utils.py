import json
from pathlib import Path
from typing import Any
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping, Sequence

def load_json(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{file_path} must contain a JSON array of objects")
    return data


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_json(data: Any, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_jsonl(records: list[dict[str, Any]], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_jsonable(x):
    if x is None or isinstance(x, (int, float, bool, str)):
        return x

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")

    if isinstance(x, Mapping):
        return {str(k): to_jsonable(v) for k, v in x.items()}

    if hasattr(x, "model_dump"):
        return to_jsonable(x.model_dump())

    if hasattr(x, "dict"):
        return to_jsonable(x.dict())

    if is_dataclass(x):
        return to_jsonable(asdict(x))

    if isinstance(x, (set, frozenset)):
        return [to_jsonable(i) for i in x]

    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return [to_jsonable(i) for i in x]

    if hasattr(x, "__dict__"):
        return {
            k: to_jsonable(v)
            for k, v in vars(x).items()
            if not k.startswith("_")
        }

    return str(x)