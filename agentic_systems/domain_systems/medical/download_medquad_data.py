import json
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MEDQUAD_ZIP_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
MEDICAL_DIR = Path(__file__).resolve().parent
OUT_PATH = MEDICAL_DIR / "langraph_agent" / "data" / "medquad_documents.json"


def _guess_qtype(question: str) -> str:
    q = question.lower()
    if "symptom" in q:
        return "symptoms"
    if "treat" in q or "therapy" in q:
        return "treatment"
    if "prevent" in q:
        return "prevention"
    if "diagnos" in q:
        return "diagnosis"
    if "test" in q or "screen" in q:
        return "tests"
    if "cause" in q:
        return "causes"
    return "general"


def _parse_medquad_xmls(root_dir: Path) -> list[dict]:
    rows: list[dict] = []
    idx = 1
    for xml_path in sorted(root_dir.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except ET.ParseError:
            continue
        topic = xml_path.stem
        for qa in xml_root.findall(".//QAPair"):
            q = (qa.findtext("Question") or "").strip()
            a = (qa.findtext("Answer") or "").strip()
            if not q or not a:
                continue
            rows.append(
                {
                    "id": f"medquad_{idx:06d}",
                    "source": "MedQuAD",
                    "domain": "medical",
                    "question": q,
                    "answer": a,
                    "text": f"Question: {q}\nAnswer: {a}",
                    "metadata": {"topic": topic.lower(), "question_type": _guess_qtype(q)},
                }
            )
            idx += 1
    return rows


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "medquad.zip"
        urlretrieve(MEDQUAD_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        docs = _parse_medquad_xmls(Path(tmp) / "MedQuAD-master")
    if not docs:
        raise RuntimeError("No MedQuAD records parsed.")
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(docs)} records to {OUT_PATH}")


if __name__ == "__main__":
    main()
