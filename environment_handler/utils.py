from pathlib import Path
from pathlib import PurePosixPath
from typing import Literal
from zipfile import ZipFile


def _extract_code_from_zipfile(
    zipfile_path: str,
    extract_to: Path,
) -> Path:
    """
    Extract a ZIP file into a deterministic directory and return its path.

    Args:
        zipfile_path: Path to the ZIP file.
        extract_to: Directory where the ZIP contents will be extracted.

    Returns:
        Path to the extracted codebase directory.
    """

    zipfile_path = Path(zipfile_path)

    if not zipfile_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zipfile_path}")

    extract_to.mkdir(parents=True, exist_ok=True)

    # Deterministic target directory
    target_dir = extract_to

    target_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zipfile_path, "r") as zf:
        members = zf.infolist()
        source_files = [
            PurePosixPath(member.filename)
            for member in members
            if not member.is_dir()
            and not member.filename.startswith("__MACOSX/")
            and PurePosixPath(member.filename).name != ".DS_Store"
        ]
        root_names = {path.parts[0] for path in source_files if path.parts}
        has_root_files = any(len(path.parts) == 1 for path in source_files)
        enclosing_dir = (
            next(iter(root_names))
            if len(root_names) == 1 and not has_root_files and source_files
            else None
        )
        if enclosing_dir:
            zf.extractall(path=target_dir)
        else:
            flat_codebase_dir = target_dir / "codebase"
            flat_codebase_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(path=flat_codebase_dir)

    # ZIPs may contain a single enclosing project directory, or may place
    # source files directly at the archive root. In the latter case the
    # extraction directory itself is the codebase root.
    return target_dir / enclosing_dir if enclosing_dir else target_dir / "codebase"


def prepare_codebase(
    source: str,
    source_type: Literal["zip"],
    destination: str = "resources/codebases",
) -> Path:
    """
    Prepare a codebase from a ZIP file.

    Args:
        source: Path to zip file
        source_type: must be "zip"
        destination: Base directory to place codebases

    Returns:
        Path to prepared codebase directory
    """

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    if source_type == "zip":
        return _extract_code_from_zipfile(source, destination_path)

    raise ValueError(f"Unsupported source_type: {source_type}")
