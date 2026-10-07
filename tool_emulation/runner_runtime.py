"""Minimal sandbox adapter used only by the tool-emulation runner."""

import io
import shutil
import tarfile
from pathlib import Path

from components.ToolEmulation.Emulator import Emulator
from environment_handler.runtime import create_sandbox_session, ensure_emulation_dependencies


def _export_sandbox(
    session,
    root_directory_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Copy a sandbox directory from Docker without allowing path traversal."""
    output_dir = Path(output_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_bytes, stat = session.get_archive(str(root_directory_path))
    if stat.get("size", 0) == 0:
        return output_dir

    resolved_output = output_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
        for member in archive.getmembers():
            target = (output_dir / Path(member.name.lstrip("/"))).resolve()
            if not target.is_relative_to(resolved_output):
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                source = archive.extractfile(member)
                if source is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read())
    return output_dir


def run_sandbox_simulation(
    node_spec,
    handler,
    sandbox_output_path: str | Path | None = None,
):
    """Emulate tools, commit the sandbox image, and optionally export /sandbox.

    The former structural and behavioral validators were pass-through checks, so the
    runner calls the retained emulator directly.
    """
    with create_sandbox_session(handler=handler, verbose=True) as session:
        ensure_emulation_dependencies(session)
        emulator = Emulator(
            handler=handler,
            emulation_stratgy_name="llm",
            default_config=False,
        )
        node_spec = emulator.create_emulations(node_spec, session=session)

        repository = handler.image_tag.split(":", 1)[0]
        docker_api = session.container.client.api
        original_timeout = docker_api.timeout
        docker_api.timeout = 900
        try:
            session.container.commit(repository=repository, tag="latest")
        finally:
            try:
                if sandbox_output_path is not None:
                    _export_sandbox(
                        session=session,
                        root_directory_path=handler.sandbox_root_directory_path or "/sandbox",
                        output_path=sandbox_output_path or Path("sandbox_output"),
                    )
            finally:
                docker_api.timeout = original_timeout
    return node_spec
