"""Build and run isolated Python environments for emulation and scanning."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import shutil

from llm_sandbox import SandboxSession
from llm_sandbox.core.session_base import BaseSession

from .language_handler import LanguageHandler
from .python_handler import PythonHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_SOCKET_PATH = "/var/run/docker.sock"


def copy_node_spec_schema_folder_to_codebase(handler: LanguageHandler) -> None:
    source = PROJECT_ROOT / "node_spec"
    destination = handler.path_to_codebase / "node_spec"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def prepare_environment(
    user_id: str,
    system_name: str,
    envfile_path: str,
    zipfile_path: str,
    execution_command: dict[str, Any],
    prog_lang: str = "python",
    prog_lang_version: str = "3.11",
) -> LanguageHandler:
    """Build a runnable sandbox image from a source ZIP."""
    if prog_lang != "python":
        raise NotImplementedError("Currently only Python is supported.")

    handler = PythonHandler(
        user_id=user_id,
        system_name=system_name,
        zipfile_path=zipfile_path,
        envfile_path=envfile_path,
        prog_language_version=prog_lang_version,
        execution_command=execution_command,
    )
    copy_node_spec_schema_folder_to_codebase(handler)
    handler.generate_dockerfile()
    handler.build_image()
    shutil.rmtree(handler.path_to_codebase.parent)
    return handler


def _docker_socket_runtime_configs(mount_docker_socket: bool) -> dict[str, Any]:
    if not mount_docker_socket or not Path(DOCKER_SOCKET_PATH).exists():
        return {}
    return {
        "volumes": {
            DOCKER_SOCKET_PATH: {"bind": DOCKER_SOCKET_PATH, "mode": "rw"}
        }
    }


@contextmanager
def create_sandbox_session(
    handler: LanguageHandler,
    verbose: bool = False,
    mount_docker_socket: bool = False,
) -> Generator[SandboxSession, None, None]:
    """Open a session backed by the handler's sandbox image."""
    with SandboxSession(
        verbose=verbose,
        lang=handler.prog_language,
        keep_template=False,
        image=handler.image_tag,
        backend="docker",
        runtime_configs=_docker_socket_runtime_configs(mount_docker_socket),
    ) as session:
        yield session


def run_command_sb(command: str, session: BaseSession, timeout_sec: int | None = None):
    timeout_prefix = f"timeout {int(timeout_sec)}s " if timeout_sec and timeout_sec > 0 else ""
    return session.execute_command([
        "bash",
        "-lc",
        (
            "if [ -f /sandbox/.env ]; then set -a; . /sandbox/.env; set +a; fi; "
            "export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}; "
            "export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING:-false}; "
            + timeout_prefix + command
        ),
    ])


def _ensure_python_packages_installed(
    session: BaseSession,
    packages: list[str],
    step_name: str,
) -> None:
    if not packages:
        return
    package_args = " ".join(packages)
    command = (
        "if command -v python >/dev/null 2>&1; then SB_PY=python; "
        "elif command -v python3 >/dev/null 2>&1; then SB_PY=python3; "
        "else echo 'python executable not found in sandbox' >&2; exit 1; fi; "
        f"for SB_PKG in {package_args}; do "
        '$SB_PY -m pip show "$SB_PKG" >/dev/null 2>&1 || '
        '$SB_PY -m pip install --cache-dir /sandbox/.sandbox-pip-cache "$SB_PKG"; '
        "done"
    )
    result = run_command_sb(command, session=session)
    returncode = getattr(result, "exit_code", getattr(result, "returncode", 0))
    if returncode:
        raise RuntimeError(f"{step_name} failed: {command}")


def ensure_emulation_dependencies(session: BaseSession) -> None:
    _ensure_python_packages_installed(
        session=session,
        packages=["langchain_openai", "fastmcp"],
        step_name="Tool emulation dependency installation",
    )


def ensure_defense_dependencies(session: BaseSession) -> None:
    _ensure_python_packages_installed(
        session=session,
        packages=["torch", "transformers"],
        step_name="Defense dependency installation",
    )
