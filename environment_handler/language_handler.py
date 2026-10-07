import subprocess
import re
from pathlib import Path
from .utils import prepare_codebase
from node_spec.structure_schema import NodeSpec

class LanguageHandler:
    def __init__(
        self,
        user_id: str,
        system_name: str,
        prog_language: str,
        prog_language_version: str,
        path_to_zipfile: str,
        path_to_envfile: str,
        execution_command: dict = {},
    ):
        # User metadata
        """
        Description:
            Initialize the class instance and assign initial state.
        Args:
            self (Any): Instance used to access class state and methods.
            user_id (str): User identifier used to scope the operation.
            system_name (str): System name associated with the operation.
            prog_language (str): prog_language parameter used by __init__.
            prog_language_version (str): prog_language_version parameter used by __init__.
            path_to_zipfile (str): Filesystem path used for reading, writing, or discovery.
            path_to_envfile (str): Filesystem path used for reading, writing, or discovery.
            command_template (str): Command string executed by the function.
        Returns:
            None: No value is returned.
        """
        self.user_id: str = user_id
        self.system_name: str = system_name
        # Language metadata
        self.prog_language: str = prog_language
        self.prog_language_version: str = prog_language_version

        # Required inputs
        self.path_to_zipfile: Path = Path(path_to_zipfile)
        self.path_to_envfile: Path = Path(path_to_envfile)
        self.execution_command: dict = execution_command

        # Codebase and file paths
        self.path_to_codebase: Path = None
        self.path_to_dockerfile: Path = None
        self.path_to_requirements: Path = None

        # Sandbox and image info
        self.sandbox_root_directory_path: Path = None
        self.image_tag: str = None

        # Extraction configuration
        self.extraction_root: Path = Path(
            f"runtime_artifacts/codebases/{self.system_name}"
        )

        # Lifecycle flags
        self.prepared: bool = False

        # validate required inputs
        self._validate_inputs_exist()


    def _validate_inputs_exist(self) -> None:
        """
        Description:
            Validates if required inputs exist and are properly formatted, raising exceptions if not.
        Args:
            self (Any): Instance used to access class state and methods.
        Returns:
            None: No value is returned.
        """
        if not str(self.path_to_zipfile).strip():
            raise ValueError("path_to_zipfile must not be empty")

        if not self.path_to_zipfile.exists() or not self.path_to_zipfile.is_file():
            raise FileNotFoundError(f"Zip file not found: {self.path_to_zipfile}")


        if not str(self.path_to_envfile).strip():
            raise ValueError("path_to_envfile must not be empty")

    def prepare(self) -> Path:
        """
        Perform I/O-heavy preparation steps (codebase extraction).
        """
        self.path_to_codebase = prepare_codebase(
            source=str(self.path_to_zipfile),
            source_type="zip",
            destination=str(self.extraction_root),
        )
        self.prepared = True
        return self.path_to_codebase

    def generate_dockerfile(self) -> Path:
        """
        Generate a Dockerfile for this language.
        """

        raise NotImplementedError(
            "This method should be implemented in subclasses to generate a Dockerfile."
        )

    def common_dockerignore_patterns(self) -> list[str]:
        """
        Shared .dockerignore patterns applicable to most languages.
        """
        return [
            "# VCS",
            ".git",
            ".gitignore",
            "",
            "# Environments and secrets",
            ".env",
            ".env.*",
            "",
            "# Build outputs",
            "dist/",
            "build/",
            "",
            "# Editors",
            ".vscode",
            ".idea",
            ".DS_Store",
            "",
        ]

    def _rewrite_mlflow_tracking_uri_for_sandbox(self) -> int:
        """
        Rewrites mlflow.set_tracking_uri(...) calls in Python files so sandbox runs
        always point to the shared mlruns location.
        """
        if self.path_to_codebase is None:
            return 0

        call_pattern = re.compile(
            r"mlflow\.set_tracking_uri\(\s*.*?\s*\)", re.DOTALL
        )

        set_tracking_str = (
            'mlflow.set_tracking_uri(__import__("os").getenv('
            '"MLFLOW_TRACKING_URI", "file:/sandbox/mlruns"))'
        )
        import_str = 'import mlflow'
        import_pattern = re.compile(
        r"(^|\n)\s*(import\s+mlflow\b)", re.MULTILINE
        )

        first_readable_py = None
        first_mlflow_import = None
        updates = 0
        for py_file in self.path_to_codebase.rglob("*.py"):
            try:
                original = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if first_readable_py is None:
                first_readable_py = py_file
            if first_mlflow_import is None and import_pattern.search(original):
                first_mlflow_import = py_file
            rewritten, count = call_pattern.subn(set_tracking_str, original)
            if count > 0 and rewritten != original:
                py_file.write_text(rewritten, encoding="utf-8")
                updates += count

        # added fallback in case the repo uses a different way of tracking the traces
        if updates == 0:
            if first_mlflow_import:
                original = first_mlflow_import.read_text(encoding="utf-8")
                rewritten = import_pattern.sub(
                    lambda m: f"{m.group(0)}\n{set_tracking_str}",
                    original,
                    count=1,
                )
                if rewritten != original:
                    first_mlflow_import.write_text(rewritten, encoding="utf-8")
                    updates = -1 # -1 to indicate forced addition
            elif first_readable_py:
                original = first_readable_py.read_text(encoding="utf-8")
                rewritten = f"{import_str}\n{set_tracking_str}\n{original}"
                first_readable_py.write_text(rewritten, encoding="utf-8")
                updates = -1

        return updates

    def build_image(self) -> Path | None:
        """Build a Docker image for the codebase using the generated Dockerfile.
        Sets the image_tag attribute upon successful build.
        """
        # self._rewrite_mlflow_tracking_uri_for_sandbox()

        self.image_tag = f"{self.user_id}_{self.system_name}:v1"

        cmd = [
            "docker",
            "build",
            "-t",
            self.image_tag,
            "-f",
            self.path_to_dockerfile.name,
            ".",
        ]

        result = subprocess.run(
            cmd,
            cwd=self.path_to_codebase,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Docker build failed.",
                f"STDOUT: {result.stdout}",
                f"STDERR: {result.stderr}",
            )

        self.sandbox_root_directory_path = Path(f"/sandbox")

    def remove_image(self, force: bool = False) -> bool:
        """
        Remove the Docker image associated with this language handler.
        """

        if self.image_tag is None:
            return False  # Image was never built.

        cmd = ["docker", "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(self.image_tag)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.lower()

            # Image does not exist → treat as success
            if "no such image" in stderr:
                return False

            raise RuntimeError(
                "Docker image removal failed.",
                f"STDOUT: {result.stdout}",
                f"STDERR: {result.stderr}",
            )

        self.image_tag = None  # Image successfully removed
        return True

    def build_cli_command_template(self) -> str:
        raise NotImplementedError(
            "This method should be implemented in subclasses to build the CLI command template based on the node spec."
        )

    def remove_envfile(self) -> bool:
        """
        Remove the environment file associated with this language handler.
        """

        if self.path_to_envfile is None or not self.path_to_envfile.exists():
            return False  # Env file does not exist.
        self.path_to_envfile.unlink(missing_ok=True)
        return True