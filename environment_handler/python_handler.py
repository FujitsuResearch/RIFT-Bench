from pathlib import Path
from .language_handler import LanguageHandler
from .language_registry import LANGUAGE_DEFS
from .docker_file_templates import python_dockerfile
from node_spec.structure_schema import NodeSpec

class PythonHandler(LanguageHandler):
    def __init__(
        self,
        user_id: str,
        system_name: str,
        zipfile_path: str,
        envfile_path: str,
        prog_language: str = "python",
        prog_language_version: str | None = None,
        execution_command: dict = {},
        auto_prepare: bool = True,
    ):
        # Default to registry version if not provided
        """
        Initialize the class instance and assign initial state.
        Args:
            self (Any): Instance used to access class state and methods.
            user_id (str): User identifier used to scope the operation.
            system_name (str): System name associated with the operation.
            zipfile_path (str): Filesystem path used for reading, writing, or discovery.
            envfile_path (str): Filesystem path used for reading, writing, or discovery.
            prog_language (str): prog_language parameter used by __init__.
            prog_language_version (str | None): prog_language_version parameter used by __init__.
            command_template (str): Command string executed by the function.
            auto_prepare (bool): auto_prepare parameter used by __init__.
        Returns:
            None: No value is returned.
        """
        if prog_language_version is None:
            prog_language_version = LANGUAGE_DEFS[prog_language]["default_version"]

        super().__init__(
            user_id=user_id,
            system_name=system_name,
            prog_language=prog_language,
            prog_language_version=prog_language_version,
            path_to_zipfile=zipfile_path,
            path_to_envfile=envfile_path,
            execution_command=execution_command,
        )

        self.path_to_requirements: Path | None = None
        self.codebase_dockerfile_path: Path | None = None
        self.image_tag: str | None = None

        if auto_prepare:
            self.prepare()

    def generate_dockerfile(self) -> Path:
        """
        Generate a Dockerfile for this Python environment.

        Returns:
            Path: The path to the generated Dockerfile.
        """

        if self.path_to_codebase is None:
            raise ValueError(
                "Codebase path is not set. Please run prepare() before generating the Dockerfile."
            )

        # Render Dockerfile fresh every time to avoid stale cache
        dockerfile_content = python_dockerfile(
            python_version=self.prog_language_version,
            install_system_deps=LANGUAGE_DEFS[self.prog_language][
                "install_system_deps"
            ],
        )

        prog_language_full_name = (
            f"{self.prog_language}_{self.prog_language_version.replace('.', '')}"
        )

        # Write only inside the extracted codebase
        target_dockerfile_path = (
            self.path_to_codebase / f"{prog_language_full_name}.Dockerfile"
        )
        # Ensure the parent directory exists
        target_dockerfile_path.parent.mkdir(parents=True, exist_ok=True)

        # Create or overwrite the Dockerfile
        with open(target_dockerfile_path, "w") as dockerfile:
            dockerfile.write(dockerfile_content)

        self.path_to_dockerfile = target_dockerfile_path

        # Write a .dockerignore file so that Docker builds do not include the .env file.
        self._write_dockerignore()

        return target_dockerfile_path

    def prepare(self) -> Path:
        """
        Prepare the codebase and set common paths for downstream steps. The paths include the .env file, requirements.txt and codebase directory as attributes of the handler.

        Returns:
            Path: The path to the prepared codebase directory.
        """
        path = super().prepare()
        self.path_to_requirements = path / "requirements.txt"
        return path

    def _write_dockerignore(self) -> None:
        """
        Creates a .dockerignore file in the codebase directory to exclude unnecessary files from the Docker build context.
        """
        dockerignore_path = self.path_to_codebase / ".dockerignore"

        common_patterns = self.common_dockerignore_patterns()

        language_patterns = LANGUAGE_DEFS[self.prog_language].get(
            "dockerignore_patterns", []
        )

        # Preserve order, de-dupe language-specific entries
        merged_patterns: list[str] = []
        for entry in common_patterns + language_patterns:
            if entry not in merged_patterns:
                merged_patterns.append(entry)

        dockerignore_content = "\n".join(merged_patterns)

        with open(dockerignore_path, "w") as fh:
            fh.write(dockerignore_content)

    def build_cli_command_template(self) -> str:
        """
        Builds a CLI command template for executing tasks in the sandbox environment.
        The template includes placeholders for the script path, task arguments, and experiment name.
        Args:
            node_spec (NodeSpec): the node spec object that respresents the system.
            sandbox_path (str): The path to the sandbox directory.
        Returns:
            str: A CLI command template string with placeholders for task execution.
        """

        # runner = self.execution_command.get("runner", "python")
        runner = "python" #TODO: When we support more languages, we'll need to use the runner from the execution command provided by the user.
        entrypoint = self.execution_command.get("entrypoint", "main.py")
        args = self.execution_command.get("args", [])
        task_arg_names = self.execution_command.get("task_arg_names", [])

        template_args = []

        for arg in args:
            name = arg.get('name')
            if not name:
                continue

            if name in task_arg_names:
                value = "{query}"
            else:
                value = arg.get("value", "")

            if name in task_arg_names:
                task_arg_names.append(name)

            template_args.append(
                {
                    "name": name,
                    "value": value,
                }
            )

        # For example, a template might look like: "python main.py --task {query} --exp_name my_exp". 
        command_template = f"{runner} {entrypoint} " + " ".join(
            [f"{arg['name']} {arg['value']}" for arg in template_args]
        )
        return command_template

