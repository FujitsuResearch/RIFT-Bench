from .docker_file_templates import python_dockerfile

# Minimal registry to support multiple languages in the future.
LANGUAGE_DEFS = {
    "python": {
        "default_version": "3.11",
        "dockerfile_fn": python_dockerfile,
        "install_system_deps": True,
        "dockerignore_patterns": [
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".venv",
            "*.egg-info",
        ],
    },
}
