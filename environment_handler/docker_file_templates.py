def python_dockerfile(
    python_version: str,
    codebase_path: str = ".",
    install_system_deps: bool = True,
) -> str:
    """
    Provides a Python Dockerfile template that is used to create a Docker image for analyzing Python codebases.
    Args:
        python_version (str): python_version parameter used by python_dockerfile.
        codebase_path (str): Filesystem path used for reading, writing, or discovery.
        install_system_deps (bool): install_system_deps parameter used by python_dockerfile.
    Returns:
        str: String result produced by the function.
    """
    base_image = f"python:{python_version}-bookworm"

    docker_cli_stage = ""
    system_deps = ""
    if install_system_deps:
        docker_cli_stage = "FROM docker:27-cli AS docker-cli\n"
        system_deps = """
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker-cli /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins
""".rstrip()

    return f"""
{docker_cli_stage}FROM {base_image}

{system_deps}

# Copy requirements file
COPY requirements.txt /sandbox/requirements.txt

# Install dependencies from requirements.txt (correct path)
RUN pip install --upgrade pip \\
    && pip install -r /sandbox/requirements.txt

# Copy full codebase
COPY {codebase_path} /sandbox

WORKDIR /sandbox

CMD ["python"]
""".strip()
