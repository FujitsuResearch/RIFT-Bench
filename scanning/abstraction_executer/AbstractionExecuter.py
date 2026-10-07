from .ScenarioAdaptation import ScenarioAdaptator
from .util import *
from llm_sandbox import SandboxSession
from llm_sandbox.core.session_base import BaseSession
from node_spec.structure_schema import NodeSpec
from typing import Any
import time


class AbstractionExecuter:
    def __init__(
        self,
        root_directory_path: str,
        traces_path: str = "scanning/resources/traces",
        image_tag: str | None = None,
        env_file_path: str | None = None,
        repo_name: str | None = None,
        scan_timeout_sec: int | None = None,
    ) -> None:
        """
        Initilaize an AbstractionExecuter instance.

        Args:
            root_directory_path (str): The path to the root direcory of the code repository.
            traces_path (str, optional): The path to save the MLFlow traces created by the scanning scenarios. Defaults to "scanning/resources/traces".
            image_tag (str, optional): The image tag to use for the sandbox environment.
            env_file_path (str, optional): The path to the .env file to push to the sandbox. Required if image_tag is provided.
            scan_timeout_sec (int | None, optional): Default timeout budget in seconds for `execute_scan` command execution.
        """
        self.root_directory_path = root_directory_path
        self.image_tag = image_tag
        self.env_file_path = env_file_path
        self.traces_path = traces_path
        self.repo_name = repo_name
        self.scan_timeout_sec = scan_timeout_sec
        self.sa = ScenarioAdaptator(
            self.root_directory_path
        )  # Scenario Adaptator instance to apply the changes on the code repository according to the given node spec and the scanning scenario

    @staticmethod
    def _build_execution_trace_key(step_index: int, execution_cmd: str) -> str:
        normalized_cmd = str(execution_cmd or "").strip()
        return f"step_{step_index:04d}::{normalized_cmd}"

    def _build_execution_trace_entries(
        self,
        execution_cmds: list[str],
    ) -> list[dict[str, Any]]:
        trace_entries: list[dict[str, Any]] = []
        for step_index, execution_cmd in enumerate(execution_cmds or []):
            normalized_cmd = str(execution_cmd or "").strip()
            if not normalized_cmd:
                continue
            trace_entries.append(
                {
                    "step_index": step_index,
                    "execution_cmd": normalized_cmd,
                    "trace_key": self._build_execution_trace_key(
                        step_index,
                        normalized_cmd,
                    ),
                }
            )
        return trace_entries

    @staticmethod
    def _store_execution_trace_entries(
        updated_top_node_spec: NodeSpec,
        trace_entries: list[dict[str, Any]],
    ) -> None:
        metadata = dict(getattr(updated_top_node_spec, "metadata", None) or {})
        instance_info = dict(metadata.get("instance_info") or {})
        instance_info["execution_trace_entries"] = trace_entries
        instance_info["execution_trace_keys"] = [
            entry["trace_key"]
            for entry in trace_entries
            if isinstance(entry, dict) and entry.get("trace_key")
        ]
        metadata["instance_info"] = instance_info
        updated_top_node_spec.metadata = metadata

    def execute_command_and_extract_trace(
        self,
        command: str,
        session: BaseSession,
        timeout_sec: int | None = None,
    ) -> str | None:
        """
        Execute a command inside the sandbox and export the created MLFlow trace.

        Args:
            command (str): The command to execute inside the sandbox.
        Returns:
            str | None: The path to the exported trace file or None if no trace was exported.
        """
        mlruns_root = "/sandbox/mlruns"
        result = run_command_sb(command, session, timeout_sec=timeout_sec)
        return_code = getattr(result, "exit_code", getattr(result, "returncode", 0))
        try:
            return_code = int(return_code)
        except (TypeError, ValueError):
            return_code = 0
        if isinstance(timeout_sec, int) and timeout_sec > 0 and return_code in (124, 137):
            raise TimeoutError(
                f"Sandbox command timed out after {timeout_sec} seconds: {command}"
            )

        if self.traces_path:
            trace_save_path = export_single_traces_json_from_sandbox_v2(
                session,
                host_dest_path=self.traces_path,
                mlruns_root=mlruns_root,
            )
            return trace_save_path
        return None

    def execute_command_without_trace(
        self,
        updated_top_node_spec: NodeSpec,
        command: str,
        affected_nodes: (
            str
            | dict[str, list[str]]
            | list[str]
            | list[dict[str, list[str]]]
            | list[str | dict[str, list[str]]]
        ) = "all",
        time_out: int | None = None,
    ) -> Any | None:
        """
        Execute one command in sandbox after applying NodeSpec changes, without
        exporting MLflow traces.

        Steps:
        1. Open a sandbox session with the configured image.
        2. Apply code changes from ``updated_top_node_spec``.
        3. Copy ``.env`` into sandbox.
        4. Execute command and return the sandbox result.
        """
        if self.image_tag:
            with SandboxSession(
                verbose=True,
                lang="python",
                keep_template=False,
                image=self.image_tag,
            ) as session:
                self.sa.apply_changes(updated_top_node_spec, session, affected_nodes)
                if self.env_file_path:
                    copy_env_to_sandbox(session, self.env_file_path)
                return run_command_sb(command, session, timeout_sec=time_out)

        # Local fallback keeps behavior consistent with execute_scan.
        self.sa.apply_changes(updated_top_node_spec, affected_nodes=affected_nodes)
        return None

    def execute_scan(
        self,
        updated_top_node_spec: NodeSpec,
        execution_cmds: list[str],
        affected_nodes: (
            str
            | dict[str, list[str]]
            | list[str]
            | list[dict[str, list[str]]]
            | list[str | dict[str, list[str]]]
        ) = "all",
        time_out: int | None = None,
    ) -> dict[str, str] | None:
        """
        Excecute a single scanning senrio, which include:
        1. Opening sandbox session with the provided image tag.
        2. Apply the changes of the given node spec on the code repository, according to the affected nodes.
        3. Push the .env file to the the sandbox.
        4. Run MLFlow ui in the background.
        5. Excecute the given scanning commands.
        6. Exporting the created MLFlow traces to self.traces_path

        Args:
            updated_top_node_spec (NodeSpec): The updated (with scanning scenrio) top level NodeSpec object.
            execution_cmds (list[str]): List of command strings to execute inside the sandbox after applying the changes.
            affected_nodes: Either `"all"`, a dict mapping `{node_id: [kind, ...]}`,
                a legacy list of node ids, or a list containing `{node_id: [kind, ...]}` mappings.
            time_out (int | None): Maximum total number of seconds allowed for this scan run. Command execution is interrupted when the timeout budget is exhausted.

        Return:
            dict[str, str] | None: Mapping of executed commands to exported trace file paths, or `None` when no sandbox image is used.
        """

        timeout_sec = self.scan_timeout_sec if time_out is None else time_out
        scan_deadline: float | None = None
        if isinstance(timeout_sec, int) and timeout_sec > 0:
            scan_deadline = time.monotonic() + timeout_sec

        image_tag = self.image_tag

        probe_name = updated_top_node_spec.metadata.get('updated_by_prob', None)
        if probe_name and 'memory_poisoning' in probe_name:
            image_tag = updated_top_node_spec.metadata['instance_info']['poisoned_tag']

        trace_entries = self._build_execution_trace_entries(execution_cmds)
        execution_cmds = [
            entry["execution_cmd"]
            for entry in trace_entries
            if isinstance(entry, dict) and entry.get("execution_cmd")
        ]
        self._store_execution_trace_entries(updated_top_node_spec, trace_entries)

        if image_tag:
            with SandboxSession(
                verbose=True,
                lang="python",
                keep_template=False,
                image=image_tag,
                # reuses the same container for subsequent scans to speed up execution. 
            ) as session:
                try:
                    self.sa.apply_changes(updated_top_node_spec, session, affected_nodes)
                    copy_env_to_sandbox(session, self.env_file_path)
                    start_mlflow_ui_in_sandbox(session, mlruns_root="/sandbox/mlruns")

                    # copy all files from sandbox to host - this should be used for debugging only.
                    # copy_file_from_sandbox(session, "/sandbox", "./debugg")

                    res = {}
                    trace_save_path = ""
                    for trace_entry in trace_entries:
                        execution_cmd = trace_entry["execution_cmd"]
                        trace_key = trace_entry["trace_key"]
                        command_timeout_sec = timeout_sec
                        if scan_deadline is not None:
                            remaining_sec = int(scan_deadline - time.monotonic())
                            if remaining_sec <= 0:
                                raise TimeoutError(
                                    "Sandbox scan timed out before command execution "
                                    f"(timeout={timeout_sec}s): {execution_cmd}"
                                )
                            command_timeout_sec = remaining_sec

                        trace_save_path = self.execute_command_and_extract_trace(
                            execution_cmd,
                            session,
                            timeout_sec=command_timeout_sec,
                        )
                        if not trace_save_path:
                            trace_save_path = ""
                        if trace_save_path == "":
                            print("WARNING: EMPTY TRACE PATH")
                            raise ValueError("trace_save_path is empty")
                        res[trace_key] = trace_save_path

                    session.container.commit(
                        repository=self.repo_name,
                        tag="malicious",
                    )
                    return res
                finally:
                    container = getattr(session, "container", None)
                    container_id = getattr(container, "id", None)
                    if container_id:
                        force_remove_container(
                            container_id,
                            use_sudo=True,
                        )
        else:
            self.sa.apply_changes(updated_top_node_spec, affected_nodes=affected_nodes)
