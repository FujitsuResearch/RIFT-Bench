from llm_sandbox.core.session_base import BaseSession
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Union
import io
import tarfile 
import shutil

from resources.system_config import EMULATED_CONFIG_SAVE_NAME
from node_spec.structure_schema import NodeSpec
from .util import *
import os
import re


class ScenarioAdaptator:
    def __init__(self, root_directory_path: str) -> None:
        """
        Initialize a `ScenarioAdaptator` instance.

        Args:
            root_directory_path (str): Path to the root directory of the code repository.
        """
        self.root_directory_path = root_directory_path
        self.session: BaseSession | None = None
        self.test = True

    def write_to_sb(self, path_to_save: Path, content: str) -> None:
        """
        Write content to a file inside the sandbox container.

        Args:
            path_to_save (Path): Destination file path.
            content (str): File content to write.
        """
        if self.session:
            cmd = (
                "import os\n"
                f"path = {str(path_to_save)!r}\n"
                f"content = {content!r}\n"
                "os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
                "with open(path, 'w', encoding='utf-8') as f:\n"
                "    f.write(content)\n"
            )
            self.session.run(cmd)
        else:
            path_to_save.write_text(content, encoding="utf-8")

    def read_from_sb(self, path: str | Path) -> str:
        """
        Read a file from the sandbox container.

        Args:
            path (str): Absolute file path inside the container or local filesystem.

        Return:
            str: File content.
        """
        path = str(path)

        if self.session:
            archive_bytes, stat = self.session.get_archive(path)

            if stat.get("size", 0) == 0:
                raise FileNotFoundError(f"File not found in sandbox: {path}")

            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tar:
                member = next((m for m in tar.getmembers() if m.isfile()), None)
                if member is None:
                    raise FileNotFoundError(f"No file content found in archive for: {path}")

                extracted = tar.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(f"Could not extract file: {path}")

                return extracted.read().decode("utf-8")

        return Path(path).read_text(encoding="utf-8")

    def apply_changes(
        self,
        root_node_spec: NodeSpec,
        session: BaseSession | None = None,
        affected_nodes: Union[
            str,
            Dict[str, List[str]],
            List[str],
            List[Dict[str, List[str]]],
            List[Union[str, Dict[str, List[str]]]],
        ] = "all",
    ) -> None:
        """
        Apply node-spec code changes to repository files based on selected nodes.

        Args:
            root_node_spec (NodeSpec): The top level NodeSpec object containing the updated code snippets to apply.
            session (BaseSession, optional): An active sandbox session to apply the changes in. If None, changes will be applied directly to the file system. Defaults to None.
            affected_nodes (str or list, optional): Either "all", a legacy list of node ids, or a list of `{node_id: [kind, ...]}` mappings.
        """
        self.session = session
        snippets_per_file = defaultdict(list)
        affected_nodes_by_kind = self._normalize_affected_nodes(affected_nodes)

        self.update_emulation_preferences(root_node_spec)

        # Go over the all the nodes in affected_nodes in oreder to arrange snipets by file.
        for node in root_node_spec.iter_descendants(include_self=True):
            allowed_kinds: Set[str] | None = None
            if affected_nodes_by_kind != "all":
                if node.id not in affected_nodes_by_kind:
                    continue
                allowed_kinds = affected_nodes_by_kind[node.id]

            for code_ref in node.code_references:
                if (
                    allowed_kinds is not None
                    and getattr(code_ref, "kind", None) not in allowed_kinds
                ):
                    continue

                # For each snippet we save the code snippet and the refernace lines.
                snippets_per_file[code_ref.file].append(
                    (code_ref.snippet, code_ref.line)
                )

        # Iter over the files to update and update each file speartly.
        for file_path, snippets in snippets_per_file.items():
            self.update_file_with_snippets(file_path, snippets)
        
        if self.test:
            self.copy_sandbox_content_to_output()

    def update_emulation_preferences(self, root_node_spec):
        tools = get_all_tools(root_node_spec)
        tools_names = []

        for tool_type, tools_set in tools.items():
            if tool_type == "local":
                tools_names.extend(
                    [(i["id"], i["name"]) for v in tools_set.values() for i in v]
                )
            elif tool_type == "external":
                tools_names.extend(
                    [(root_node_spec.get_node(k).id,root_node_spec.get_node(k).name) for k in tools_set.keys()]
                )
            else:
                tools_names.extend([(i["id"], i["name"]) for i in tools_set])

        tools_names = "".join(
            [
                f"{i[1].upper()}_EMULATION = {root_node_spec.get_node(i[0]).emulated} \n"
                for i in tools_names
            ]
        )
        self.write_to_sb(
            Path(f"{self.root_directory_path}/{EMULATED_CONFIG_SAVE_NAME}.py"),
            tools_names,
        )
    
    @staticmethod
    def _normalize_affected_nodes(
        affected_nodes: Union[
            str,
            Dict[str, List[str]],
            List[str],
            List[Dict[str, List[str]]],
            List[Union[str, Dict[str, List[str]]]],
        ],
    ) -> str | Dict[str, Set[str] | None]:
        """
        Normalize affected-node selectors to a node->kinds mapping.

        Returns:
            - "all": apply every code reference.
            - dict[node_id] = None: apply all kinds for that node (legacy node-id list).
            - dict[node_id] = {kinds...}: apply only those code-reference kinds.
        """
        if affected_nodes == "all":
            return "all"

        normalized: Dict[str, Set[str] | None] = {}
        if isinstance(affected_nodes, dict):
            entries = [affected_nodes]
        else:
            entries = affected_nodes or []

        for entry in entries:
            if isinstance(entry, str):
                normalized[entry] = None
                continue
            if not isinstance(entry, dict):
                continue

            for node_id, kinds in entry.items():
                if not isinstance(node_id, str):
                    continue

                if normalized.get(node_id) is None and node_id in normalized:
                    continue

                kinds_set = normalized.get(node_id)
                if kinds_set is None:
                    kinds_set = set()

                for kind in kinds or []:
                    if kind:
                        kinds_set.add(str(kind))
                normalized[node_id] = kinds_set

        return normalized

    def copy_sandbox_content_to_output(
        self,
        output_path: str | Path = "sandbox_output",
    ) -> None:
        output_dir = Path(output_path)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.session:
            archive_bytes, stat = self.session.get_archive(str(self.root_directory_path))
            if stat.get("size", 0) == 0:
                return

            base_output = output_dir.resolve()
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tar:
                for member in tar.getmembers():
                    member_path = Path(member.name.lstrip("/"))
                    target_path = (output_dir / member_path).resolve()

                    if not str(target_path).startswith(str(base_output)):
                        continue

                    if member.isdir():
                        target_path.mkdir(parents=True, exist_ok=True)
                        continue

                    # Docker archives may include symlink/hardlink entries whose targets
                    # are not part of the streamed tar. Skip those to avoid KeyError.
                    if not member.isfile():
                        continue

                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue

                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with target_path.open("wb") as f:
                        f.write(extracted.read())
        else:
            shutil.copytree(self.root_directory_path, output_dir, dirs_exist_ok=True)

    def exists_sb(self, file_path: str | Path) -> bool:
        cmd = f"""
import os
import sys
sys.exit(0 if os.path.exists("{file_path}") else 1)
"""
        out = self.session.run(cmd)

        return getattr(out, "exit_code", 1) == 0

    def update_file_with_snippets(
        self,
        file_path: str,
        snippets: list[tuple[str, int | list[int] | tuple[int, int]]],
    ) -> None:
        """
        Update a single file according to the set of updated snippets.

        Args:
            file_path (str): Path of the file to update.
            snippets (list[tuple[str, int | list[int] | tuple[int, int]]]): Snippet and line-selector pairs.
        """
        # Store the snippets base on the lines from high to low so changes made wont effect the rest.
        snippets = sorted(
            snippets,
            key=lambda x: x[1] if isinstance(x[1], int) else x[1][-1],
            reverse=True,
        )
        file_path = Path(os.path.join(self.root_directory_path, file_path))

        # Check if the file need to be updated or created.
        if (self.session and self.exists_sb(file_path)) or os.path.exists(file_path):

            # Case of exsisiting file - we load it.
            file_lines = self.read_from_sb(file_path).splitlines(keepends=True)
        else:
            # Case of new file - we create blank content to be update.
            file_lines = ["\n"] * normalize_line_numbers(snippets[0][-1])[-1]

        # Insert each of the snippets into the file upwards in the file.
        for snippet, line_numbers in snippets:
            file_lines = self.replace_lines_with_snippet(
                file_lines, snippet, line_numbers
            )

        # Write the new/updated file to the sandbox.
        self.write_to_sb(Path(file_path), "".join(file_lines))

    def replace_lines_with_snippet(
        self,
        lines: List[str],
        snippet: str,
        line_numbers: int | list[int] | tuple[int, int],
    ) -> List[str]:
        """
        Replace specific line(s) with snippet, preserving contextual indentation.

        Special cases:
        - line_numbers == 0  -> insert at beginning
        - line_numbers == -1 -> insert at end
        """

        def is_blank(line: str) -> bool:
            return not line.strip()

        def leading_ws(line: str) -> str:
            return line[: len(line) - len(line.lstrip())]

        def detect_newline(lines: List[str]) -> str:
            for line in lines:
                if line.endswith("\r\n"):
                    return "\r\n"
                if line.endswith("\n"):
                    return "\n"
            return "\n"

        def common_indent(snippet_lines: List[str]) -> str:
            indents = [leading_ws(line) for line in snippet_lines if line.strip()]
            if not indents:
                return ""

            prefix = indents[0]
            for indent in indents[1:]:
                while not indent.startswith(prefix):
                    prefix = prefix[:-1]
                    if not prefix:
                        return ""
            return prefix

        def expand_partial_statement_range(
            start_idx: int,
            end_idx: int,
        ) -> tuple[int, int]:
            """Expand a range that starts in the middle of a Python statement.

            Some discovered node specs have a one-line offset in an assignment
            reference. Replacing that range with a complete assignment leaves
            the original opening line behind (for example, an unclosed
            ``StructuredTool(``). When both the source and replacement are
            parseable single statements, use the source statement boundaries.
            """
            try:
                source_tree = ast.parse("".join(lines))
                replacement_tree = ast.parse(snippet)
            except (IndentationError, SyntaxError):
                return start_idx, end_idx

            if len(replacement_tree.body) != 1:
                return start_idx, end_idx

            replacement_statement = replacement_tree.body[0]
            for source_statement in source_tree.body:
                statement_start = source_statement.lineno - 1
                statement_end = source_statement.end_lineno
                if (
                    isinstance(source_statement, type(replacement_statement))
                    and statement_start < start_idx < statement_end
                ):
                    return statement_start, statement_end

            return start_idx, end_idx

        newline = detect_newline(lines)

        # Resolve replacement bounds
        if isinstance(line_numbers, int):
            if line_numbers == 0:
                start_idx = end_idx = 0
            elif line_numbers == -1:
                start_idx = end_idx = len(lines)
            elif line_numbers >= 1:
                start_idx = line_numbers - 1
                end_idx = line_numbers
            else:
                raise ValueError("line_numbers must be >= 1, 0, or -1")
        else:
            if len(line_numbers) != 2:
                raise ValueError("line_numbers must be an int or [start, end]")

            start, end = line_numbers
            if start < 1 or end < start:
                raise ValueError("Invalid line range")

            start_idx = start - 1
            end_idx = end

        # Clamp to content bounds
        start_idx = max(0, min(start_idx, len(lines)))
        end_idx = max(start_idx, min(end_idx, len(lines)))
        start_idx, end_idx = expand_partial_statement_range(start_idx, end_idx)

        # Infer indentation from nearest useful context
        base_indent = ""

        # Prefer the replaced line / next line
        for i in range(start_idx, len(lines)):
            if not is_blank(lines[i]):
                base_indent = leading_ws(lines[i])
                break
        else:
            # Fallback upward
            for i in range(start_idx - 1, -1, -1):
                if not is_blank(lines[i]):
                    base_indent = leading_ws(lines[i])
                    break

        # Normalize snippet
        raw_snippet_lines = snippet.rstrip("\r\n").splitlines()
        if snippet and not raw_snippet_lines:
            raw_snippet_lines = [snippet]

        if not raw_snippet_lines:
            indented_snippet_lines = []
        else:
            snippet_base_indent = common_indent(raw_snippet_lines)

            indented_snippet_lines = []
            for line in raw_snippet_lines:
                if not line.strip():
                    indented_snippet_lines.append(newline)
                    continue

                normalized = (
                    line[len(snippet_base_indent):]
                    if line.startswith(snippet_base_indent)
                    else line
                )
                indented_snippet_lines.append(base_indent + normalized + newline)

        return lines[:start_idx] + indented_snippet_lines + lines[end_idx:]
