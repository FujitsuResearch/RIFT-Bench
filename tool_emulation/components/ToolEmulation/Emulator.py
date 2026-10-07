import os
import re
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal
import tarfile
import io

from .utils.node_spec_oprations import *
from .utils.coding import *
from .LLMEmulation import LLMEmulation
from node_spec.structure_schema import CodeReference, NodeSpec
from resources.system_config import EMULATED_TOOLS_FILE_SAVE_NAME, EMULATED_CONFIG_SAVE_NAME

FAILED_EMULATION_INTEGRATION_NODE_IDS_METADATA_KEY = (
    "failed_emulation_integration_node_ids"
)

class Emulator:
    def __init__(
        self,
        handler: Literal["LanguageHandler"] = None,
        emulation_stratgy_name: str = "llm",
        default_config: bool = False,
    ):
        """
        Initialize an `Emulator` instance.

        Args:
            handler (Literal["LanguageHandler"]): Environment handler.
            emulation_stratgy_name (str): Emulation strategy name. Currently, only `"llm"` is supported.
            default_config (bool): Default value for emulation flags (whether to use original or emulated tools).
        """
        self.handler: Literal["LanguageHandler"] = handler
        self.root_directory_path: str = handler.sandbox_root_directory_path

        self.root_node_spec: NodeSpec = None
        self.tools: dict = None
        self.default_config: bool = default_config

        self.emulation_stratgy_name: str = emulation_stratgy_name
        self.emulation_stratgy: LLMEmulation = None

        self.local_mcps_dir: str = ""
        self.failed_emulation_integration_node_ids: set[str] = set()

        self.session: Literal["SandboxSession"] = None

    def _reset_failed_emulation_integration_node_ids(self) -> None:
        self.failed_emulation_integration_node_ids = set()

    def _record_failed_emulation_integration_node_id(self, node_id: str) -> None:
        if node_id:
            self.failed_emulation_integration_node_ids.add(node_id)

    def _update_failed_emulation_integration_metadata(self) -> None:
        if self.root_node_spec.metadata is None:
            self.root_node_spec.metadata = {}

        self.root_node_spec.metadata[
            FAILED_EMULATION_INTEGRATION_NODE_IDS_METADATA_KEY
        ] = sorted(self.failed_emulation_integration_node_ids)

    def _configure_from_root_node_spec(self, root_node_spec: NodeSpec) -> None:
        """
        Configure emulator state from the provided root node specification.

        Args:
            root_node_spec (NodeSpec): The root node specification of the system for which the tools are to be emulated.
        """
        self.root_node_spec = deepcopy(root_node_spec)
        self.get_tools()

        self._initialize_emulation_strategy()

    def _initialize_emulation_strategy(self) -> None:
        """
        Initialize the configured emulation strategy.
        """
        if self.emulation_stratgy_name == "llm":
            self.emulation_stratgy = LLMEmulation(
                node_spec=self.root_node_spec,
                enf_file_path=self.handler.path_to_envfile
            )

    def get_tools(self) -> None:
        """
        Collect all tool definitions from the current node specification.
        """
        self.tools = get_all_tools(self.root_node_spec)

    def write_to_sb(self, path_to_save: Path | str, content: str) -> None:
        """
        Write content to a file, either in the sandbox session or directly on disk.

        Args:
            path_to_save (Path | str): Destination file path.
            content (str): File content to write.
        """
        if self.session:
            cmd = (
                f"""
import os

os.makedirs(os.path.dirname("{path_to_save}"), exist_ok=True)

with open("{path_to_save}", "w", encoding="utf-8") as f:
    f.write("""
                + repr(content)
                + """)
    """
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

    def duplicate_path(self, src_path: str, dst_path: str) -> None:
        """
        Duplicate a file from source path to destination path.

        Args:
            src_path (str): Source file path.
            dst_path (str): Destination file path.
        """
        src = Path(src_path)
        dst = Path(dst_path)

        file = self.read_from_sb(src)
        self.write_to_sb(dst, file)


    def create_config_file(self) -> None:
        """
        Generate and save a configuration file for the emulated tools and servers.
        """
        tools_names = []

        for tool_type, tools_set in self.tools.items():
            if tool_type == "local":
                tools_names.extend(
                    [item["name"] for v in tools_set.values() for item in v]
                )
            elif tool_type == "external":
                tools_names.extend(
                    [self.root_node_spec.get_node(k).name for k in tools_set.keys()]
                )
            else:
                tools_names.extend([i["name"] for i in tools_set])

        tools_names = "".join(
            [
                f"{i.upper()}_EMULATION = {str(self.default_config)} \n"
                for i in tools_names
            ]
        )
        self.write_to_sb(
            Path(f"{self.root_directory_path}/{EMULATED_CONFIG_SAVE_NAME}.py"),
            tools_names,
        )

    def create_emulations(
        self, root_node_spec: NodeSpec, session: Literal["SandboxSession"] = None
    ) -> NodeSpec:
        """
        Create emulations for regular tools, local MCP servers, and external MCP servers.

        Args:
            root_node_spec (NodeSpec): Root node specification to emulate.
            session (Literal["SandboxSession"]): Optional runtime session used to save files in a sandbox container.

        Return:
            NodeSpec: Updated root node specification with refreshed code references.
        """
        self._configure_from_root_node_spec(root_node_spec)
        self._reset_failed_emulation_integration_node_ids()
        self.session = session
        self.create_config_file()
        files_to_update = []
        emulation_file = None
        if self.tools["regular"]:
            regular_tools_files_to_update, emulation_file = self.create_regular_tools_emulations()
            files_to_update.extend(regular_tools_files_to_update)
        if self.tools["local"]:
            files_to_update.extend(self.create_local_mcp_emulations(emulation_file))
        if self.tools["external"]:
            files_to_update.extend(self.create_external_mcp_emulations())

        for file_path in list(set(files_to_update)):
            file_content = self.read_from_sb(
                os.path.join(self.root_directory_path, file_path)
            )
            update_nodespec(
                node_spec=self.root_node_spec,
                file_path=file_path,
                file_content=file_content,
            )
        self._update_failed_emulation_integration_metadata()
        return self.root_node_spec

    def create_regular_tools_emulations(self) -> list[str]:
        """
        Create emulations for standard (non-MCP) tools.

        Return:
            list: File paths that were updated.
        """
        successful_tools, emulation_file = self.extend_emulation_file(
            self.tools["regular"]
        )

        # Update the original code of the tools and return the files thats need to be updated in the node_spec
        return self.update_original_code(successful_tools), emulation_file

    def extend_emulation_file(
        self,
        tools: list[dict[str, Any]],
        emulation_file: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        save_path = f"{self.root_directory_path}/{EMULATED_TOOLS_FILE_SAVE_NAME}.py"
        if not emulation_file:
            emulation_file = '\n'.join(['from __future__ import annotations',self.emulation_stratgy.create_basic_emulation_file(), ''])

        successful_tools = []
        for tool in tools:
            try:
                tool_node = self.root_node_spec.get_node(tool["id"])

                # Append emulation of single tool to the emulation file.
                emulation_file = self.emulation_stratgy.create_tool_emulation(
                    tool_node, f"{EMULATED_TOOLS_FILE_SAVE_NAME}.py", emulation_file
                )
                successful_tools.append(tool)
            except Exception:
                self._record_failed_emulation_integration_node_id(tool["id"])

        # Save the emulation file in the project direcory
        self.write_to_sb(Path(save_path), emulation_file)
        return successful_tools, emulation_file

    def update_original_code(self, tools: list[dict[str, Any]]) -> list[str]:
        """
        Update the original source code with tool emulations selection.

        Args:
            tools (list): Tool definitions to alter in the original code.

        Return:
            list: File paths that were updated.
        """
        def get_duplicate_node_ids(tool_node: NodeSpec) -> list[str]:
            duplicates = getattr(tool_node, "duplicates", None)
            if not duplicates or not getattr(duplicates, "exists", False):
                return []

            duplicate_ids = getattr(duplicates, "ids", None)
            if duplicate_ids:
                return list(duplicate_ids)

            duplicate_instances = getattr(duplicates, "instances", None) or []
            return [instance["id"] for instance in duplicate_instances]

        def replace_definition_ref(
            tool_node: NodeSpec,
            definition_file_path: str,
            original_definition_snippet: str,
            updated_definition_snippet: str,
        ) -> None:
            if tool_node.code_references is None:
                tool_node.code_references = []

            definition_ref_index = next(
                (
                    index
                    for index, code_ref in enumerate(tool_node.code_references)
                    if code_ref.kind == "definition"
                    and code_ref.file == definition_file_path
                ),
                None,
            )

            if definition_ref_index is None:
                tool_node.code_references.append(
                    CodeReference(
                        kind="definition",
                        file=definition_file_path,
                        line=1,
                        snippet=updated_definition_snippet,
                    )
                )
            else:
                updated_definition_ref = deepcopy(
                    tool_node.code_references[definition_ref_index]
                )
                updated_definition_ref.snippet = updated_definition_snippet
                tool_node.code_references[definition_ref_index] = updated_definition_ref

            for code_ref in tool_node.code_references:
                if (
                    code_ref.kind == "assignment"
                    and code_ref.file == definition_file_path
                    and code_ref.snippet == original_definition_snippet
                ):
                    code_ref.snippet = updated_definition_snippet

        def refresh_definition_ref_lines(
            nodes_to_update: dict[str, NodeSpec],
            definition_file_path: str,
            file_content: str,
        ) -> None:
            for tool_node in nodes_to_update.values():
                if not tool_node.code_references:
                    continue

                for code_ref in tool_node.code_references:
                    if (
                        code_ref.kind == "definition"
                        and code_ref.file == definition_file_path
                    ):
                        updated_line = find_snippet_lines(
                            file_content, code_ref.snippet
                        )
                        if updated_line is not None:
                            code_ref.line = updated_line

        files_to_update = defaultdict(list)

        # Arrange the tools definitions by files
        for tool in tools:
            try:
                files_to_update[tool["definition_file_path"]].append(tool)
            except Exception:
                continue

        # Go over each of the files that need to be altered.
        for file_path, set_of_tools in files_to_update.items():

            # Sort the tools definitions by line from high to low.
            set_of_tools = sorted(set_of_tools, key=lambda x: x["definition_lines"][-1], reverse=True)
            updated_nodes_for_file: dict[str, NodeSpec] = {}

            final_file_path = Path(os.path.join(self.root_directory_path, file_path))

            # Read the original file.
            file_content = self.read_from_sb(final_file_path)

            # Go over each of the tool's definitions in the file
            for tool in set_of_tools:
                try:
                    tool_node = self.root_node_spec.get_node(tool["id"])

                    name, desc, exp_out, exp_inp, tool_example_pairs = (
                        tool_node.metadata.get(
                            "component_handle", sanitize_function_name(tool_node.name)
                        ),
                        tool_node.description,
                        tool_node.outputs,
                        tool_node.inputs,
                        tool_node.tool_example_pairs,
                    )
                    args_names = [input_item.name for input_item in exp_inp]
                    if len(args_names) != 1:
                        tool_args = ', '.join(args_names)
                    else:
                        tool_args = args_names[0] + ','
                    exp_out = format_expected_io_for_prompt(exp_out)
                    exp_inp = format_expected_io_for_prompt(exp_inp)
                    if tool_example_pairs:
                        tool_example_pairs = [
                            (pair.input, unwrap_single_key_dict(pair.output))
                            for pair in tool_example_pairs
                        ][:3]
                    # Add imports to the emulated tool from the emulation file as well as to
                    # the config file to get the flag for this tool emulation.

                    imports = f"""try:
    from {EMULATED_CONFIG_SAVE_NAME} import {tool["name"].upper()}_EMULATION
    from {EMULATED_TOOLS_FILE_SAVE_NAME} import llm_emulate_response

except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[{file_path.count("/")}]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from {EMULATED_CONFIG_SAVE_NAME} import {tool["name"].upper()}_EMULATION
    from {EMULATED_TOOLS_FILE_SAVE_NAME} import llm_emulate_response"""

                    if_to_add = f"""{imports}

if {tool["name"].upper()}_EMULATION:
    return llm_emulate_response(
        name={name!r},
        desc={desc!r},
        tool_args=({tool_args}),
        exp_out={exp_out!r},
        exp_in={exp_inp!r},
        tool_example_pairs={tool_example_pairs!r},
    )
"""
                    original_definition_snippet = tool["definition_snippet"]
                    definition_insert_line = get_first_executable_line(
                        original_definition_snippet
                    )
                    if definition_insert_line is None:
                        raise ValueError(
                            f"Could not find an executable line for tool definition: {tool['id']}"
                        )

                    updated_file_content = insert_line_in_content(
                        file_content,
                        tool["definition_lines"][0] + definition_insert_line - 1,
                        if_to_add,
                    )

                    updated_definition_snippet = insert_line_in_content(
                        original_definition_snippet,
                        definition_insert_line,
                        if_to_add,
                    )

                    nodes_to_update = {tool_node.id: tool_node}
                    for duplicate_node_id in get_duplicate_node_ids(tool_node):
                        duplicate_tool_node = self.root_node_spec.get_node(duplicate_node_id)
                        if duplicate_tool_node is not None:
                            nodes_to_update[duplicate_tool_node.id] = duplicate_tool_node

                    for current_tool_node in nodes_to_update.values():
                        replace_definition_ref(
                            current_tool_node,
                            tool["definition_file_path"],
                            original_definition_snippet,
                            updated_definition_snippet,
                        )
                        updated_nodes_for_file[current_tool_node.id] = current_tool_node

                    tool["definition_snippet"] = updated_definition_snippet
                    file_content = updated_file_content
                except Exception:
                    self._record_failed_emulation_integration_node_id(tool["id"])
                    continue

            refresh_definition_ref_lines(
                updated_nodes_for_file,
                file_path,
                file_content,
            )

            # Save the updated file inplace
            self.write_to_sb(Path(final_file_path), file_content)

        # Return list of files that were updated
        return list(files_to_update.keys())

    def create_local_mcp_emulations(self, emulation_file=None) -> set[str]:
        """
        Create emulations for local MCP servers.

        Return:
            set: File paths that were updated.
        """
        files_to_update = []

        self.set_local_mcps_dir()

        # Go over the local MCPs
        for v in self.tools["local"].values():
            successful_tools, emulation_file = self.extend_emulation_file(
                v, emulation_file
            )
            files_to_update.extend(self.update_original_code(successful_tools))

        # Return list of files that were updated
        return list(set(files_to_update))

    def set_local_mcps_dir(self):
        mcp_servers = self.root_node_spec.list_by_types({"Local_MCP_server"})
        for mcp_server in mcp_servers:
            mcp_node = self.root_node_spec.get_node(mcp_server['ids'][0])
            if not self.is_connected_to_agent(mcp_node):
                continue
            # Extract the MCP's definition file path
            try:
                mcp_node_server_definition_file = [
                    i
                    for i in mcp_node.code_references
                    if i.kind == "definition"
                ][0].file
            except:
                continue

            # Save the location of MCP files
            self.local_mcps_dir = mcp_node_server_definition_file.replace(
                mcp_node_server_definition_file.split("/")[-1], ""
            )


    def is_connected_to_agent(self, node):
        agents = self.root_node_spec.list_agents()
        for agent in agents:
            agent_node = self.root_node_spec.get_node(agent['ids'][0])
            if node in agent_node.nodes:
                return True
        return False

    def create_external_mcp_emulations(self) -> list[str]:
        """
        Create emulations for external MCP servers.

        Return:
            list: File paths that were updated.
        """
        files_to_update = set()
        external_mcps = []

        # Go over the external MCPs
        for k, v in self.tools["external"].items():
            mcp_node = self.root_node_spec.get_node(k)
            try:
                mcp_name = mcp_node.name
                mcp_node_assignment_file = [
                    i for i in mcp_node.code_references if i.kind == "assignment"
                ][0].file
                files_to_update.add(mcp_node_assignment_file)

                # If the local MCPs have a dir, will save our emulation there, otherwise in the main directory.
                if self.local_mcps_dir == "":
                    save_path = f"{mcp_name}_{EMULATED_TOOLS_FILE_SAVE_NAME}.py"
                else:
                    save_path = f"{self.local_mcps_dir}{mcp_name}_{EMULATED_TOOLS_FILE_SAVE_NAME}.py"

                # Create full emulation of the MCP (either using the original or all emulated - not hybrid)
                emulation_file = self.create_full_mcp_emulation(mcp_node, save_path, v)
                self.write_to_sb(Path(os.path.join(self.root_directory_path, save_path)),emulation_file)

                # Save the MCP definition file
                self.write_to_sb(
                    Path(os.path.join(self.root_directory_path, save_path)), emulation_file
                )
                external_mcps.append((mcp_node, save_path))
            except Exception:
                self._record_failed_emulation_integration_node_id(mcp_node.id)
                continue

        # Insert the emulated MCP definition to exisiting file or create new file for it.
        self.update_mcps_main_files(external_mcps)

        # Return list of files that were updated
        return list(files_to_update)

    def create_full_mcp_emulation(
        self, mcp_node: NodeSpec, save_path: str, tools: list[dict[str, Any]]
    ) -> str:
        """
        Create a fully emulated MCP server for external MCPs.

        Args:
            mcp_node (NodeSpec): MCP node specification.
            save_path (str): File path where the generated emulation file is stored.
            tools (list): Tool definitions available for emulation.

        Return:
            str: Generated emulation file content.
        """
        server_name = mcp_node.name

        # Define the imports required and set the root
        imports = f"""
from fastmcp import FastMCP
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[{save_path.count('/')}]  # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT))
"""

        # Crete MCP server definition
        mcp_definition = f"emulated_{server_name} = FastMCP('emulated_{server_name}')"

        # Create execution command for the server in the main.
        end = f"""
if __name__ == "__main__":
    emulated_{server_name}.run()
"""
        # Define decortor for the server to link the tools
        decoretor = f"@emulated_{server_name}.tool()"

        emulation_file = '\n'.join(
            ['from __future__ import annotations', imports, self.emulation_stratgy.create_basic_emulation_file(), '']
            )


        update_info = [
            CodeReference(
                kind="definition",
                file=save_path,
                line=len(emulation_file.splitlines()) + 1,
                snippet=mcp_definition,
            )
        ]

        emulation_file += mcp_definition

        mcp_node.emulated_code_references = update_info

        # Iter over the tools to create the tool emulation.
        for tool in tools:

            tool_node = mcp_node.get_node(tool["id"])

            # Build the tool signature using the inputs definition from the nodespec.
            tool_signature = build_function_signature(
                tool_node.metadata.get(
                    'component_handle', sanitize_function_name(tool_node.name)
                ),
                tool_node.inputs,
            )

            # Append the new emulated tool to the end of the file
            emulation_file = self.emulation_stratgy.create_external_mcp_tool_emulation(
                tool_node, save_path, tool_signature, emulation_file, decoretor
            )

        # Join the emulation file and server execution together
        emulation_file = "\n".join([emulation_file, end])
        return emulation_file

    def update_mcps_main_files(self, mcps: list[tuple[NodeSpec, str]]) -> None:
        """
        Update MCP definition files to include emulated MCP instances and selection flags.

        Args:
            mcps (list[tuple]): Tuples in the form `(mcp_node, save_path)` for external MCP servers.
        """

        def reindent_block(block: str, base_indent: str = "") -> str:
            normalized = textwrap.dedent(block).strip("\n")
            if not normalized:
                return ""

            return "\n".join(
                f"{base_indent}{line}" if line.strip() else ""
                for line in normalized.splitlines()
            )

        files_to_update = defaultdict(list)
        for mcp_node, save_path in mcps:
            try:
                mcp_node_assignment = [
                    i for i in mcp_node.code_references if i.kind == "assignment"
                ][0]
                files_to_update[mcp_node_assignment.file].append(
                    (mcp_node, normalize_line_numbers(mcp_node_assignment.line)[-1], save_path)
                )
            except Exception:
                self._record_failed_emulation_integration_node_id(mcp_node.id)
                continue

        for file_path, set_of_mcps in files_to_update.items():
            set_of_mcps = sorted(set_of_mcps, key=lambda x: x[1], reverse=True)
            dest_file_path = Path(os.path.join(self.root_directory_path, file_path))
            file_content = self.read_from_sb(dest_file_path)
            block_replacements: list[tuple[str, str]] = []
            assignment_updates: list[tuple[NodeSpec, CodeReference, str]] = []

            for mcp_node, _mcp_end_line, save_path in set_of_mcps:
                try:
                    mcp_node_assignment_name = mcp_node.metadata.get(
                        "component_handle", sanitize_function_name(mcp_node.name)
                    )
                    mcp_node_assignment = [
                        i for i in mcp_node.code_references if i.kind == "assignment"
                    ][0]
                    original_assignment_snippet = mcp_node_assignment.snippet

                    external_mcp_sub_snippet = find_external_mcp_sub_snippet(
                        mcp_node_assignment_name=mcp_node_assignment_name,
                        snippet=mcp_node_assignment.snippet,
                        description=mcp_node.description,
                    )
                    external_mcp_sub_snippet_block = find_external_mcp_sub_snippet_block(
                        external_mcp_sub_snippet,
                        mcp_node_assignment.snippet,
                    )
                    external_mcp_sub_snippet_block_with_emulated_version = create_external_mcp_sub_snippet_block(
                        external_mcp_sub_snippet_block=external_mcp_sub_snippet_block,
                        external_mcp_sub_snippet=external_mcp_sub_snippet,
                        mcp_node_assignment_name=mcp_node_assignment_name,
                        save_path=save_path,
                        description=mcp_node.description,
                    )

                    if external_mcp_sub_snippet_block not in file_content:
                        raise ValueError(
                            f"Could not find MCP block in {file_path} for {mcp_node.name}."
                        )

                    flag_name = f"{mcp_node.name.upper()}_EMULATION"
                    import_line = f"from {EMULATED_CONFIG_SAVE_NAME} import {flag_name}"

                    base_indent = ""
                    for line in external_mcp_sub_snippet_block.splitlines():
                        if line.strip():
                            base_indent = re.match(r"[ 	]*", line).group(0)
                            break

                    original_block = textwrap.dedent(external_mcp_sub_snippet_block).strip("\n")
                    emulated_block = textwrap.dedent(
                        external_mcp_sub_snippet_block_with_emulated_version
                    ).strip("\n")

                    conditional_block = "\n".join(
                        [
                            import_line,
                            "",
                            f"if {flag_name}:",
                            textwrap.indent(emulated_block, "    ") if emulated_block else "    pass",
                            "else:",
                            textwrap.indent(original_block, "    ") if original_block else "    pass",
                        ]
                    )
                    conditional_block = reindent_block(conditional_block, base_indent)

                    file_content = file_content.replace(
                        external_mcp_sub_snippet_block,
                        conditional_block,
                        1,
                    )
                    block_replacements.append((external_mcp_sub_snippet_block, conditional_block))
                    assignment_updates.append(
                        (mcp_node, mcp_node_assignment, original_assignment_snippet)
                    )
                except Exception:
                    self._record_failed_emulation_integration_node_id(mcp_node.id)
                    continue

            for mcp_node, mcp_node_assignment, original_assignment_snippet in assignment_updates:
                try:
                    updated_assignment_snippet = original_assignment_snippet
                    for original_block, replacement_block in block_replacements:
                        if original_block in updated_assignment_snippet:
                            updated_assignment_snippet = updated_assignment_snippet.replace(
                                original_block,
                                replacement_block,
                                1,
                            )

                    grounded_line = find_snippet_lines(file_content, updated_assignment_snippet)
                    if grounded_line is None:
                        raise ValueError(
                            f"Could not ground updated assignment snippet in {file_path} for {mcp_node.name}."
                        )

                    mcp_node_assignment.file = str(file_path)
                    mcp_node_assignment.snippet = updated_assignment_snippet
                    mcp_node_assignment.line = grounded_line
                except Exception:
                    self._record_failed_emulation_integration_node_id(mcp_node.id)
                    continue

            self.write_to_sb(Path(dest_file_path), file_content)
