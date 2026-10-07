import os
from typing import Any

from .utils.node_spec_oprations import *
from .utils.coding import *
from pathlib import Path

try:
    from node_spec.structure_schema import CodeReference, NodeSpec
except ImportError:
    from ToolEmulation.structure_schema import CodeReference, NodeSpec

SIMPLE_TEMPLATE_PATH = Path(__file__).with_name("utils") / "simple_template.py"

class LLMEmulation:
    def __init__(self, node_spec: NodeSpec, enf_file_path: str) -> None:
        """
        Initialize an `LLMEmulation` instance.

        Args:
            node_spec (NodeSpec): Top-level node specification used to extract LLM configuration.
        """
        self.root_node_spec = node_spec
        self.copy_env_variables(enf_file_path)


    @staticmethod
    def create_basic_emulation_file() -> str:
        """
        Create the base emulation file from LLM and tool definition templates.

        Return:
            str: Full emulation module content.
        """
        return Path(SIMPLE_TEMPLATE_PATH).read_text(encoding="utf-8")
    
    @staticmethod
    def copy_env_variables(destenation_enf_file_path, source_env_file_path="rift.env"):
        current_env_file_path = Path(source_env_file_path)
        destination_env_file_path = Path(destenation_enf_file_path)

        env_varbiels_to_copy = [
            "AZURE_DEPLOYMENT_NAME_EMULATION",
            "AZURE_MODEL_NAME_EMULATION",
            "AZURE_API_VERSION_EMULATION",
            "AZURE_OPENAI_ENDPOINT_EMULATION",
            "AZURE_API_KEY_EMULATION",
        ]

        if not current_env_file_path.exists():
            raise FileNotFoundError(f"Current env file not found: {current_env_file_path}")

        # Read variables from the current .env file
        variables_to_copy = {}

        for line in current_env_file_path.read_text().splitlines():
            stripped_line = line.strip()

            if not stripped_line or stripped_line.startswith("#"):
                continue

            if "=" not in stripped_line:
                continue

            name, value = stripped_line.split("=", 1)
            name = name.strip()

            if name in env_varbiels_to_copy:
                variables_to_copy[name] = value.strip()

        # Read destination file if it exists
        destination_lines = []
        if destination_env_file_path.exists():
            destination_lines = destination_env_file_path.read_text().splitlines()

        # Replace existing values in destination file
        updated_lines = []
        already_written = set()

        for line in destination_lines:
            stripped_line = line.strip()

            if "=" not in stripped_line or stripped_line.startswith("#"):
                updated_lines.append(line)
                continue

            name, _ = stripped_line.split("=", 1)
            name = name.strip()

            if name in variables_to_copy:
                updated_lines.append(f"{name}={variables_to_copy[name]}")
                already_written.add(name)
            else:
                updated_lines.append(line)

        # Append missing variables
        for name in env_varbiels_to_copy:
            if name in variables_to_copy and name not in already_written:
                updated_lines.append(f"{name}={variables_to_copy[name]}")

        destination_env_file_path.write_text("\n".join(updated_lines) + "\n")

    def format_expected_io_for_prompt(self, exp_io: list[dict]) -> str:
        """
        Convert expected tool inputs/outputs into a clean system-prompt string.

        Args:
            exp_io (list[dict]): Expected inputs or outputs extracted from node metadata.

        Return:
            str: Prompt-friendly formatted text block.
        """

        lines = []
        if exp_io:
            for item in exp_io:
                lines.extend(
                    [
                        f"- name: {item.name}",
                        f"  type: {item.dtype}",
                        f"  description: {item.description}",
                    ]
                )

        return "\n".join(lines)

    def create_tool_emulation(
        self, tool_node: NodeSpec, save_path: str, file_content: str = ""
    ) -> str:
        """
        Create an emulation implementation for a single tool and append it to content.

        Args:
            tool_node (NodeSpec): Tool node to emulate.
            save_path (str): Path of the emulation file used in generated code references.
            file_content (str): Existing file content to append to.

        Return:
            str: Updated file content with the generated tool emulation.
        """
        name, desc, exp_out, exp_inp, tool_example_pairs = (
            tool_node.metadata.get(
                'component_handle', sanitize_function_name(tool_node.name)
            ),
            tool_node.description,
            tool_node.outputs,
            tool_node.inputs,
            tool_node.tool_example_pairs,
        )
        exp_out = format_expected_io_for_prompt(exp_out)
        exp_inp = format_expected_io_for_prompt(exp_inp)

        if tool_example_pairs:
            tool_example_pairs = [
                (pair.input, unwrap_single_key_dict(pair.output))
                for pair in tool_example_pairs
            ][:3]

        stripped_content = file_content.rstrip("\n")

        # Create the tool function
        implementation_block = f"""def {name}_emulated_iner(*args : Any, **kwargs: Any):
    \"\"\"{desc}\"\"\"
    return (
        llm_emulate_response(
            name={name!r},
            desc={desc!r},
            tool_args=(args, kwargs),
            exp_out={exp_out!r},
            exp_in={exp_inp!r},
            tool_example_pairs={tool_example_pairs!r},
        )
    )"""
        # Join the peovided content with the emulation block
        updated_content = "\n".join(
            [stripped_content, implementation_block]
        )

        return updated_content

    def create_external_mcp_tool_emulation(
        self,
        tool_node: NodeSpec,
        save_path: str,
        tool_signature: str,
        file_content: str = "",
        decoretor: str = "",
    ) -> str:
        """
        Create an emulation implementation for a single external MCP tool.

        Args:
            tool_node (NodeSpec): Tool node to emulate.
            save_path (str): Path of the emulation file used in generated code references.
            tool_signature (str): Python signature of the generated tool wrapper.
            file_content (str): Existing file content to append to.
            decoretor (str): Optional decorator line for the generated tool.

        Return:
            str: Updated file content with the generated MCP tool emulation.
        """
        name, desc, exp_out, exp_inp, tool_example_pairs = (
            tool_node.metadata.get(
                'component_handle', sanitize_function_name(tool_node.name)
            ),
            tool_node.description,
            tool_node.outputs,
            tool_node.inputs,
            tool_node.tool_example_pairs,
        )
        exp_out = format_expected_io_for_prompt(exp_out)
        exp_inp = format_expected_io_for_prompt(exp_inp)

        # Format the tool's information to fit the prompt
        if tool_example_pairs:
            tool_example_pairs = [
                (pair.input, unwrap_single_key_dict(pair.output))
                for pair in tool_example_pairs
            ][:3]

        stripped_content = file_content.rstrip("\n")

        # Compute start line (1-based indexing)
        start_line = stripped_content.count("\n") + 4  # blank line before def

        # Extract the tool's argument from the tool's signature
        tool_args = extract_variable_names_from_signature_ast(tool_signature)
        if len(tool_args) != 1:
            tool_args = ', '.join(tool_args)
        else:
            tool_args = tool_args[0] + ','
        update_info = []

        # Create the tool function
        implementation_block = f"""{decoretor}
{tool_signature}:
    \"\"\"{desc}\"\"\"
    return (
        llm_emulate_response(
            name={name!r},
            desc={desc!r},
            tool_args=({tool_args}),
            exp_out={exp_out!r},
            exp_in={exp_inp!r},
            tool_example_pairs={tool_example_pairs!r},
        )
    )"""

        # Calculate the lines of addition for the node spec code reference.
        function_lines = implementation_block.strip("\n").count("\n")
        end_line = start_line + function_lines - 2

        update_info.extend([
            CodeReference(
                kind="assignment",
                file=save_path,
                line=[start_line-1, end_line+1],
                snippet=implementation_block,
            ),
            CodeReference(
                kind="definition",
                file=save_path,
                line=[start_line-1, end_line+1],
                snippet=implementation_block,
            )]
        )

        if tool_node.duplicates.exists:
            instances_ids = [i["id"] for i in tool_node.duplicates.instances]
            for instances_id in instances_ids:
                instances_tool_node = self.root_node_spec.get_node(instances_id)
                instances_tool_node.emulated_code_references = update_info
        else:
            tool_node.emulated_code_references = update_info

        updated_content = stripped_content + "\n\n" + implementation_block

        return updated_content
