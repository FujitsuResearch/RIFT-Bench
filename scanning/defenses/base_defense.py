from scanning.probes.base_probe import BaseProbe
from node_spec.structure_schema import NodeSpec
import shlex
from scanning.probes.utils.base_util import get_main_user_task_argument_name

class BaseDefense(BaseProbe):

    def __init__(self):
        pass

    def apply_defense_tool(self, tool_node: NodeSpec, user_instruction: str = ''):
        return {}

    def apply_defense_agent(self, agent_node: NodeSpec):
        return {}
    
    @staticmethod
    def extract_user_instruction(full_node_spec: NodeSpec, execution_cmd) -> str:
        main_argument_name = get_main_user_task_argument_name(full_node_spec).lstrip("-")
        try:
            tokens = shlex.split(execution_cmd)
        except ValueError:
            tokens = execution_cmd.split()

        for index, token in enumerate(tokens):
            if not token.startswith("-"):
                continue

            argument_token, has_equal, value = token.partition("=")
            argument_name = argument_token.lstrip("-")
            if argument_name != main_argument_name:
                continue

            if has_equal:
                return value
            if index + 1 < len(tokens):
                return tokens[index + 1]
            return ""

        return ""


    def apply_defense_tools(self,
        full_node_spec: NodeSpec,
        execution_cmd):
        affected_nodes: dict[str, list[str]] = {}
        for tool in full_node_spec.list_tools():
            tool_id = tool["ids"][0]
            tool_node = full_node_spec.get_node(node_id=tool_id)
            if getattr(tool_node, "code_references", None):
                code_references_before_injection = self.snapshot_code_reference_snippets(
                    tool_node)
                self.apply_defense_tool(tool_node, self.extract_user_instruction(full_node_spec, execution_cmd))

                changed_kinds = self.changed_code_reference_kinds(
                    owner_node=tool_node,
                    before_snapshot=code_references_before_injection,
                )
                if not tool_node.id:
                    continue

                node_kinds = affected_nodes.setdefault(tool_node.id, [])
                for kind in changed_kinds:
                    if kind not in node_kinds:
                        node_kinds.append(kind)
        return affected_nodes

    def apply_defense_agents(self,
        full_node_spec: NodeSpec):
        affected_nodes: dict[str, list[str]] = {}
        for agent in full_node_spec.list_agents():
            agent_id = agent["ids"][0]
            agent_node = full_node_spec.get_node(node_id=agent_id)
            if getattr(agent_node, "code_references", None):
                code_references_before_injection = self.snapshot_code_reference_snippets(
                    agent_node)
                self.apply_defense_agent(agent_node)

                changed_kinds = self.changed_code_reference_kinds(
                    owner_node=agent_node,
                    before_snapshot=code_references_before_injection,
                )
                if not agent_node.id:
                    continue

                node_kinds = affected_nodes.setdefault(agent_node.id, [])
                for kind in changed_kinds:
                    if kind not in node_kinds:
                        node_kinds.append(kind)
        return affected_nodes

    def update_node_spec(
        self,
        full_node_spec: NodeSpec,
        affected_nodes: list[dict[str, list[str]]] | dict[str, list[str]],
        execution_cmds):

        defense_affected_nodes = self.apply_defense_tools(full_node_spec, execution_cmds[0])
        defense_affected_nodes = self.combine_affected_nodes(
            defense_affected_nodes,
            self.apply_defense_agents(full_node_spec),
        )
        affected_nodes = self.combine_affected_nodes(defense_affected_nodes, affected_nodes)

        return full_node_spec, affected_nodes

    @staticmethod
    def combine_affected_nodes(
        defense_affected_nodes: list[dict[str, list[str]]] | dict[str, list[str]],
        affected_nodes: list[dict[str, list[str]]] | dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged_nodes: dict[str, list[str]] = {}

        def merge_entries(entries: list[dict[str, list[str]]] | dict[str, list[str]]):
            if isinstance(entries, dict):
                iterable = [entries]
            else:
                iterable = entries or []

            for entry in iterable:
                if not isinstance(entry, dict):
                    continue
                for node_id, kinds in entry.items():
                    if not isinstance(node_id, str):
                        continue
                    node_kinds = merged_nodes.setdefault(node_id, [])
                    for kind in kinds or []:
                        if not kind:
                            continue
                        kind_text = str(kind)
                        if kind_text not in node_kinds:
                            node_kinds.append(kind_text)

        merge_entries(defense_affected_nodes)
        merge_entries(affected_nodes)

        return merged_nodes
