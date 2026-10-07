
from scanning.defenses.base_defense import BaseDefense
from node_spec.structure_schema import NodeSpec

class PromptSandwiching(BaseDefense):

    @staticmethod
    def add_user_instruction(old_return: str, user_instruction: str) -> str:
        return (
            f"return str({old_return}) + "
            f"{f' The user instruction is: {user_instruction}'!r}"
        )

    def apply_defense_tool(self, tool_node: NodeSpec, user_instruction: str = ''):
        def replacement_builder(old_return: str) -> str:
            return self.add_user_instruction(old_return, user_instruction)

        try:
            self.rewrite_first_return_statement(tool_node, replacement_builder)
        except ValueError:
            return
