
from scanning.defenses.base_defense import BaseDefense
from node_spec.structure_schema import NodeSpec

class PromptInjectionDetection(BaseDefense):

    @staticmethod
    def get_detector_snippet(old_return: str) -> str:
        return f"""import torch
from transformers import pipeline
pipeline = pipeline(\"text-classification\", model=\"protectai/deberta-v3-base-prompt-injection-v2\", device=torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\"))
result: dict = pipeline(str({old_return}))[0] 
safety_score: float = result[\"score\"] if result[\"label\"] == \"SAFE\" else 1 - result[\"score\"]
is_injection: bool = safety_score < 0.5

if is_injection:
    return \'Blocked Tool Output\'
else:
    return {old_return}
"""

    def apply_defense_tool(self, tool_node: NodeSpec, user_instruction: str = ''):
        def replacement_builder(old_return: str) -> str:
            return self.get_detector_snippet(old_return)

        try:
            self.rewrite_first_return_statement(tool_node, replacement_builder)
        except ValueError:
            return
