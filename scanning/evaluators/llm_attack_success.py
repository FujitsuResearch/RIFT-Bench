from scanning.evaluators.base_evaluator import BaseEvaluator
from scanning.scanning_orchestrator.components.scanning_results import ScanningResults
from scanning.evaluators.evaluation_util import *


class LLMAttackSuccess(BaseEvaluator):
    """Checks whether the overall attack goal succeeded from the LLM perspective."""

    def __init__(self, category = "Attack Success"):
        super().__init__(category)

    def evaluate_results(
        self,
        scanning_results: ScanningResults,
    ):
        """Return the attack-success signal inferred from ``attack_goal`` and the trace."""
        malicious_node_spec = scanning_results.malicious_node_spec
        attack_metadata = malicious_node_spec.metadata['instance_info']
        attack_goal = attack_metadata['attack_goal']
        execution_cmd = self._get_execution_cmd(scanning_results, attack_metadata)

        trace = self._get_trace(scanning_results, execution_cmd)

        if trace is None:
            return {self.category: None}

        attack_success_llm = get_attack_success_llm(attack_goal, trace)

        return {self.category: attack_success_llm}
