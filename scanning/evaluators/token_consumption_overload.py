from scanning.evaluators.base_evaluator import BaseEvaluator
from scanning.scanning_orchestrator.components.scanning_results import ScanningResults
from scanning.evaluators.evaluation_util import *


class TokenConsumptionOverload(BaseEvaluator):
    """Checks whether the attack led to higher token consumption."""

    def __init__(self, category = "Attack Success"):
        super().__init__(category)

    def evaluate_results(
        self,
        scanning_results: ScanningResults,
    ):
        """Return the attack-success signal inferred from the attacked trace compare to the benign trace."""
        malicious_node_spec = scanning_results.malicious_node_spec
        attack_metadata = malicious_node_spec.metadata['instance_info']
        executed_flow = attack_metadata['executed_flow']
        execution_cmd = self._get_execution_cmd(scanning_results, attack_metadata)

        trace = self._get_trace(scanning_results, execution_cmd)

        if trace is None:
            return {self.category: None}

        token_consumption_overload = get_token_consumption_overload(executed_flow, trace)

        return {self.category: token_consumption_overload}
