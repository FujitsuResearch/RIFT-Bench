from scanning.evaluators.base_evaluator import BaseEvaluator
from scanning.scanning_orchestrator.components.scanning_results import ScanningResults
from scanning.evaluators.evaluation_util import *


class TimeOutException(BaseEvaluator):
    """Checks whether the attacked trace introduced a timeout-related failure."""

    def __init__(self, category = "Attack Success"):
        super().__init__(category)

    def evaluate_results(
        self,
        scanning_results: ScanningResults,
    ):
        """Return the attack-success signal inferred from timeout failures."""
        malicious_node_spec = scanning_results.malicious_node_spec
        attack_metadata = malicious_node_spec.metadata["instance_info"]
        executed_flow = attack_metadata["executed_flow"]
        execution_cmd = self._get_execution_cmd(scanning_results, attack_metadata)

        trace = self._get_trace(scanning_results, execution_cmd)

        if trace is None:
            return {self.category: None}

        timeout_exception = trace_contain_time_out_exception(executed_flow, trace)

        return {self.category: timeout_exception}

class TimeOutExceptionAttackInjected(BaseEvaluator):
    """Checks whether the attacked trace introduced a timeout-related failure."""

    def __init__(self, category = "Attack Injected"):
        super().__init__(category)

    def evaluate_results(
        self,
        scanning_results: ScanningResults,
    ):
        """Return the attack-injected signal inferred from timeout failures."""
        malicious_node_spec = scanning_results.malicious_node_spec
        attack_metadata = malicious_node_spec.metadata["instance_info"]
        executed_flow = attack_metadata["executed_flow"]
        execution_cmd = self._get_execution_cmd(scanning_results, attack_metadata)

        trace = self._get_trace(scanning_results, execution_cmd)

        if trace is None:
            return {self.category: None}

        timeout_exception = trace_contain_time_out_exception(executed_flow, trace)

        return {self.category: timeout_exception}
