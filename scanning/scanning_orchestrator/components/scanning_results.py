import hashlib
import json

from node_spec.structure_schema import FlowSpec, NodeSpec
from typing import Any, Optional


class ScanningResults:

    def __init__(
        self,
        user_id: str,
        system_name: str,
        scan_id: str,
        probe_results: dict,
    ):
        """
        Scanning Results class to hold the results of a scanning process including probe results, malicious node specifications, and evaluator results.
        This class is used throughout the scanning executor component.
        Args:
            user_id (str): The unique identifier of the user.
            system_name (str): The unique identifier of the system.
            scan_id (str): The unique identifier of the scan.
            probe_name (str): The name of the probe used.
            scanning_status (int): The status of the scanning process (default is 0).
            probe_results (Optional[dict]): The results obtained from the probe execution.
            malicious_node_spec (Optional[dict]): The specifications of the malicious twin node identified during scanning
        """

        # Primary key of the results table
        self.user_id: str = user_id
        self.system_name: str = system_name
        self.scan_id: str = scan_id
        # These come from probe_results dict
        self.probe_name: str = probe_results["probe_name"]
        self.scanning_status: int = probe_results["status"]
        self.malicious_node_spec: Optional[NodeSpec] = probe_results["malicious_twin"]
        # a dictionary mapping execution command to its flow (parsed trace)
        self.traces_of_execution: Optional[dict[str, FlowSpec]] = self.flow_to_flow_spec(probe_results.get("traces_of_execution", {}))
        # Will be populated while executing evaluators
        self.evaluator_results: dict = {}
        self.evaluator_name: str = ""

    def flow_to_flow_spec(self, trace_of_execution: dict[str, dict]) -> dict[str, FlowSpec]:
        """Converts the traces of execution from their JSON-serializable format back to FlowSpec objects.
        Returns:
            dict[str, FlowSpec]: A dictionary mapping execution commands to their corresponding FlowSpec objects.
        """
        return {cmd: FlowSpec.model_validate(flow) if isinstance(flow, dict) else flow for cmd, flow in trace_of_execution.items()}
 

    def populate_evaluator_results(
        self,
        evaluator_name: str,
        evaluator_results: dict,
    ) -> None:
        """
        Saves the results of an evaluator into the scanning results.
        Args:
            evaluator_name (str): The name of the evaluator.
            evaluator_results (dict): The results obtained from the evaluator execution.
        Returns:
            None
        """
        self.evaluator_name = evaluator_name
        self.evaluator_results = evaluator_results

    def to_dict(self) -> dict:
        """
        Converts the ScanningResults instance to a dictionary representation.
        Returns:
            dict: A dictionary containing the scanning results data.
        """
        return {
            "user_id": self.user_id,
            "system_name": self.system_name,
            "scan_id": self.scan_id,
            "probe_name": self.probe_name,
            "evaluator_name": self.evaluator_name,
            "scanning_status": self.scanning_status,
            "malicious_node_spec": self.malicious_node_spec,
            "traces_of_execution": self.traces_of_execution,
            "evaluator_results": self.evaluator_results,
        }

    def to_db_dict(self) -> dict:
        """
        Converts the ScanningResults instance to a dictionary matching the
        results table column names.
        Returns:
            dict: A dictionary containing the scan result payload for the DB.
        """
        return {
            "user_id": self.user_id,
            "system_name": self.system_name,
            "scan_id": self.scan_id,
            "scanner_name": self.probe_name,
            "evaluator_name": self.evaluator_name,
            "scanning_status": self.scanning_status,
            "malicious_node_spec": self.malicious_node_spec,
            "traces_of_execution": self.traces_of_execution,
            "evaluator_results": self.evaluator_results,
        }

    @staticmethod
    def from_dict(
        user_id: str,
        scan_id: int,
        system_name: str,
        probe_name: str = "",
        data: dict = None,
        scanner_name: str = "",
    ) -> "ScanningResults":
        """
        Populate an instance from a dictionary representation.
        Args:
            user_id (str): The unique identifier of the user.
            system_name (str): The unique identifier of the system.
            scan_id (str): The unique identifier of the scan.
            probe_name (str): The name of the probe used.
            data (dict): A dictionary containing the scanning results data.
            scanner_name (str): The name of the scanner used, used as a fallback if probe_name is not provided.
        Returns:
            ScanningResults: An instance of ScanningResults populated with the provided data.
        """
        data = data or {}
        probe_name = probe_name or scanner_name
        malicious_node_spec = data.get(
            "malicious_node_spec", data.get("malicious_twin")
        )
        if isinstance(malicious_node_spec, str):
            malicious_node_spec = json.loads(malicious_node_spec)
        if isinstance(malicious_node_spec, dict):
            malicious_node_spec = NodeSpec.from_json(data=malicious_node_spec)

        traces_of_execution = data.get("traces_of_execution")
        if isinstance(traces_of_execution, str):
            traces_of_execution = json.loads(traces_of_execution)
        evaluator_results = data.get("evaluator_results", {})
        if isinstance(evaluator_results, str):
            evaluator_results = json.loads(evaluator_results)


        scanning_results = ScanningResults(
            user_id=user_id,
            system_name=system_name,
            scan_id=scan_id,
            probe_results={
                "probe_name": probe_name,
                "status": data["scanning_status"],
                "malicious_twin": malicious_node_spec,
                "traces_of_execution": traces_of_execution,
            },
        )
        scanning_results.populate_evaluator_results(
            evaluator_name=data.get("evaluator_name", ""),
            evaluator_results=evaluator_results or {},
        )
        return scanning_results
