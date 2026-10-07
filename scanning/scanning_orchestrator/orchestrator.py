from typing import List

from scanning.scanning_orchestrator.components.scanning_executer import ScanningExecuter


from node_spec.structure_schema import FlowSpec, NodeSpec


class ScanningOrchestrator:
    """
    Scanning Orchestrator component is resposible for orchestrating the whole scanning service flow including:
    1. Filtering relevant probes using SmartFilter component
    2. Executing the scanning process using ScanningExecuter component
    """

    def __init__(self):
        self.scanning_executer = None

    def execute_scanning(
        self,
        user_id: str,
        system_name: str,
        full_analysis_node_spec: NodeSpec,
        selected_flows: List[FlowSpec],
        guidance_spec: dict,
        scan_id: str,
        env_file_path: str,
        scan_results_dir: str | None = None,
        apply_defenses: bool = False,
        scan_timeout_sec: int | None = None,
        flows_per_probe: int = 3,
    ) -> dict:
        """
        Executes the scanning process using the ScanningExecuter component.

        Args:
            user_id (str): the user identifier
            system_name (str): the system identifier
            full_analysis_node_spec (NodeSpec): the full analysis node specification object
            guidance_spec (dict): The selected probe configuration.
            scan_id (str): scan uuid4 identifier
            env_file_path (str): the path to the environment variables file to be used during scanning execution
            scan_timeout_sec (int | None): Optional timeout budget in seconds for each scan run.
            flows_per_probe (int): Maximum number of relevant flows to execute per probe.
        Returns:
            dict: The results of the scanning executions and evaluations with the user_id, system_name, and scan_id included.
        """

        self.scanning_executer = ScanningExecuter(
            nodespec=full_analysis_node_spec,
            guidance_spec=guidance_spec,
            env_file_path=env_file_path,
            image_tag=f"{user_id}_{system_name.lower()}:latest",
            user_id=user_id,
            system_name=system_name,
            scan_timeout_sec=scan_timeout_sec,
        )
        return self.scanning_executer.execute_scanning_process(
            user_id=user_id,
            system_name=system_name,
            scan_id=scan_id,
            full_analysis_node_spec=full_analysis_node_spec,
            flows=selected_flows,
            scan_results_dir=scan_results_dir,
            apply_defenses=apply_defenses,
            flows_per_probe=flows_per_probe,
        )
