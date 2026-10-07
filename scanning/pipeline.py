import logging
from typing import List
from pathlib import Path
from node_spec.structure_schema import NodeSpec, FlowSpec

from scanning.scanning_orchestrator.components.smart_filter import SmartFilter as SmartFilter
from scanning.scanning_orchestrator.orchestrator import ScanningOrchestrator

def run_filtering_pipeline(
    node_spec: NodeSpec, selected_flows: List[FlowSpec] | None = None
):
    logger = logging.getLogger(__name__)
    logger.info(
        "[Scanning] Filtering relevant probes for the provided agentic system"
    )

    smart_filter = SmartFilter()
    # Filter relevant probes based on the NodeSpec and system description.
    probes_list = smart_filter.run_smart_filtering(
        full_analysis_node_spec=node_spec,
        selected_flows=selected_flows or [],
    )
    logger.info(
        "[Scanning] Relevant probes filtered successfully. Total probes: %d",
        len(probes_list),
    )

    return probes_list


def run_scanning_pipeline(
    user_id: str,
    node_spec: NodeSpec,
    scan_id: str,
    probes_list: List[dict],
    env_file_path: str,
    scan_results_dir: str | None = None,
    apply_defenses: bool = False,
    scan_timeout_sec: int | None = None,
    flows_per_probe: int = 3,
    selected_flows: List[FlowSpec] | None = None,
    user_db_paths: List[str] | None = None,
    db_node_path_mappings: List[dict[str, str]] | None = None,
    demo_mode: bool = False,
    local_testing: bool = True,
) -> dict[str, str]:
    """
    Runs the complete scanning pipeline including:
    1. Filtering relevant probes using ScanningOrchestrator's SmartFilter component.
    2. Preparing the selected probe configuration.
    3. Executing the scanning process using ScanningOrchestrator's ScanningExecuter component.
    Args:
        user_id (str): The user ID for the scanning process.
        node_spec (NodeSpec): The complete analysis NodeSpec of the system.
        probes_list (List[dict]): pre-filtered probes list.
        env_file_path (str): the path to the environment variables file to be used during scanning execution
        scan_timeout_sec (int | None): Optional timeout budget in seconds for each scan run.
        flows_per_probe (int): Maximum number of relevant flows to execute per probe.
        selected_flows (List[FlowSpec] | None): provided by the user - flows to scan
        scan_id (int | None): Optional scan id. When None, a new scan id is allocated.
        user_db_paths (List[str]) | None: Optional list of user database paths to be used during scanning execution.
        db_node_path_mappings (List[dict[str,str]]) | None: Optional list of mappings between NodeSpec node names and database paths, to be used during scanning execution.
        demo_mode (bool): Whether to run the scanning pipeline in demo mode, which may include additional logging and relaxed constraints for testing purposes.
        local_testing (bool): Whether to run the scanning pipeline in local testing mode. When True, .env file of the user does not get deleted after scanning execution.
    Returns:
        dict[str, str]: Scan identifiers for the completed file-based scan.
    """
    logger = logging.getLogger(__name__)
    logger.info("[Scanning] Starting scanning pipeline")

    logger.info("[Scanning] Initializing Scanning Module components")
    # Initialize scanning orchestration.
    orchestrator = ScanningOrchestrator()

    logger.info("[Scanning] Preparing selected probes for execution")
    guidance_spec = {
        probe["probe_submodule_path"]: {"instructions": "", "node_id": ""}
        for probe in probes_list
        if probe.get("probe_submodule_path")
    }

    logger.info("[Scanning] Executing the scanning process")
    # Execute Scanning Process
    scanning_process_metadata: dict = orchestrator.execute_scanning(
        user_id=user_id,
        system_name=node_spec.name,
        full_analysis_node_spec=node_spec,
        selected_flows=selected_flows,
        guidance_spec=guidance_spec,
        scan_id=scan_id,
        env_file_path=env_file_path,
        scan_results_dir=scan_results_dir,
        apply_defenses=apply_defenses,
        scan_timeout_sec=scan_timeout_sec,
        flows_per_probe=flows_per_probe,
    )

    # Extract scan identifiers for the runner result.
    user_id, system_name, scan_id = (
        scanning_process_metadata["user_id"],
        scanning_process_metadata["system_name"],
        scanning_process_metadata["scan_id"],
    )

    print(f"Scan Complete. Scan ID: {scan_id}")
    if not local_testing:
        Path(env_file_path).unlink(missing_ok=True)
    return {
        "user_id": user_id,
        "system_name": system_name,
        "scan_id": scan_id,
    }
