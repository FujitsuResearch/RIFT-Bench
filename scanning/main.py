import argparse
import logging
import sys
import uuid
from typing import Dict, List

from scanning.pipeline import run_scanning_pipeline
from scanning.scanning_orchestrator.components.smart_filter import SmartFilter
from node_spec.structure_schema import NodeSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the scanning pipeline on a given NodeSpec."
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
        help="The user ID for the scanning process.",
    )
    parser.add_argument(
        "--scan_timeout_sec",
        type=int,
        default=None,
        help="Maximum number of seconds allowed for each sandbox scan run.",
    )

    parser.add_argument(
        "--flows_per_probe",
        type=int,
        default=3,
        help="Maximum number of relevant flows to execute for each probe.",
    )

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _probe_submodule_path(probe_class) -> str:
    module_name = probe_class.__module__
    if module_name.startswith("scanning.probes."):
        module_name = module_name.removeprefix("scanning.probes.")
    return f"{module_name}.{probe_class.__name__}"


def get_all_probes() -> List[Dict[str, object]]:
    """
    Return metadata for every concrete probe available under scanning.probes.

    Concrete probes exclude top-level template classes in probe families where
    runnable probes are defined in dedicated subpackages.
    """
    excluded_top_level_template_packages = {
        "tool_output_injection",
        "tool_implementation_injection",
    }

    concrete_probe_classes = []
    for probe_class in SmartFilter.discover_probe_classes():
        module_name = probe_class.__module__
        if module_name.startswith("scanning.probes."):
            module_name = module_name.removeprefix("scanning.probes.")

        package_parts = module_name.split(".")
        is_top_level_template = (
            len(package_parts) == 2
            and package_parts[0] in excluded_top_level_template_packages
        )
        if is_top_level_template:
            continue
        concrete_probe_classes.append(probe_class)

    return [
        SmartFilter.normalize_probe_class(probe_class)
        for probe_class in concrete_probe_classes
    ]


def main(full_analysis_node_spec: NodeSpec, args):
    setup_logging()
    scan_results_dir = getattr(args, "scan_results_dir", None)
    apply_defenses = bool(getattr(args, "apply_defenses", False))
    scan_timeout_sec = getattr(args, "scan_timeout_sec", None)
    flows_per_probe = getattr(args, "flows_per_probe", 3)
    run_scanning_pipeline(
        user_id=args.user_id,
        node_spec=full_analysis_node_spec,
        scan_id=str(uuid.uuid4()),
        probes_list=get_all_probes()[20:],
        env_file_path=args.env_file_path,
        scan_results_dir=scan_results_dir,
        apply_defenses=apply_defenses,
        scan_timeout_sec=scan_timeout_sec,
        flows_per_probe=flows_per_probe,
    )