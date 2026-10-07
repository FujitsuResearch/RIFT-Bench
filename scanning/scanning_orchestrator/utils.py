import importlib
from typing import List, Tuple, Dict
from node_spec.structure_schema import NodeSpec
from scanning.evaluators.base_evaluator import BaseEvaluator
from scanning.probes.base_probe import BaseProbe


def instantiate_probes(
    guidance_spec: dict,
    nodespec: NodeSpec
) -> List[BaseProbe]:
    """
    Instantiates probe instances based on the provided guidance specifications.
    Args:
        guidance_spec (dict): A dictionary containing guidance specifications for each probe.
    Returns:
        List[Baseprobe]
    """

    probe_instances = []
    for probe_submodule_path, guidance in guidance_spec.items():
        # probe name is the name that was fetched from the DB. The names are saved as submodule.class_name e.g., base_probe.Baseprobe
        probe_submodule_name, probe_class_name = probe_submodule_path.rsplit(
            ".", 1
        )
        probe_module = importlib.import_module(
            f"scanning.probes.{probe_submodule_name}"
        )
        probe_class = getattr(probe_module, probe_class_name)
        probe_instance = probe_class()
        probe_instances.append(probe_instance)

    return probe_instances


def instantiate_probe_evaluators(probe_instance) -> Dict[str, List[BaseEvaluator]]:
    """
    Instantiates evaluator instances for a given probe instance.
    Args:
        probe_instance: An instance of a probe.
    Returns:
        dict: A dictionary mapping probe names to their evaluator instances."""
    probe_name = probe_instance._get_name()
    evaluator_instances = {probe_name: []}
    for evaluator_full_name in probe_instance.probs_evaluators:
        evaluator_submodule_name, evaluator_class_name = evaluator_full_name.rsplit(
            ".", 1
        )
        module = importlib.import_module(
            f"scanning.evaluators.{evaluator_submodule_name}"
        )
        evaluator_class = getattr(module, evaluator_class_name)
        evaluator_instance = evaluator_class()
        evaluator_instances[probe_name].append(
            evaluator_instance
        )  # {probe_name: [evaluator_instances]}
    return evaluator_instances


def prepare_probes_and_evaluators(
    guidance_spec: dict,
    nodespec: NodeSpec
) -> Tuple[List[BaseProbe], Dict[str, List[BaseEvaluator]]]:
    """
    Prepares probe and evaluator instances based on the provided guidance specifications.
    Args:
        guidance_spec (dict): A dictionary containing guidance specifications for each probe.
    Returns:
        tuple: A tuple containing a list of probe instances and a dictionary of evaluator instances.
    """
    probes = instantiate_probes(guidance_spec, nodespec)
    evaluators = {}
    for probe in probes:
        evaluators.update(
            instantiate_probe_evaluators(probe)
        )  # {probe_name: [evaluator_instances]}
    return probes, evaluators
