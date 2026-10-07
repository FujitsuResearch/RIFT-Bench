import csv
import copy
from datetime import datetime, timezone
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Union
from scanning.probes.utils.base_util import build_input_arguments, build_execution_cmds
from scanning.defenses.base_defense import BaseDefense
from scanning.abstraction_executer.AbstractionExecuter import AbstractionExecuter
from node_spec.structure_schema import FlowSpec, NodeSpec
from scanning.probes.base_probe import BaseProbe
from scanning.evaluators.base_evaluator import BaseEvaluator
from scanning.evaluators.evaluator_runtime import collect_evaluator_runtime_metrics
from scanning.scanning_orchestrator.components.scanning_results import ScanningResults
from ..utils import prepare_probes_and_evaluators
from trace_parser import parse_single_trace
import subprocess

class ScanningExecuter:
    """
    Scanning Executer is responsible for executing the scanning process using the provided probes and evaluators on the full analysis NodeSpec.
    """
    EVALUATOR_RESULTS_CSV_HEADER = [
        "probe_name",
        "attack_suite",
        "attack_surface",
        "attacker_intent",
        "flow_number",
        "evaluator_name",
        "category",
        "value",
        "failure_threshold",
        "time",
        "number_of_llm_calls",
        "total_tokens_used",
    ]
    FLOW_SKIP_MARKER_FILENAME = "skip_flow_due_to_exception.json"
    MISSING_CODE_REFERENCES_SKIP_REASON = "missing_code_references"

    def __init__(
        self,
        nodespec: NodeSpec,
        guidance_spec: dict | None = None,
        image_tag: str = "",
        env_file_path: str = ".env",
        user_id: str = "",
        system_name: str = "",
        scan_timeout_sec: int | None = None,
    ):
        """
        Initializes the ScanningExecuter with the relevant probes and evaluators. The guidance specification on this part is only used to filter out the relevant probes.
        Args:
            guidance_spec (dict, optional): The guidance specification for filtering probes.
            image_tag (str): The image tag for the abstraction executer.
            scan_timeout_sec (int | None, optional): Default timeout budget in seconds for each scan run.
        """
        self.guidance_spec = guidance_spec or {}
        self.probes, self.evaluators = prepare_probes_and_evaluators(guidance_spec, nodespec)
        self.abstraction_executer = AbstractionExecuter(
            root_directory_path="/sandbox",
            image_tag=image_tag,
            env_file_path=env_file_path,
            repo_name=f"{user_id}_{system_name}",
            scan_timeout_sec=scan_timeout_sec,
        )
        self.empty_trace_events: List[Dict[str, Any]] = [] # temporary, naive logging of failed scans

    def _record_empty_trace_paths(
        self,
        probe_instance: BaseProbe,
        execution_cmds: List[str],
        cmd_trace_path_dict: dict | None,
    ) -> None:
        raw_trace_paths = cmd_trace_path_dict or {}
        ordered_trace_paths = (
            list(raw_trace_paths.values()) if isinstance(raw_trace_paths, dict) else []
        )

        empty_commands = []
        for step_index, command in enumerate(execution_cmds or []):
            raw_trace_path = (
                ordered_trace_paths[step_index]
                if step_index < len(ordered_trace_paths)
                else ""
            )
            if str(raw_trace_path or "").strip():
                continue
            empty_commands.append(f"[step {step_index}] {command}")

        if not empty_commands:
            return

        self.empty_trace_events.append(
            {
                "probe_name": probe_instance._get_name(),
                "probe_display_name": probe_instance.name,
                "empty_commands": empty_commands,
                "raw_trace_paths": raw_trace_paths,
            }
        )


    def _print_empty_trace_summary(self) -> None:
        if not self.empty_trace_events:
            print("[Scanning] Empty trace summary: no empty trace paths")
            return

        print(
            f"[Scanning] Empty trace summary: "
            f"{len(self.empty_trace_events)} probe(s) had at least one empty trace path"
        )

        for event in self.empty_trace_events:
            print(
                "[Scanning] EMPTY TRACE EVENT | "
                f"probe={event['probe_name']} | "
                f"display_name={event['probe_display_name']} | "
                f"empty_commands={event['empty_commands']} | "
                f"raw_trace_paths={event['raw_trace_paths']}"
            )

    @staticmethod
    def _probe_identity(probe: BaseProbe) -> dict[str, str]:
        probe_name = ""
        if hasattr(probe, "_get_name"):
            try:
                probe_name = str(probe._get_name() or "").strip()
            except Exception:
                probe_name = ""

        probe_display_name = str(getattr(probe, "name", "") or "").strip()
        if not probe_name:
            probe_name = probe_display_name or probe.__class__.__name__
        if not probe_display_name:
            probe_display_name = probe_name

        return {
            "probe_name": probe_name,
            "probe_display_name": probe_display_name,
        }

    @staticmethod
    def _save_probes_not_run_summary(
        scan_results_dir: str | None,
        scan_id: str,
        user_id: str,
        system_name: str,
        no_relevant_flow_probes: list[dict[str, str]],
        missing_code_reference_probes: list[dict[str, Any]],
        probe_scan_exceptions: list[dict[str, Any]],
    ) -> None:
        if not scan_results_dir:
            return

        os.makedirs(scan_results_dir, exist_ok=True)
        summary_path = os.path.join(scan_results_dir, "probes_not_run_summary.json")
        payload = {
            "scan_id": scan_id,
            "user_id": user_id,
            "system_name": system_name,
            "counts": {
                "no_relevant_flows": len(no_relevant_flow_probes),
                "missing_code_references": len(missing_code_reference_probes),
                "exception_in_complete_probe_scan": len(probe_scan_exceptions),
            },
            "probes_not_run": {
                "no_relevant_flows": no_relevant_flow_probes,
                "missing_code_references": missing_code_reference_probes,
                "exception_in_complete_probe_scan": probe_scan_exceptions,
            },
        }
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            json.dump(payload, summary_file, ensure_ascii=False, indent=2)

    @staticmethod
    def _is_missing_code_reference_error(error: Exception | str) -> bool:
        error_text = str(error or "").strip().lower()
        if not error_text:
            return False
        if "code_references" in error_text:
            return "does not define" in error_text
        if "code references" in error_text:
            return "does not define" in error_text or "has no" in error_text
        return False

    @classmethod
    def _skip_reason_for_error(cls, error: Exception | str) -> str:
        if cls._is_missing_code_reference_error(error):
            return cls.MISSING_CODE_REFERENCES_SKIP_REASON
        return "exception_in_complete_probe_scan"

    @classmethod
    def _build_probe_scan_skip_entry(
        cls,
        probe_identity: dict[str, str],
        *,
        flow_num: int,
        error: Exception | str,
        synthetic_results_applied: bool = False,
    ) -> dict[str, Any]:
        return {
            **probe_identity,
            "flow_num": flow_num,
            "error": str(error),
            "skip_reason": cls._skip_reason_for_error(error),
            "synthetic_results_applied": synthetic_results_applied,
        }

    @staticmethod
    def _safe_dir_name(value: Any) -> str:
        text = str(value or "unknown").strip()
        if not text:
            return "unknown"
        text = text.replace(" ", "_")
        text = text.replace(os.sep, "_")
        if os.altsep:
            text = text.replace(os.altsep, "_")
        return text

    @classmethod
    def _probe_results_dir(cls, scan_results_dir: str, probe_name: str) -> str:
        return os.path.join(scan_results_dir, cls._safe_dir_name(probe_name))

    @classmethod
    def _flow_results_dir(
        cls,
        scan_results_dir: str,
        probe_name: str,
        flow_num: int,
    ) -> str:
        return os.path.join(
            cls._probe_results_dir(scan_results_dir, probe_name),
            f"flow_{flow_num}",
        )

    @classmethod
    def _flow_skip_marker_path(
        cls,
        scan_results_dir: str,
        probe_name: str,
        flow_num: int,
    ) -> str:
        return os.path.join(
            cls._flow_results_dir(scan_results_dir, probe_name, flow_num),
            cls.FLOW_SKIP_MARKER_FILENAME,
        )

    @classmethod
    def _is_flow_marked_to_skip(
        cls,
        scan_results_dir: str,
        probe_name: str,
        flow_num: int,
    ) -> bool:
        marker_path = cls._flow_skip_marker_path(
            scan_results_dir=scan_results_dir,
            probe_name=probe_name,
            flow_num=flow_num,
        )
        return os.path.exists(marker_path)

    @classmethod
    def _load_flow_skip_marker(
        cls,
        scan_results_dir: str | None,
        probe_name: str,
        flow_num: int,
    ) -> dict[str, Any] | None:
        if not scan_results_dir:
            return None

        marker_path = cls._flow_skip_marker_path(
            scan_results_dir=scan_results_dir,
            probe_name=probe_name,
            flow_num=flow_num,
        )
        if not os.path.exists(marker_path):
            return None

        try:
            with open(marker_path, "r", encoding="utf-8") as marker_file:
                payload = json.load(marker_file) or {}
        except (OSError, json.JSONDecodeError):
            return {}

        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _flow_skip_counts_as_failure(
        cls,
        marker_payload: dict[str, Any] | None,
    ) -> bool:
        if not marker_payload:
            return True

        if bool(marker_payload.get("synthetic_results_applied")):
            return False

        skip_reason = str(marker_payload.get("skip_reason") or "").strip()
        if skip_reason:
            return skip_reason != cls.MISSING_CODE_REFERENCES_SKIP_REASON

        return not cls._is_missing_code_reference_error(marker_payload.get("error"))

    @classmethod
    def _remove_flow_probe_scan_metadata(
        cls,
        scan_results_dir: str | None,
        probe_name: str,
        flow_num: int,
    ) -> None:
        if not scan_results_dir:
            return

        flow_dir = cls._flow_results_dir(
            scan_results_dir=scan_results_dir,
            probe_name=probe_name,
            flow_num=flow_num,
        )
        if not os.path.isdir(flow_dir):
            return

        for current_root, _, filenames in os.walk(flow_dir):
            if "complete_probe_scan_metadata.json" not in filenames:
                continue
            metadata_path = os.path.join(current_root, "complete_probe_scan_metadata.json")
            try:
                os.remove(metadata_path)
            except OSError:
                continue

    @classmethod
    def _mark_flow_for_skip(
        cls,
        scan_results_dir: str | None,
        probe_name: str,
        flow_num: int,
        error: Exception,
        synthetic_results_applied: bool = False,
    ) -> None:
        if not scan_results_dir:
            return

        flow_dir = cls._flow_results_dir(
            scan_results_dir=scan_results_dir,
            probe_name=probe_name,
            flow_num=flow_num,
        )
        os.makedirs(flow_dir, exist_ok=True)

        marker_path = cls._flow_skip_marker_path(
            scan_results_dir=scan_results_dir,
            probe_name=probe_name,
            flow_num=flow_num,
        )
        marker_payload = {
            "probe_name": probe_name,
            "flow_num": flow_num,
            "error": str(error),
            "skip_reason": cls._skip_reason_for_error(error),
            "synthetic_results_applied": synthetic_results_applied,
        }
        with open(marker_path, "w", encoding="utf-8") as marker_file:
            json.dump(marker_payload, marker_file, ensure_ascii=False, indent=2)

    @classmethod
    def _leaf_results_dir(
        cls,
        scan_results_dir: str,
        probe_name: str,
        flow_num: int,
    ) -> str:
        return cls._flow_results_dir(scan_results_dir, probe_name, flow_num)

    @classmethod
    def _defense_results_dir(cls, leaf_dir: str, defense_name: str) -> str:
        return os.path.join(leaf_dir, cls._safe_dir_name(defense_name))

    @staticmethod
    def _load_all_defenses() -> list[BaseDefense]:
        defenses_dir = Path(__file__).resolve().parents[2] / "defenses"
        if not defenses_dir.is_dir():
            print(f"[Scanning] Defenses directory not found at {defenses_dir}")
            return []

        loaded_defenses: list[BaseDefense] = []
        for module_path in sorted(defenses_dir.glob("*.py")):
            module_stem = module_path.stem
            if module_stem.startswith("_") or module_stem == "base_defense":
                continue

            module_name = f"scanning.defenses.{module_stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception as import_error:
                print(
                    f"[Scanning] Failed to import defense module {module_name}: "
                    f"{import_error}"
                )
                continue

            module_defenses: list[BaseDefense] = []
            for _, defense_cls in inspect.getmembers(module, inspect.isclass):
                if defense_cls.__module__ != module.__name__:
                    continue
                if not issubclass(defense_cls, BaseDefense) or defense_cls is BaseDefense:
                    continue
                try:
                    module_defenses.append(defense_cls())
                except Exception as defense_init_error:
                    print(
                        f"[Scanning] Failed to initialize defense "
                        f"{defense_cls.__name__}: {defense_init_error}"
                    )

            loaded_defenses.extend(module_defenses)

        loaded_defenses.sort(key=lambda defense: defense.__class__.__name__)
        return loaded_defenses

    @classmethod
    def _load_enabled_defenses(
        cls,
        *,
        apply_defenses: bool,
    ) -> list[BaseDefense]:
        if not apply_defenses:
            return []
        return cls._load_all_defenses()

    @staticmethod
    def _parsed_trace_path(leaf_dir: str) -> str:
        return os.path.join(leaf_dir, "parsed_trace.json")

    @staticmethod
    def _results_csv_path(leaf_dir: str) -> str:
        csv_path = os.path.join(leaf_dir, "evaluator_results.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=ScanningExecuter.EVALUATOR_RESULTS_CSV_HEADER
                )
                writer.writeheader()
        return csv_path

    @staticmethod
    def _malicious_nodespec_path(leaf_dir: str) -> str:
        return os.path.join(leaf_dir, "malicious_node_spec.json")

    @staticmethod
    def _probe_result_metadata_path(leaf_dir: str) -> str:
        return os.path.join(leaf_dir, "probe_result_metadata.json")

    @staticmethod
    def _augment_probe_instance_info_with_runtime_metrics(
        malicious_node_spec: NodeSpec,
        *,
        duration_seconds: float,
        llm_calls: int,
        total_tokens_used: int,
    ) -> NodeSpec:
        metadata = dict(malicious_node_spec.metadata or {})
        instance_info = dict(metadata.get("instance_info") or {})
        instance_info.update(
            {
                "time": round(duration_seconds, 6),
                "number_of_llm_calls": llm_calls,
                "total_tokens_used": total_tokens_used,
            }
        )
        metadata["instance_info"] = instance_info
        malicious_node_spec.metadata = metadata
        return malicious_node_spec

    def _build_malicious_twin_payload(
        self,
        probe_instance: BaseProbe,
        full_analysis_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: Dict | None = None,
    ) -> tuple[NodeSpec, Any, List[str]]:
        if hasattr(probe_instance, "abstraction_executer"):
            probe_instance.abstraction_executer = self.abstraction_executer

        probe_started_at = time.perf_counter()
        with collect_evaluator_runtime_metrics() as runtime_metrics:
            malicious_twin_payload = probe_instance.malicious_twin_update(
                full_node_spec=full_analysis_node_spec,
                flow=flow,
                guidance=guidance or {},
            )

        malicious_node_spec, affected_nodes, execution_cmds = malicious_twin_payload
        malicious_node_spec = self._augment_probe_instance_info_with_runtime_metrics(
            malicious_node_spec,
            duration_seconds=time.perf_counter() - probe_started_at,
            llm_calls=runtime_metrics.llm_calls,
            total_tokens_used=runtime_metrics.total_tokens_used,
        )
        return malicious_node_spec, affected_nodes, execution_cmds

    @staticmethod
    def _safe_non_negative_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0

        if isinstance(value, int):
            return value if value >= 0 else 0

        if isinstance(value, float):
            if value < 0:
                return 0
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                return 0

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return 0
            try:
                parsed = int(normalized)
            except ValueError:
                return 0
            return parsed if parsed >= 0 else 0

        return 0

    @classmethod
    def _extract_probe_instance_usage(
        cls,
        malicious_node_spec: NodeSpec | None,
    ) -> tuple[float, int, int, int]:
        metadata = dict(getattr(malicious_node_spec, "metadata", None) or {})
        instance_info = dict(metadata.get("instance_info") or {})
        return (
            cls._safe_non_negative_float(instance_info.get("time")),
            cls._safe_non_negative_int(instance_info.get("number_of_llm_calls")),
            cls._safe_non_negative_int(instance_info.get("total_tokens_used")),
            cls._safe_non_negative_int(
                BaseProbe.infer_system_execution_count(instance_info)
            ),
        )

    @classmethod
    def _aggregate_evaluator_usage_from_results_tree(
        cls,
        leaf_dir: str | None,
    ) -> tuple[float, int, int]:
        if not leaf_dir or not os.path.isdir(leaf_dir):
            return 0.0, 0, 0

        total_duration_seconds = 0.0
        total_llm_calls = 0
        total_tokens_used = 0
        for current_root, _, filenames in os.walk(leaf_dir):
            if "evaluator_results.csv" not in filenames:
                continue

            csv_path = os.path.join(current_root, "evaluator_results.csv")
            with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    total_duration_seconds += cls._safe_non_negative_float(
                        row.get("time")
                    )
                    total_llm_calls += cls._safe_non_negative_int(
                        row.get("number_of_llm_calls")
                    )
                    total_tokens_used += cls._safe_non_negative_int(
                        row.get("total_tokens_used")
                    )

        return total_duration_seconds, total_llm_calls, total_tokens_used

    @staticmethod
    def _safe_non_negative_float(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0

        if isinstance(value, (int, float)):
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            return parsed if parsed >= 0 else 0.0

        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return 0.0
            try:
                parsed = float(normalized)
            except ValueError:
                return 0.0
            return parsed if parsed >= 0 else 0.0

        return 0.0

    @staticmethod
    def _complete_probe_scan_metadata_path(leaf_dir: str) -> str:
        return os.path.join(leaf_dir, "complete_probe_scan_metadata.json")

    @staticmethod
    def _utc_now_isoformat() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _save_complete_probe_scan_metadata(
        cls,
        leaf_dir: str | None,
        *,
        probe_name: str,
        flow_num: int,
        guidance: dict | None,
        apply_defenses: bool,
        started_at_utc: str,
        completed_at_utc: str,
        duration_seconds: float,
        probe_creation_duration_seconds: float = 0.0,
        evaluator_duration_seconds: float = 0.0,
        system_execution_count: int = 0,
        status: str,
        probe_creation_number_of_llm_calls: int = 0,
        probe_creation_total_tokens_used: int = 0,
        evaluator_number_of_llm_calls: int = 0,
        evaluator_total_tokens_used: int = 0,
        error: str | None = None,
    ) -> None:
        if not leaf_dir:
            return

        metadata = {
            "probe_name": probe_name,
            "flow_num": flow_num,
            "guidance": copy.deepcopy(guidance or {}),
            "apply_defenses": apply_defenses,
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "duration_seconds": round(duration_seconds, 6),
            "probe_creation_duration_seconds": round(
                probe_creation_duration_seconds, 6
            ),
            "evaluator_duration_seconds": round(evaluator_duration_seconds, 6),
            "system_execution_count": system_execution_count,
            "status": status,
            "probe_creation_number_of_llm_calls": probe_creation_number_of_llm_calls,
            "probe_creation_total_tokens_used": probe_creation_total_tokens_used,
            "evaluator_number_of_llm_calls": evaluator_number_of_llm_calls,
            "evaluator_total_tokens_used": evaluator_total_tokens_used,
            "aggregated_number_of_llm_calls": (
                probe_creation_number_of_llm_calls + evaluator_number_of_llm_calls
            ),
            "aggregated_total_tokens_used": (
                probe_creation_total_tokens_used + evaluator_total_tokens_used
            ),
            "error": error,
        }
        metadata_path = cls._complete_probe_scan_metadata_path(leaf_dir)
        with open(metadata_path, "w", encoding="utf-8") as out_file:
            json.dump(metadata, out_file, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _trace_artifacts_dir(leaf_dir: str) -> str:
        return os.path.join(leaf_dir, "traces")

    @staticmethod
    def _resolve_trace_source_path(
        trace_path: str,
        traces_root: str | None = None,
    ) -> Path | None:
        raw_trace_path = str(trace_path or "").strip()
        if not raw_trace_path:
            return None

        raw_path = Path(raw_trace_path).expanduser()
        candidate_paths: list[Path] = [raw_path]
        project_root = Path(__file__).resolve().parents[3]

        if traces_root:
            traces_root_path = Path(traces_root).expanduser()
            if not traces_root_path.is_absolute():
                traces_root_path = project_root / traces_root_path
            candidate_paths.append(traces_root_path / raw_path.name)

        candidate_paths.append(project_root / raw_path)
        candidate_paths.append(
            project_root / "scanning" / "resources" / "traces" / raw_path.name
        )

        seen_candidates: set[str] = set()
        for candidate_path in candidate_paths:
            candidate_text = str(candidate_path)
            if candidate_text in seen_candidates:
                continue
            seen_candidates.add(candidate_text)
            if candidate_path.is_file():
                return candidate_path.resolve()

        return None

    @classmethod
    def _copy_traces_to_leaf_dir(
        cls,
        traces_of_execution: dict[str, str] | None,
        leaf_dir: str | None,
        traces_root: str | None = None,
    ) -> dict[str, str]:
        normalized_traces = dict(traces_of_execution or {})
        if not leaf_dir:
            return normalized_traces

        traces_leaf_dir = cls._trace_artifacts_dir(leaf_dir)
        os.makedirs(traces_leaf_dir, exist_ok=True)

        copied_trace_paths: dict[str, str] = {}
        for execution_cmd, trace_path in normalized_traces.items():
            raw_trace_path = str(trace_path or "").strip()
            if not raw_trace_path:
                copied_trace_paths[execution_cmd] = ""
                continue

            source_path = cls._resolve_trace_source_path(
                trace_path=raw_trace_path,
                traces_root=traces_root,
            )
            if source_path is None:
                copied_trace_paths[execution_cmd] = raw_trace_path
                continue

            destination_path = Path(traces_leaf_dir) / source_path.name
            try:
                if source_path.resolve() != destination_path.resolve():
                    shutil.copy2(source_path, destination_path)
            except OSError as copy_error:
                print(
                    f"[Scanning] Failed to copy trace '{source_path}' "
                    f"to '{destination_path}': {copy_error}"
                )
                copied_trace_paths[execution_cmd] = raw_trace_path
                continue

            copied_trace_paths[execution_cmd] = str(destination_path)

        return copied_trace_paths

    @staticmethod
    def _serialize_parsed_traces(
        traces_of_execution: dict[str, FlowSpec] | None,
    ) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for execution_cmd, parsed_trace in (traces_of_execution or {}).items():
            if hasattr(parsed_trace, "model_dump"):
                serialized[execution_cmd] = parsed_trace.model_dump(mode="json")
            elif isinstance(parsed_trace, dict):
                serialized[execution_cmd] = parsed_trace
            else:
                serialized[execution_cmd] = str(parsed_trace)
        return serialized

    @staticmethod
    def _save_parsed_trace(
        parsed_trace_path: str,
        traces_of_execution: dict[str, FlowSpec] | None,
    ) -> None:
        serialized_traces = ScanningExecuter._serialize_parsed_traces(
            traces_of_execution
        )
        with open(parsed_trace_path, "w", encoding="utf-8") as trace_file:
            json.dump(serialized_traces, trace_file, ensure_ascii=False, indent=2)

    @staticmethod
    def _normalize_loaded_traces_payload(payload: Any) -> dict[str, dict]:
        if isinstance(payload, dict):
            # New format: command -> flow payload.
            if payload and all(isinstance(v, dict) for v in payload.values()):
                return payload
            # Legacy/single-flow format: one flow payload saved directly.
            if "events" in payload:
                return {"cached_execution_0": payload}
        return {}

    @staticmethod
    def _trace_events(trace_payload: Any) -> list[Any]:
        if isinstance(trace_payload, dict):
            events = trace_payload.get("events")
            return events if isinstance(events, list) else []

        events = getattr(trace_payload, "events", None)
        return events if isinstance(events, list) else []

    @classmethod
    def _is_short_terminal_error_trace(cls, trace_payload: Any) -> bool:
        events = cls._trace_events(trace_payload)
        if not events or len(events) > 3:
            return False

        last_event = events[-1]
        if isinstance(last_event, dict):
            event_type = last_event.get("type")
        else:
            event_type = getattr(last_event, "type", None)

        return str(event_type or "").strip().lower() == "error"

    @classmethod
    def _infer_probe_result_status(cls, traces_of_execution: Any) -> int:
        normalized_traces = cls._normalize_loaded_traces_payload(traces_of_execution)
        if normalized_traces:
            traces_by_command = normalized_traces
        elif isinstance(traces_of_execution, dict):
            traces_by_command = dict(traces_of_execution)
        else:
            traces_by_command = {}

        if not traces_by_command:
            return 0

        for parsed_trace in traces_by_command.values():
            if not cls._is_short_terminal_error_trace(parsed_trace):
                return 1
        return 0

    @classmethod
    def _save_probe_results_cache(cls, leaf_dir: str, probe_results: dict) -> None:
        parsed_trace_path = cls._parsed_trace_path(leaf_dir)
        malicious_nodespec_path = cls._malicious_nodespec_path(leaf_dir)
        probe_result_metadata_path = cls._probe_result_metadata_path(leaf_dir)

        cls._save_parsed_trace(
            parsed_trace_path=parsed_trace_path,
            traces_of_execution=probe_results.get("traces_of_execution") or {},
        )

        malicious_twin = probe_results.get("malicious_twin")
        if isinstance(malicious_twin, NodeSpec):
            malicious_twin.to_json(path=malicious_nodespec_path)
        elif isinstance(malicious_twin, dict):
            with open(malicious_nodespec_path, "w", encoding="utf-8") as out_file:
                json.dump(malicious_twin, out_file, ensure_ascii=False, indent=2)
        else:
            with open(malicious_nodespec_path, "w", encoding="utf-8") as out_file:
                json.dump({}, out_file)

        metadata = {
            "probe_name": probe_results.get("probe_name"),
            "status": probe_results.get("status"),
        }
        with open(probe_result_metadata_path, "w", encoding="utf-8") as out_file:
            json.dump(metadata, out_file, ensure_ascii=False, indent=2)

    @classmethod
    def _load_cached_probe_results(cls, leaf_dir: str) -> dict | None:
        parsed_trace_path = cls._parsed_trace_path(leaf_dir)
        malicious_nodespec_path = cls._malicious_nodespec_path(leaf_dir)
        probe_result_metadata_path = cls._probe_result_metadata_path(leaf_dir)
        if not (
            os.path.exists(parsed_trace_path)
            and os.path.exists(malicious_nodespec_path)
            and os.path.exists(probe_result_metadata_path)
        ):
            return None

        try:
            with open(parsed_trace_path, "r", encoding="utf-8") as trace_file:
                traces_payload_raw = json.load(trace_file)
            traces_payload = cls._normalize_loaded_traces_payload(traces_payload_raw)
            if not isinstance(traces_payload, dict):
                return None

            with open(malicious_nodespec_path, "r", encoding="utf-8") as ns_file:
                malicious_nodespec_payload = json.load(ns_file) or {}
            malicious_node_spec = NodeSpec.from_json(data=malicious_nodespec_payload)

            with open(probe_result_metadata_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file) or {}
        except Exception:
            return None
        probe_name = metadata.get("probe_name") or ""
        cached_status = metadata.get("status")
        inferred_status = cls._infer_probe_result_status(traces_payload)
        if isinstance(cached_status, int):
            status = 0 if cached_status == 0 or inferred_status == 0 else 1
        else:
            status = inferred_status

        return {
            "probe_name": probe_name,
            "status": status,
            "traces_of_execution": traces_payload,
            "malicious_twin": malicious_node_spec,
        }

    @staticmethod
    def _existing_evaluator_names(csv_path: str) -> set[str]:
        if not os.path.exists(csv_path):
            return set()

        existing_evaluators: set[str] = set()
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                evaluator_name = str(row.get("evaluator_name", "")).strip()
                if evaluator_name:
                    existing_evaluators.add(evaluator_name)
        return existing_evaluators

    @staticmethod
    def _append_probe_evaluator_result_row(
        csv_path: str,
        probe_name: str,
        attack_suite: str,
        attack_surface: str,
        attacker_intent: str,
        flow_number: int,
        evaluator_name: str,
        evaluator_results: dict,
    ) -> None:
        evaluator_results = evaluator_results or {}
        row = {
            "probe_name": probe_name,
            "attack_suite": attack_suite,
            "attack_surface": attack_surface,
            "attacker_intent": attacker_intent,
            "flow_number": flow_number,
            "evaluator_name": evaluator_name,
            "category": evaluator_results.get("category"),
            "value": evaluator_results.get("value"),
            "failure_threshold": evaluator_results.get("failure_threshold"),
            "time": evaluator_results.get("time", 0),
            "number_of_llm_calls": evaluator_results.get("number_of_llm_calls", 0),
            "total_tokens_used": evaluator_results.get("total_tokens_used", 0),
        }

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=ScanningExecuter.EVALUATOR_RESULTS_CSV_HEADER
            )
            writer.writerow(row)

    @staticmethod
    def _is_resource_overload_probe(probe: BaseProbe) -> bool:
        attacker_intent = str(getattr(probe, "attacker_intent", "") or "").strip()
        return attacker_intent.lower() == "resource overload"

    @staticmethod
    def _is_utility_evaluator(evaluator: BaseEvaluator) -> bool:
        evaluator_category = str(getattr(evaluator, "category", "") or "").strip().lower()
        evaluator_name = evaluator.__class__.__name__.strip().lower()
        return evaluator_category == "utility" or "utility" in evaluator_name

    def _write_synthetic_resource_overload_results(
        self,
        probe: BaseProbe,
        flow_num: int,
        scan_results_dir: str | None,
        apply_defenses: bool,
    ) -> None:
        if not scan_results_dir:
            return

        probe_key = probe._get_name()
        evaluators = self.evaluators.get(probe_key, [])
        if not evaluators:
            return

        base_leaf_dir = self._leaf_results_dir(
            scan_results_dir=scan_results_dir,
            probe_name=probe.name,
            flow_num=flow_num,
        )
        os.makedirs(base_leaf_dir, exist_ok=True)

        target_leaf_dirs: list[str] = [base_leaf_dir]
        if apply_defenses:
            for defense in self._load_enabled_defenses(
                apply_defenses=apply_defenses,
            ):
                defense_leaf_dir = self._defense_results_dir(
                    leaf_dir=base_leaf_dir,
                    defense_name=defense.__class__.__name__,
                )
                no_attack_leaf_dir = self._defense_results_dir(
                    leaf_dir=defense_leaf_dir,
                    defense_name="no_attack_results",
                )
                under_attack_leaf_dir = self._defense_results_dir(
                    leaf_dir=defense_leaf_dir,
                    defense_name="results_under_attack",
                )
                os.makedirs(no_attack_leaf_dir, exist_ok=True)
                os.makedirs(under_attack_leaf_dir, exist_ok=True)
                target_leaf_dirs.extend([no_attack_leaf_dir, under_attack_leaf_dir])

        for target_leaf_dir in target_leaf_dirs:
            csv_path = self._results_csv_path(target_leaf_dir)
            existing_evaluator_names = self._existing_evaluator_names(csv_path)

            for evaluator in evaluators:
                evaluator_name = evaluator.__class__.__name__
                if evaluator_name in existing_evaluator_names:
                    continue
                if self._is_utility_evaluator(evaluator):
                    continue

                synthetic_results = {
                    "category": getattr(evaluator, "category", "Attack Success"),
                    "value": True,
                    "failure_threshold": getattr(evaluator, "failure_threshold", 0.5),
                    "time": 0,
                    "number_of_llm_calls": 0,
                    "total_tokens_used": 0,
                }
                self._append_probe_evaluator_result_row(
                    csv_path=csv_path,
                    probe_name=probe.name,
                    attack_suite=getattr(probe, "attack_suit", ""),
                    attack_surface=getattr(probe, "attack_surface", ""),
                    attacker_intent=getattr(probe, "attacker_intent", ""),
                    flow_number=flow_num,
                    evaluator_name=evaluator_name,
                    evaluator_results=synthetic_results,
                )

    def execute_scanning_process(
        self,
        user_id: str,
        system_name: str,
        scan_id: str,
        full_analysis_node_spec: NodeSpec,
        flows: List[FlowSpec],
        scan_results_dir: str | None = None,
        flows_per_probe: int = 3,
        apply_defenses: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the scanning process for the given full analysis node spec using the initialized probes and evaluators.

        Args:
            user_id (str): The user identifier.
            system_name (str): The system identifier.
            scan_id (str): The unique identifier for the scanning process.
            full_analysis_node_spec (NodeSpec): The complete analysis NodeSpec of the system to be scanned.
        Returns:
            Dict[str, Any]: Scan status payload including probe execution summary.
        """
        self.empty_trace_events = []
        no_relevant_flow_probes: list[dict[str, str]] = []
        missing_code_reference_probes: list[dict[str, Any]] = []
        probe_scan_exceptions: list[dict[str, Any]] = []
        if scan_results_dir:
            os.makedirs(scan_results_dir, exist_ok=True)

        scan_defenses: list[BaseDefense | None] = []
        if apply_defenses:
            scan_defenses.extend(
                self._load_enabled_defenses(
                    apply_defenses=apply_defenses,
                )
            )
            if not scan_defenses:
                print(
                    "[Scanning] No defenses were loaded, falling back to "
                    "non-defense probe execution"
                )
        if not scan_defenses:
            scan_defenses.append(None)

        for probe in self.probes:
            probe_identity = self._probe_identity(probe)
            relevant_flows = probe.filter_relevant_flows(
                full_node_spec=full_analysis_node_spec,
                flows=flows,
            )
            if not relevant_flows:
                no_relevant_flow_probes.append(probe_identity)
            selected_flow_numbers: list[int] = []

            for defense_index, defense in enumerate(scan_defenses):
                defense_successful_flows: list[int] = []
                candidate_flow_numbers = (
                    list(range(len(relevant_flows)))
                    if defense_index == 0
                    else selected_flow_numbers
                )
                candidate_index = 0
                failed = 0
                while (
                    len(defense_successful_flows) < flows_per_probe
                    and candidate_index < len(candidate_flow_numbers)
                    and failed < 10
                ):
                    flow_num = candidate_flow_numbers[candidate_index]
                    existing_skip_marker = self._load_flow_skip_marker(
                        scan_results_dir=scan_results_dir,
                        probe_name=probe.name,
                        flow_num=flow_num,
                    )
                    if existing_skip_marker is not None:
                        print(
                            f"[Scanning] Skipping probe {probe.name} flow_{flow_num} "
                            "because a skip marker file already exists"
                        )
                        marker_entry = self._build_probe_scan_skip_entry(
                            probe_identity,
                            flow_num=flow_num,
                            error=existing_skip_marker.get("error"),
                            synthetic_results_applied=bool(
                                existing_skip_marker.get("synthetic_results_applied")
                            ),
                        )
                        skip_counts_as_failure = self._flow_skip_counts_as_failure(
                            existing_skip_marker
                        )
                        if not skip_counts_as_failure:
                            self._remove_flow_probe_scan_metadata(
                                scan_results_dir=scan_results_dir,
                                probe_name=probe.name,
                                flow_num=flow_num,
                            )
                        if existing_skip_marker.get("synthetic_results_applied"):
                            defense_successful_flows.append(flow_num)
                        elif skip_counts_as_failure:
                            failed += 1
                            probe_scan_exceptions.append(marker_entry)
                        else:
                            missing_code_reference_probes.append(marker_entry)
                        candidate_index += 1
                        continue
    
                    flow = relevant_flows[flow_num]
                    try:
                        probe_scan_succeeded = self._complete_probe_scan(
                            probe=probe,
                            flow=flow,
                            flow_num=flow_num,
                            full_analysis_node_spec=full_analysis_node_spec,
                            guidance={},
                            scan_results_dir=scan_results_dir,
                            user_id=user_id,
                            system_name=system_name,
                            scan_id=scan_id,
                            apply_defenses=apply_defenses,
                            defense=defense,
                        )
                        if probe_scan_succeeded:
                            defense_successful_flows.append(flow_num)
                        else:
                            failed += 1
                    except Exception as scan_error:
                        is_resource_overload = self._is_resource_overload_probe(probe)
                        self._mark_flow_for_skip(
                            scan_results_dir=scan_results_dir,
                            probe_name=probe.name,
                            flow_num=flow_num,
                            error=scan_error,
                            synthetic_results_applied=is_resource_overload,
                        )
                        skip_entry = self._build_probe_scan_skip_entry(
                            probe_identity,
                            flow_num=flow_num,
                            error=scan_error,
                            synthetic_results_applied=is_resource_overload,
                        )
                        if is_resource_overload:
                            self._write_synthetic_resource_overload_results(
                                probe=probe,
                                flow_num=flow_num,
                                scan_results_dir=scan_results_dir,
                                apply_defenses=apply_defenses,
                            )
                            defense_successful_flows.append(flow_num)
                        elif self._is_missing_code_reference_error(scan_error):
                            missing_code_reference_probes.append(skip_entry)
                        else:
                            failed += 1
                            probe_scan_exceptions.append(skip_entry)
                    candidate_index += 1

                if defense_index == 0:
                    selected_flow_numbers = defense_successful_flows[:flows_per_probe]

        self._print_empty_trace_summary()
        self._save_probes_not_run_summary(
            scan_results_dir=scan_results_dir,
            scan_id=scan_id,
            user_id=user_id,
            system_name=system_name,
            no_relevant_flow_probes=no_relevant_flow_probes,
            missing_code_reference_probes=missing_code_reference_probes,
            probe_scan_exceptions=probe_scan_exceptions,
        )
        return {
            "user_id": user_id,
            "system_name": system_name,
            "scan_id": scan_id,
            "probes_list": self.probes,
            "probes_not_run": {
                "no_relevant_flows": no_relevant_flow_probes,
                "missing_code_references": missing_code_reference_probes,
                "exception_in_complete_probe_scan": probe_scan_exceptions,
            },
            "status": "completed",
        }

    def _complete_probe_scan(
        self,
        probe: BaseProbe,
        flow: FlowSpec,
        flow_num: int,
        full_analysis_node_spec: NodeSpec,
        guidance: dict | None,
        scan_results_dir: str | None,
        user_id: str,
        system_name: str,
        scan_id: str,
        apply_defenses: bool = False,
        defense: BaseDefense | None = None,
    ) -> bool:
        leaf_dir: str | None = None
        if scan_results_dir:
            leaf_dir = self._leaf_results_dir(
                scan_results_dir=scan_results_dir,
                probe_name=probe.name,
                flow_num=flow_num,
            )
            os.makedirs(leaf_dir, exist_ok=True)

        scan_started_at_utc = self._utc_now_isoformat()
        scan_started_at_perf = time.perf_counter()
        scan_status = "completed"
        scan_error: str | None = None
        save_probe_scan_metadata = True
        probe_scan_succeeded = True
        probe_creation_duration_seconds = 0.0
        system_execution_count = 0
        probe_creation_number_of_llm_calls = 0
        probe_creation_total_tokens_used = 0
        expected_result_count = 0
        completed_result_count = 0

        def update_probe_creation_usage(malicious_node_spec: NodeSpec | None) -> None:
            nonlocal probe_creation_duration_seconds, system_execution_count, probe_creation_number_of_llm_calls, probe_creation_total_tokens_used
            (
                probe_creation_duration_seconds,
                probe_creation_number_of_llm_calls,
                probe_creation_total_tokens_used,
                system_execution_count,
            ) = self._extract_probe_instance_usage(malicious_node_spec)

        try:
            target_leaf_dir = leaf_dir
            if defense is not None and leaf_dir:
                target_leaf_dir = self._defense_results_dir(
                    leaf_dir=leaf_dir,
                    defense_name=defense.__class__.__name__,
                )
                os.makedirs(target_leaf_dir, exist_ok=True)
            scan_targets = [(defense, target_leaf_dir)]
            shared_malicious_twin_payload: tuple[NodeSpec, Any, List[str]] | None = None

            def get_shared_malicious_twin_payload() -> tuple[NodeSpec, Any, List[str]]:
                nonlocal shared_malicious_twin_payload
                if shared_malicious_twin_payload is None:
                    shared_malicious_twin_payload = self._build_malicious_twin_payload(
                        probe_instance=probe,
                        full_analysis_node_spec=full_analysis_node_spec,
                        flow=flow,
                        guidance=guidance,
                    )
                    update_probe_creation_usage(shared_malicious_twin_payload[0])
                return shared_malicious_twin_payload

            for defense, target_leaf_dir in scan_targets:
                results_to_process: list[tuple[dict, str | None]] = []
                if defense is None:
                    probe_results: dict | None = None
                    if target_leaf_dir:
                        probe_results = self._load_cached_probe_results(target_leaf_dir)
                        if probe_results is not None:
                            print(
                                f"[Scanning] Reusing cached traces for probe {probe.name} "
                                f"on flow_{flow_num}"
                            )

                    if probe_results is None:
                        print(f"Executing probe {probe.name} on flow_{flow_num}")
                        probe_results = self.execute_probe(
                            probe_instance=probe,
                            full_analysis_node_spec=full_analysis_node_spec,
                            flow=flow,
                            guidance=guidance,
                            leaf_dir=target_leaf_dir,
                            malicious_twin_payload=get_shared_malicious_twin_payload(),
                        )
                        if target_leaf_dir:
                            self._save_probe_results_cache(
                                leaf_dir=target_leaf_dir,
                                probe_results=probe_results,
                            )
                    results_to_process.append((probe_results, target_leaf_dir))
                else:
                    no_attack_leaf_dir: str | None = None
                    under_attack_leaf_dir: str | None = None
                    no_attack_results: dict | None = None
                    under_attack_results: dict | None = None

                    if target_leaf_dir:
                        no_attack_leaf_dir = self._defense_results_dir(
                            leaf_dir=target_leaf_dir,
                            defense_name="no_attack_results",
                        )
                        under_attack_leaf_dir = self._defense_results_dir(
                            leaf_dir=target_leaf_dir,
                            defense_name="results_under_attack",
                        )

                        no_attack_results = self._load_cached_probe_results(
                            no_attack_leaf_dir
                        )
                        under_attack_results = self._load_cached_probe_results(
                            under_attack_leaf_dir
                        )
                        if (
                            no_attack_results is not None
                            and under_attack_results is not None
                        ):
                            print(
                                f"[Scanning] Reusing cached traces for probe {probe.name} "
                                f"on flow_{flow_num} with defense {defense.__class__.__name__}"
                            )

                    if no_attack_results is None or under_attack_results is None:
                        print(
                            f"Executing probe {probe.name} on flow_{flow_num} "
                            f"with defense {defense.__class__.__name__}"
                        )
                        no_attack_results, under_attack_results = (
                            self.execute_probe_with_defense(
                                probe_instance=probe,
                                full_analysis_node_spec=full_analysis_node_spec,
                                flow=flow,
                                defense=defense,
                                guidance=guidance,
                                leaf_dir=target_leaf_dir,
                                malicious_twin_payload=get_shared_malicious_twin_payload(),
                            )
                        )
                        if no_attack_leaf_dir:
                            self._save_probe_results_cache(
                                leaf_dir=no_attack_leaf_dir,
                                probe_results=no_attack_results,
                            )
                        if under_attack_leaf_dir:
                            self._save_probe_results_cache(
                                leaf_dir=under_attack_leaf_dir,
                                probe_results=under_attack_results,
                            )

                    results_to_process.append((no_attack_results, no_attack_leaf_dir))
                    results_to_process.append(
                        (under_attack_results, under_attack_leaf_dir)
                    )

                for probe_results, results_leaf_dir in results_to_process:
                    expected_result_count += 1
                    if not isinstance(probe_results, dict):
                        probe_scan_succeeded = False
                        scan_error = (
                            "Probe execution did not produce results for every "
                            "defense run."
                        )
                        continue

                    completed_result_count += 1
                    update_probe_creation_usage(probe_results.get("malicious_twin"))
                    if not probe_results.get("probe_name"):
                        probe_results["probe_name"] = probe._get_name()

                    inferred_probe_status = self._infer_probe_result_status(
                        probe_results.get("traces_of_execution")
                    )
                    current_probe_status = probe_results.get("status")
                    if isinstance(current_probe_status, int):
                        probe_results["status"] = (
                            0
                            if current_probe_status == 0 or inferred_probe_status == 0
                            else 1
                        )
                    else:
                        probe_results["status"] = inferred_probe_status

                    if probe_results["status"] == 0:
                        probe_scan_succeeded = False

                    scanning_results = ScanningResults(
                        user_id=user_id,
                        system_name=system_name,
                        scan_id=scan_id,
                        probe_results=probe_results,
                    )

                    probe_csv_path: str | None = None
                    existing_evaluator_names: set[str] = set()
                    if results_leaf_dir:
                        probe_csv_path = self._results_csv_path(results_leaf_dir)
                        existing_evaluator_names = self._existing_evaluator_names(
                            probe_csv_path
                        )

                    for evaluator in self.evaluators.get(
                        scanning_results.probe_name, []
                    ):
                        evaluator_name = evaluator.__class__.__name__
                        if evaluator_name in existing_evaluator_names:
                            print(
                                f"[Scanning] Skipping evaluator {evaluator_name} for "
                                f"probe {scanning_results.probe_name} because its "
                                "result row already exists in CSV"
                            )
                            continue

                        scanning_results = self.execute_evaluator(
                            evaluator,
                            scanning_results,
                        )
                        if probe_csv_path:
                            self._append_probe_evaluator_result_row(
                                csv_path=probe_csv_path,
                                probe_name=probe.name,
                                attack_suite=getattr(probe, "attack_suit", ""),
                                attack_surface=getattr(probe, "attack_surface", ""),
                                attacker_intent=getattr(probe, "attacker_intent", ""),
                                flow_number=flow_num,
                                evaluator_name=scanning_results.evaluator_name,
                                evaluator_results=scanning_results.evaluator_results,
                            )
                            existing_evaluator_names.add(scanning_results.evaluator_name)

            defense_results_complete = (
                expected_result_count > 0
                and completed_result_count == expected_result_count
            )
            if not defense_results_complete:
                probe_scan_succeeded = False
                if not scan_error:
                    scan_error = (
                        "Probe execution did not produce results for every "
                        "defense run."
                    )

            if not probe_scan_succeeded:
                scan_status = "failed"
                if not scan_error:
                    scan_error = (
                        "Probe execution produced a parsed trace with 3 or fewer "
                        "events and a terminal error event."
                    )

            self.remove_image(f"{user_id}_{system_name}:malicious")
            if defense is not None:
                return defense_results_complete
            return probe_scan_succeeded
        except Exception as error:
            scan_error = str(error)
            if self._is_missing_code_reference_error(error):
                scan_status = self.MISSING_CODE_REFERENCES_SKIP_REASON
                save_probe_scan_metadata = False
            else:
                scan_status = "failed"
            raise
        finally:
            (
                evaluator_duration_seconds,
                evaluator_number_of_llm_calls,
                evaluator_total_tokens_used,
            ) = self._aggregate_evaluator_usage_from_results_tree(leaf_dir)
            if save_probe_scan_metadata:
                self._save_complete_probe_scan_metadata(
                    leaf_dir=leaf_dir,
                    probe_name=probe._get_name(),
                    flow_num=flow_num,
                    guidance=guidance,
                    apply_defenses=apply_defenses,
                    started_at_utc=scan_started_at_utc,
                    completed_at_utc=self._utc_now_isoformat(),
                    duration_seconds=time.perf_counter() - scan_started_at_perf,
                    probe_creation_duration_seconds=probe_creation_duration_seconds,
                    evaluator_duration_seconds=evaluator_duration_seconds,
                    system_execution_count=system_execution_count,
                    status=scan_status,
                    probe_creation_number_of_llm_calls=probe_creation_number_of_llm_calls,
                    probe_creation_total_tokens_used=probe_creation_total_tokens_used,
                    evaluator_number_of_llm_calls=evaluator_number_of_llm_calls,
                    evaluator_total_tokens_used=evaluator_total_tokens_used,
                    error=scan_error,
                )

    @staticmethod
    def remove_image(image_tag, force: bool = False) -> bool:
        """
        Remove the Docker image associated with this language handler.
        """
        cmd = ["docker", "rmi"]
        if force:
            cmd.append("-f")
        cmd.append(image_tag)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.lower()

            # Image does not exist → treat as success
            if "no such image" in stderr:
                return False

            raise RuntimeError(
                "Docker image removal failed.",
                f"STDOUT: {result.stdout}",
                f"STDERR: {result.stderr}",
            )
    
    def parse_traces_of_execution(self, traces_of_execution: Dict[str, str], runtime_mapping_file: str, full_analysis_node_spec: NodeSpec, node_spec_file: str | Path) -> Dict[str, FlowSpec]:
        """
        Parse the raw traces of execution using a standalone trace parser function, and return a dictionary of parsed traces (flows).

        Args:
            traces_of_execution (Dict[str, str]): A dictionary where keys are execution commands and values are paths to the raw traces of execution to be parsed.
            runtime_mapping_file (str): The path to the runtime mapping file used for parsing the traces.
            full_analysis_node_spec (NodeSpec): The complete analysis NodeSpec of the system to be scanned, used for parsing the traces and mapping them to the nodes in the specification.
            node_spec_file (str | Path): The path to the node specification file.
        returns:
            parsed_traces (Dict[str, FlowSpec]): A dictionary where keys are execution commands and values are parsed traces converted to FlowSpec.
        """
        parsed_traces = {}
        for cmd, trace_path in traces_of_execution.items():
            parsed_trace = parse_single_trace(
                trace_file=trace_path,
                parsed_out_file=None,
                runtime_mapping_file=runtime_mapping_file,
                nodespec_file=node_spec_file,
            )
            source_trace_path = str(trace_path or "").strip()
            if isinstance(parsed_trace, dict):
                parsed_trace["source_trace_file"] = source_trace_path
            elif hasattr(parsed_trace, "source_trace_file"):
                try:
                    setattr(parsed_trace, "source_trace_file", source_trace_path)
                except Exception:
                    pass
            parsed_traces[cmd] = parsed_trace
        
        return parsed_traces

    def execute_probe(
        self,
        probe_instance: BaseProbe,
        full_analysis_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: Dict | None = None,
        leaf_dir: str | None = None,
        malicious_twin_payload: tuple[NodeSpec, Any, List[str]] | None = None,
    ) -> dict:
        """
        Executes a single probe instance on the provided full analysis node spec.
        Args:
            probe_instance (BaseProbe): The probe instance to be executed.
            full_analysis_node_spec (NodeSpec): The complete analysis NodeSpec of the system to be scanned.
            flow (FlowSpec): The single flow to be scanned.
        Returns:
            dict: the command of exection and it's corresponding trace of exection path.
        """
        runtime_mapping_file = full_analysis_node_spec.metadata.get("runtime_mapping_path", "")
        (
            malicious_node_spec,
            affected_nodes,
            execution_cmds,
        ) = self._resolve_malicious_twin_payload(
            probe_instance=probe_instance,
            full_analysis_node_spec=full_analysis_node_spec,
            flow=flow,
            guidance=guidance,
            malicious_twin_payload=malicious_twin_payload,
        )
        cmd_trace_path_dict = self.abstraction_executer.execute_scan(malicious_node_spec, 
                                                                     execution_cmds, 
                                                                     affected_nodes)
        self._record_empty_trace_paths(
            probe_instance=probe_instance,
            execution_cmds=execution_cmds,
            cmd_trace_path_dict=cmd_trace_path_dict,
        )
        cmd_trace_path_dict = probe_instance._normalize_traces_payload(
            cmd_trace_path_dict
        )
        cmd_trace_path_dict = self._copy_traces_to_leaf_dir(
            traces_of_execution=cmd_trace_path_dict,
            leaf_dir=leaf_dir,
            traces_root=getattr(self.abstraction_executer, "traces_path", None),
        )


        parsed_traces = self.parse_traces_of_execution(cmd_trace_path_dict, 
                                                       runtime_mapping_file=runtime_mapping_file, 
                                                       full_analysis_node_spec=full_analysis_node_spec,
                                                       node_spec_file = full_analysis_node_spec.metadata.get("node_spec_py_path", ""))
        probe_results = probe_instance._build_result(parsed_traces, malicious_node_spec)
        probe_results["status"] = self._infer_probe_result_status(
            probe_results.get("traces_of_execution")
        )
        return probe_results

    def execute_probe_with_defense(
        self,
        probe_instance: BaseProbe,
        full_analysis_node_spec: NodeSpec,
        flow: FlowSpec,
        defense: BaseDefense,
        guidance: Dict | None = None,
        leaf_dir: str | None = None,
        malicious_twin_payload: tuple[NodeSpec, Any, List[str]] | None = None,
    ) -> tuple[dict, dict]:
        """
        Executes a single probe instance on the provided full analysis node spec.
        Args:
            probe_instance (BaseProbe): The probe instance to be executed.
            full_analysis_node_spec (NodeSpec): The complete analysis NodeSpec of the system to be scanned.
            flow (FlowSpec): The single flow to be scanned.
        Returns:
            tuple[dict, dict]: first is results with defense on the original flow
            (no attack), second is results with defense on the malicious flow.
        """
        runtime_mapping_file = full_analysis_node_spec.metadata.get("runtime_mapping_path", "")
        (
            malicious_node_spec,
            affected_nodes,
            execution_cmds,
        ) = self._resolve_malicious_twin_payload(
            probe_instance=probe_instance,
            full_analysis_node_spec=full_analysis_node_spec,
            flow=flow,
            guidance=guidance,
            malicious_twin_payload=malicious_twin_payload,
        )

        no_attack_leaf_dir: str | None = None
        under_attack_leaf_dir: str | None = None
        if leaf_dir:
            no_attack_leaf_dir = self._defense_results_dir(
                leaf_dir=leaf_dir,
                defense_name="no_attack_results",
            )
            under_attack_leaf_dir = self._defense_results_dir(
                leaf_dir=leaf_dir,
                defense_name="results_under_attack",
            )
            os.makedirs(no_attack_leaf_dir, exist_ok=True)
            os.makedirs(under_attack_leaf_dir, exist_ok=True)

        def run_probe(node_spec, execution_cmds, affected_nodes, run_leaf_dir):

            node_spec, affected_nodes = defense.update_node_spec(full_node_spec=node_spec, affected_nodes=affected_nodes, execution_cmds=execution_cmds)

            cmd_trace_path_dict = self.abstraction_executer.execute_scan(node_spec, 
                                                                        execution_cmds, 
                                                                        affected_nodes)
            self._record_empty_trace_paths(
                probe_instance=probe_instance,
                execution_cmds=execution_cmds,
                cmd_trace_path_dict=cmd_trace_path_dict,
            )
            cmd_trace_path_dict = probe_instance._normalize_traces_payload(
                cmd_trace_path_dict
            )
            cmd_trace_path_dict = self._copy_traces_to_leaf_dir(
                traces_of_execution=cmd_trace_path_dict,
                leaf_dir=run_leaf_dir,
                traces_root=getattr(self.abstraction_executer, "traces_path", None),
            )


            parsed_traces = self.parse_traces_of_execution(cmd_trace_path_dict, 
                                                       runtime_mapping_file=runtime_mapping_file, 
                                                       full_analysis_node_spec=full_analysis_node_spec,
                                                       node_spec_file = full_analysis_node_spec.metadata.get("node_spec_py_path", ""))
            probe_results = probe_instance._build_result(parsed_traces, node_spec)
            probe_results["status"] = self._infer_probe_result_status(
                probe_results.get("traces_of_execution")
            )
            return probe_results
        input_arguments = build_input_arguments(flow=flow, guidance=guidance)
        benign_execution_cmds = build_execution_cmds(
            malicious_node_spec=malicious_node_spec,
            input_arguments=input_arguments,
        )
        full_analysis_node_spec.metadata["instance_info"] = malicious_node_spec.metadata["instance_info"]
        no_attack_results = run_probe(
            full_analysis_node_spec.model_copy(deep=True),
            benign_execution_cmds,
            {},
            no_attack_leaf_dir,
        )
        under_attack_results = run_probe(
            malicious_node_spec,
            execution_cmds,
            affected_nodes,
            under_attack_leaf_dir,
        )
        return no_attack_results, under_attack_results

    def _resolve_malicious_twin_payload(
        self,
        probe_instance: BaseProbe,
        full_analysis_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: Dict | None = None,
        malicious_twin_payload: tuple[NodeSpec, Any, List[str]] | None = None,
    ) -> tuple[NodeSpec, Any, List[str]]:
        if malicious_twin_payload is None:
            malicious_twin_payload = self._build_malicious_twin_payload(
                probe_instance=probe_instance,
                full_analysis_node_spec=full_analysis_node_spec,
                flow=flow,
                guidance=guidance,
            )

        malicious_node_spec, affected_nodes, execution_cmds = malicious_twin_payload

        malicious_node_spec = malicious_node_spec.model_copy(deep=True)
        affected_nodes = copy.deepcopy(affected_nodes)
        execution_cmds = list(execution_cmds or [])
        return malicious_node_spec, affected_nodes, execution_cmds


    def execute_evaluator(
        self,
        evaluator_instance: BaseEvaluator,
        scanning_results: ScanningResults,
    ) -> ScanningResults:
        """
        Executes a single evaluator instance on the provided scanning results.The evaluators that are loaded depends on the probes that were used in the scanning process.
        Args:
            evaluator_instance (BaseEvaluator): The evaluator instance to be executed.
            scanning_results (ScanningResults): The results object from the scanning execution.
        Returns:
            ScanningResults: The results of the evaluation along with the scanning status.
        """
        evaluator_started_at = time.perf_counter()
        with collect_evaluator_runtime_metrics() as runtime_metrics:
            evaluator_results = evaluator_instance.evaluate_results(
                scanning_results=scanning_results,
            )
        evaluator_runtime_payload = {
            "time": round(time.perf_counter() - evaluator_started_at, 6),
            "number_of_llm_calls": runtime_metrics.llm_calls,
            "total_tokens_used": runtime_metrics.total_tokens_used,
        }
        if isinstance(evaluator_results, ScanningResults):
            merged_results = dict(evaluator_results.evaluator_results or {})
            merged_results.update(evaluator_runtime_payload)
            evaluator_results.populate_evaluator_results(
                evaluator_name=evaluator_results.evaluator_name or evaluator_instance.__class__.__name__,
                evaluator_results=merged_results,
            )
            return evaluator_results
        if isinstance(evaluator_results, dict):
            evaluator_results = evaluator_instance.normalize_evaluator_results(
                evaluator_results
            )
            evaluator_results.update(evaluator_runtime_payload)
            scanning_results.populate_evaluator_results(
                evaluator_name=evaluator_instance.__class__.__name__,
                evaluator_results=evaluator_results,
            )
            return scanning_results
        raise TypeError(
            "Evaluators must return either ScanningResults or a dict of "
            "evaluator results."
        )
