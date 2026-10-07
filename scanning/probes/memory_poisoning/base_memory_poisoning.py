import random
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any
import tempfile

from node_spec.structure_schema import FlowSpec, NodeSpec
from llm_sandbox import SandboxSession
from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.rag_utils.json_modifier import JsonTextModifier
from scanning.probes.utils.rag_utils.llm import FalseAnswerGenerator, FalseCorpusGenerator
from scanning.probes.utils.base_util import (
    get_unique_invoked_tool_ids,
    normalize_flows,
)
from scanning.probes.utils.flow_filltering import (
    get_database_node_ids,
    get_database_node_ids_with_data_path,
    get_database_node_ids_with_data_path_code_references,
    get_invoked_rag_tool_ids,
    has_entry_point_script,
)


class BaseMemoryPoisoning(BaseProbe, ABC):
    """
    Shared base class for probes that poison persisted memory used by a RAG
    component.

    These probes mutate the backing database file used by a RAG tool so the
    agent later retrieves malicious records. The shared helpers here cover:

    1. Selecting flows that invoke a tool containing a RAG node backed by a
       database node.
    2. Resolving the retrieval tool, database node, and attacked agent from
       the selected flow.
    3. Extracting the attacked agent system prompt and RAG call context.
    4. Building execution commands and probe metadata.
    5. Generating and writing poisoned database records to a cloned data file.
    """
    guidance_fields = ["attacked_agent_llm_node_id", "rag_args", "rag_results"]

    def __init__(
        self,
        name: str = "Generic Threat",
        description: str = "",
        attack_surface: str = "",
        attack_suit: str = "",
        attacker_intent: str = "",
        requirements: list[str] = [],
    ):
        super().__init__(
            name=name,
            description=description,
            attack_surface=attack_surface,
            attack_suit=attack_suit,
            attacker_intent=attacker_intent,
            requirements=requirements,
        )
        self.abstraction_executer = None

    def get_attackable_retrieval_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked retrieval-tool ids that this probe can structurally attack.
        """
        return get_invoked_rag_tool_ids(
            full_node_spec=full_node_spec,
            flow=flow,
        )

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata memory-poisoning pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The flow invoked at least one RAG retrieval tool.
        3. The node spec defines at least one database node with a concrete
           ``data_path`` value.
        4. At least one such database node defines ``data_path`` code references.

        Deeper flow-context and code-snippet checks, such as recovering the exact
        RAG call context or extracting the attacked agent's prompt text, are
        intentionally deferred to the later attack-construction step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        database_node_ids = get_database_node_ids_with_data_path_code_references(
            full_node_spec
        )
        if not database_node_ids:
            return []

        for flow in normalize_flows(full_node_spec, flows):
            try:
                invoked_rag_tool_ids = get_invoked_rag_tool_ids(
                    full_node_spec=full_node_spec,
                    flow=flow,
                )
                if invoked_rag_tool_ids:
                    relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @staticmethod
    def find_retrieval_tool_and_db_node(
        full_node_spec: NodeSpec,
        invoked_tool_ids: list[str],
    ) -> tuple[Any, Any]:
        """
        Return the first invoked tool that contains both a RAG node and its
        backing database node.
        """
        retrieval_tool_node = None
        for tool_id in invoked_tool_ids:
            tool_node = full_node_spec.get_node(node_id=tool_id)
            if tool_node is None:
                raise ValueError(f"Invoked tool '{tool_id}' was not found in node spec.")
            if getattr(tool_node, "is_rag_tool", False):
                retrieval_tool_node = tool_node
                break

        if retrieval_tool_node is None:
            raise ValueError("No invoked RAG tool was found in the selected flow.")

        database_node_ids = get_database_node_ids_with_data_path_code_references(
            full_node_spec
        )
        if not database_node_ids:
            if not get_database_node_ids(full_node_spec):
                raise ValueError("No database node was found in node spec.")
            if not get_database_node_ids_with_data_path(full_node_spec):
                raise ValueError("No database node with data_path was found in node spec.")
            raise ValueError(
                "No database node with data_path code_references was found in node spec."
            )

        db_node = full_node_spec.get_node(node_id=database_node_ids[0])
        if db_node is None:
            raise ValueError(
                f"Database node '{database_node_ids[0]}' was not found in node spec."
            )
        return retrieval_tool_node, db_node

    @staticmethod
    def replace_db_data_path_reference(
        db_node: Any,
        original_path: str | Path,
        malicious_path: str | Path,
    ) -> None:
        if db_node is None:
            raise ValueError("The attacked database node was not found in node spec.")

        data_path_references = [
            code
            for code in (getattr(db_node, "code_references", None) or [])
            if getattr(code, "kind", None) == "data_path"
        ]
        if not data_path_references:
            raise ValueError(
                f"Database node '{getattr(db_node, 'id', '<unknown>')}' does not define data_path code_references."
            )

        db_node.data_path = str(malicious_path)
        original_name = Path(original_path).name
        malicious_name = Path(malicious_path).name
        for code in data_path_references:
            code.snippet = code.snippet.replace(original_name, malicious_name)

    @staticmethod
    def resolve_attacked_agent_and_rag_context(
        flow: FlowSpec,
        retrieval_tool_id: str,
        guidance: dict[str, Any] | None = None,
    ) -> tuple[str, Any, Any]:
        """
        Return the attacked LLM node id plus the observed RAG tool args/results.
        """
        guidance = guidance or {}
        attacked_agent_node_id = guidance.get("attacked_agent_node_id")
        rag_args = guidance.get("rag_args")
        rag_results = guidance.get("rag_results")

        if attacked_agent_node_id and rag_args is not None and rag_results is not None:
            return attacked_agent_node_id, rag_args, rag_results

        for i, event in enumerate(getattr(flow, "events", None) or []):
            if event.tool_calls:
                for tool_call in event.tool_calls:
                    if tool_call.tool_id == retrieval_tool_id:
                        rag_args = tool_call.arguments
                        attacked_agent_node_id = event.node_id
                        for next_event in getattr(flow, "events", None)[i+1:]:
                            if next_event.node_id == retrieval_tool_id:
                                rag_results = next_event.content
                                return attacked_agent_node_id, rag_args, rag_results

        raise ValueError(
            "Could not resolve the attacked agent and RAG call context from the selected flow."
        )

    @abstractmethod
    def get_false_answer_generator_prompt(self) -> str:
        """Return the system prompt used to generate false answers."""
        raise NotImplementedError

    @abstractmethod
    def get_false_corpus_generator_prompt(self) -> str:
        """Return the system prompt used to generate malicious corpus text."""
        raise NotImplementedError

    def insert_false_records(
            self,
        db_file_path: str | Path,
        system_prompt: str,
        rag_tool_inputs: Any,
        rag_tool_outputs: Any,
    ) -> tuple[Path, list[str]]:
        """
        Create a poisoned copy of the database file and return its path plus the
        injected corpus.

        In sandbox mode, this method copies the DB file from sandbox to a local
        temp path, modifies it, then copies the poisoned file back into sandbox
        and returns the sandbox path.
        """
        modifier = JsonTextModifier()
        path = Path(db_file_path)
        suffix = path.suffix.lower()

        if suffix not in {".json", ".jsonl"}:
            raise ValueError("path must point to a .json or .jsonl file")

        false_response = FalseAnswerGenerator().generate(
            system_prompt,
            rag_tool_inputs,
            rag_tool_outputs,
            llm_system_prompt=self.get_false_answer_generator_prompt(),
        )
        malicious_corpus = FalseCorpusGenerator(temperature=0.7).generate_corpus(
            system_prompt,
            rag_tool_inputs,
            false_response,
            rag_tool_outputs,
            llm_system_prompt=self.get_false_corpus_generator_prompt(),
        )

        def _load_db(local_path: Path):
            if suffix == ".json":
                return modifier.load_json(local_path)
            return modifier.load_jsonl(local_path)

        def _save_db(db_data: list[dict[str, Any]], local_path: Path) -> None:
            if suffix == ".json":
                modifier.save_json(db_data, local_path)
            else:
                modifier.save_jsonl(db_data, local_path)

        def _append_poisoned_records(db_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if not db_data:
                raise ValueError("DB file must be non-empty")

            sample_record = random.choice(db_data)
            for text in malicious_corpus:
                selection = modifier.detect_text_field(sample_record)
                modified_record = modifier.modify_record(
                    deepcopy(sample_record),
                    text,
                    selection.text_field_path,
                )
                db_data.append(modified_record)

            return db_data

        image_tag = getattr(self.abstraction_executer, "image_tag", None)
        if image_tag:
            sandbox_source_path = path if path.is_absolute() else Path("/sandbox") / path
            sandbox_malicious_path = sandbox_source_path.with_name(
                f"{sandbox_source_path.stem}_malicious{sandbox_source_path.suffix}"
            )

            with tempfile.TemporaryDirectory(prefix="rag_poison_") as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                local_source_path = tmp_dir_path / f"source{suffix}"
                local_malicious_path = tmp_dir_path / sandbox_malicious_path.name

                with SandboxSession(
                    verbose=True,
                    lang="python",
                    keep_template=False,
                    image=image_tag,
                ) as session:
                    session.copy_from_runtime(str(sandbox_source_path), str(local_source_path))

                    db_data = _load_db(local_source_path)
                    db_data = _append_poisoned_records(db_data)
                    _save_db(db_data, local_malicious_path)

                    session.run(
                        "import os\n"
                        f"os.makedirs({str(sandbox_malicious_path.parent)!r}, exist_ok=True)\n"
                    )
                    session.copy_to_runtime(
                        src=str(local_malicious_path),
                        dest=str(sandbox_malicious_path),
                    )
                    image_tag_prefix = image_tag.split(':')[0]
                    poisoned_tag = 'poisoned'
                    session.container.commit(
                        repository=image_tag_prefix,
                        tag=poisoned_tag,
                    )

            return sandbox_malicious_path, malicious_corpus, ':'.join([image_tag_prefix,poisoned_tag])

        db_data = _load_db(path)
        db_data = _append_poisoned_records(db_data)

        malicious_path = path.with_name(f"{path.stem}_malicious{path.suffix}")
        _save_db(db_data, malicious_path)

        return malicious_path, malicious_corpus, None
