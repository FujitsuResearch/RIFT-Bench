import json
import shlex
from typing import Any


def build_command(query: str, command_template: str, quote_query: bool = True) -> str:
    query_value = shlex.quote(query) if quote_query else query
    return command_template.format(query=query_value)


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_response_candidate(span: dict[str, Any]) -> Any | None:
    attributes = span.get("attributes", {})
    raw_outputs = attributes.get("mlflow.spanOutputs")
    if raw_outputs in (None, ""):
        return None

    span_type = _safe_json_loads(attributes.get("mlflow.spanType", '"UNKNOWN"'))
    outputs = _safe_json_loads(raw_outputs)

    if span_type == "AGENT" and isinstance(outputs, dict):
        message = outputs.get("message")
        if isinstance(message, dict) and message.get("role") == "function":
            return message.get("content")
        if isinstance(message, dict):
            return message.get("content")
        return message

    if span_type == "TOOL":
        return outputs

    if span_type == "LLM" and isinstance(outputs, dict):
        choices = outputs.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")

    return outputs


def _populate_missing_trace_response(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    data = payload.get("data")
    if not isinstance(data, dict) or data.get("response") not in (None, ""):
        return payload

    spans = data.get("spans")
    if not isinstance(spans, list):
        return payload

    response = None
    for span in reversed(spans):
        candidate = _extract_response_candidate(span)
        if candidate in (None, "", "TERMINATE"):
            continue
        response = json.dumps(candidate, ensure_ascii=False)
        break

    if response is None:
        return payload

    data["response"] = response

    info = payload.get("info")
    if isinstance(info, dict):
        request_metadata = info.get("request_metadata")
        if isinstance(request_metadata, dict) and not request_metadata.get(
            "mlflow.traceOutputs"
        ):
            request_metadata["mlflow.traceOutputs"] = response

    return payload
