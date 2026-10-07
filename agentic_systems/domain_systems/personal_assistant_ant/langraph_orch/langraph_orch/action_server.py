import json
import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("workspace_action_mcp")


def _workspace_dir() -> Path:
    val = os.getenv("PA_WORKSPACE_DIR")
    if not val:
        raise RuntimeError("PA_WORKSPACE_DIR is not set — never run MCP servers directly; always start via main.py.")
    d = Path(val)
    if not d.exists():
        raise FileNotFoundError(f"Workspace not found at {d}.")
    return d


def _load(filename: str) -> list:
    path = _workspace_dir() / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save(filename: str, data: list) -> None:
    (_workspace_dir() / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _record_action(action_type: str, details: dict) -> None:
    actions = _load("actions.json")
    actions.append({"action": action_type, "timestamp": datetime.utcnow().isoformat() + "Z", **details})
    _save("actions.json", actions)


@mcp.tool()
def create_task(title: str, due_date: str, details: str = "") -> str:
    """Create a new task in the workspace."""
    tasks = _load("tasks.json")
    new_id = f"task{str(len(tasks) + 1).zfill(3)}"
    task = {
        "id": new_id, "title": title, "due_date": due_date,
        "status": "pending", "priority": "medium", "details": details,
    }
    tasks.append(task)
    _save("tasks.json", tasks)
    _record_action("create_task", {"task_id": new_id, "title": title})
    return f"Task created: id={new_id}, title={title}, due={due_date}"


@mcp.tool()
def update_task(task_id: str, status: str = "", priority: str = "", title: str = "", due_date: str = "", details: str = "") -> str:
    """Update one or more fields of a task. Provide only the fields you want to change.
    status: pending, in_progress, done. priority: low, medium, high. Leave others empty."""
    tasks = _load("tasks.json")
    update_dict = {k: v for k, v in {"status": status, "priority": priority, "title": title, "due_date": due_date, "details": details}.items() if v}
    if not update_dict:
        return "No fields provided to update."
    for task in tasks:
        if task["id"] == task_id:
            task.update(update_dict)
            _save("tasks.json", tasks)
            _record_action("update_task", {"task_id": task_id, "updates": update_dict})
            return f"Task updated: id={task_id}, changed={list(update_dict.keys())}"
    return f"Task not found: {task_id}"


@mcp.tool()
def create_calendar_event(title: str, date: str, time: str, attendees: str = "", details: str = "") -> str:
    """Create a new calendar event. attendees is a comma-separated list of emails."""
    events = _load("calendar.json")
    new_id = f"cal{str(len(events) + 1).zfill(3)}"
    attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]
    event = {
        "id": new_id, "title": title, "date": date, "time": time,
        "duration_minutes": 60, "attendees": attendee_list,
        "location": "", "notes": details,
    }
    events.append(event)
    _save("calendar.json", events)
    _record_action("create_calendar_event", {"event_id": new_id, "title": title, "date": date})
    return f"Calendar event created: id={new_id}, title={title}, date={date}, time={time}"


@mcp.tool()
def update_calendar_event(event_id: str, title: str = "", date: str = "", time: str = "", attendees: str = "", details: str = "") -> str:
    """Update one or more fields of a calendar event. Provide only the fields you want to change.
    attendees: comma-separated list of emails. Leave others empty."""
    events = _load("calendar.json")
    update_dict = {k: v for k, v in {"title": title, "date": date, "time": time, "notes": details}.items() if v}
    if attendees:
        update_dict["attendees"] = [a.strip() for a in attendees.split(",") if a.strip()]
    if not update_dict:
        return "No fields provided to update."
    for event in events:
        if event["id"] == event_id:
            event.update(update_dict)
            _save("calendar.json", events)
            _record_action("update_calendar_event", {"event_id": event_id, "updates": update_dict})
            return f"Calendar event updated: id={event_id}, changed={list(update_dict.keys())}"
    return f"Event not found: {event_id}"


@mcp.tool()
def draft_email(to: str, subject: str, body: str) -> str:
    """Draft an email. Creates a local draft only — no email is sent."""
    emails = _load("emails.json")
    new_id = f"draft{str(len(emails) + 1).zfill(3)}"
    draft = {
        "id": new_id, "from": "me@example.com", "to": to,
        "subject": subject, "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "body": body, "thread_id": None, "read": True, "draft": True,
    }
    emails.append(draft)
    _save("emails.json", emails)
    _record_action("draft_email", {"draft_id": new_id, "to": to, "subject": subject})
    return f"Email draft created: id={new_id}, to={to}, subject={subject}"


if __name__ == "__main__":
    mcp.run(transport="stdio")