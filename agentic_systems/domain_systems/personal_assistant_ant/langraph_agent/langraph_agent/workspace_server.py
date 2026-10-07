import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("workspace_read_mcp")

WORKSPACE_DATE = "2026-09-14"


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


def _text_match(item: dict, query: str) -> bool:
    q = query.lower()
    return any(q in str(v).lower() for v in item.values())


@mcp.tool()
def search_emails(query: str, unread_only: bool = False) -> str:
    """Search emails by keyword. Set unread_only=true to return only unread emails.
    Returns matching emails with id, from, to, date, subject, body, and read status."""
    emails = _load("emails.json")
    results = emails
    if unread_only:
        results = [e for e in results if not e.get("read", True)]
    if query:
        results = [e for e in results if _text_match(e, query)]
    if not results:
        msg = "No unread emails found." if unread_only and not query else f"No emails found matching: {query}"
        return msg
    lines = []
    for e in results:
        read_status = "unread" if not e.get("read", True) else "read"
        draft_flag = " [DRAFT]" if e.get("draft") else ""
        lines.append(
            f"id: {e['id']} | from: {e['from']} | to: {e['to']} | date: {e['date']} | {read_status}{draft_flag}\n"
            f"subject: {e['subject']}\nbody: {e['body']}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def search_calendar(query: str = "", date_range: str = "") -> str:
    """Search calendar events by keyword and/or date range.
    Use date_range (format: 'YYYY-MM-DD to YYYY-MM-DD') to filter by date.
    Leave query empty to return all events in a date range.
    Each event includes start_time, end_time, attendees, and location."""
    events = _load("calendar.json")
    if query:
        q = query.lower()
        results = [
            ev for ev in events
            if q in ev["title"].lower()
            or q in ev.get("notes", "").lower()
            or q in ev.get("date", "").lower()
            or any(q in a for a in ev.get("attendees", []))
        ]
    else:
        results = list(events)
    if date_range:
        parts = [p.strip() for p in date_range.split("to")]
        if len(parts) == 2:
            start, end = parts
            results = [ev for ev in results if start <= ev["date"] <= end]
    if not results:
        label = query or date_range
        return f"No calendar events found for: {label}"
    lines = []
    for ev in results:
        attendees = ", ".join(ev.get("attendees", [])) or "none"
        lines.append(
            f"id: {ev['id']} | title: {ev['title']} | date: {ev['date']}\n"
            f"start: {ev.get('start_time', '?')} | end: {ev.get('end_time', '?')} | duration: {ev['duration_minutes']} min\n"
            f"attendees: {attendees}\n"
            f"location: {ev.get('location', '')} | notes: {ev.get('notes', '')}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def search_contacts(query: str) -> str:
    """Search contacts by name, email, role, or company."""
    contacts = _load("contacts.json")
    results = [c for c in contacts if _text_match(c, query)]
    if not results:
        return f"No contacts found matching: {query}"
    lines = []
    for c in results:
        parts = [
            f"id: {c['id']}",
            f"name: {c['name']}",
            f"email: {c['email']}",
            f"phone: {c.get('phone', '')}",
            f"role: {c.get('role', '')}",
            f"team: {c.get('team', '')}",
        ]
        if c.get("company"):
            parts.append(f"company: {c['company']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


@mcp.tool()
def search_tasks(query: str = "", status: str = "", priority: str = "") -> str:
    """Search tasks by keyword, status, and/or priority.
    status: 'pending', 'in_progress', or 'done'. priority: 'high', 'medium', or 'low'.
    Leave query empty to list all tasks (optionally filtered by status/priority).
    Note: today's date is 2026-05-03. Tasks with due_date before today and status 'pending' are overdue."""
    tasks = _load("tasks.json")
    results = list(tasks)
    if status:
        results = [t for t in results if t.get("status", "") == status]
    if priority:
        results = [t for t in results if t.get("priority", "") == priority]
    if query:
        results = [t for t in results if _text_match(t, query)]
    if not results:
        return f"No tasks found."
    lines = []
    for t in results:
        overdue = (
            t.get("status") == "pending"
            and t.get("due_date", "9999") < WORKSPACE_DATE
        )
        overdue_flag = " [OVERDUE]" if overdue else ""
        lines.append(
            f"id: {t['id']} | title: {t['title']} | due: {t['due_date']} | status: {t['status']} | priority: {t['priority']}{overdue_flag}\n"
            f"details: {t.get('details', '')}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def search_notes(query: str) -> str:
    """Search personal notes by keyword."""
    notes = _load("notes.json")
    results = [n for n in notes if _text_match(n, query)]
    if not results:
        return f"No notes found matching: {query}"
    lines = [
        f"id: {n['id']} | title: {n['title']} | created: {n['created']}\n{n['content']}"
        for n in notes if n in results
    ]
    return "\n\n".join(lines)


@mcp.tool()
def read_task_file(file_name: str) -> str:
    """Read a task file from the workspace task_files directory.
    Available files: task_001.md, task_002.md, task_003.md, task_004.md, task_005.md."""
    safe_name = Path(file_name).name
    path = _workspace_dir() / "task_files" / safe_name
    if not path.exists():
        return f"Task file not found: {safe_name}"
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run(transport="stdio")