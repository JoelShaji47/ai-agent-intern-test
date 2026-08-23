"""In-memory conversation session store for multi-turn support.

Maps session_id -> list[types.Content] in the exact Gemini conversation
history format used by run_agent_turn. Plain dict, no persistence, no
expiry -- sufficient for this assignment's scope.
"""

from __future__ import annotations

from google.genai import types

_SESSIONS: dict[str, list[types.Content]] = {}


def get_or_create_session(session_id: str) -> list[types.Content]:
    """Return the history list for session_id, creating it if missing."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    return _SESSIONS[session_id]


def append_turn(session_id: str, new_contents: list[types.Content]) -> None:
    """Append one completed turn's contents (user msg, tool exchanges,
    final model response) to the stored history for session_id."""
    get_or_create_session(session_id).extend(new_contents)


def clear_session(session_id: str) -> None:
    """Drop a session entirely (testing / explicit reset)."""
    _SESSIONS.pop(session_id, None)
