"""Multi-turn conversation verification CLI (Phase 6).

Runs two shared-history scenarios plus two session-isolation probes against
the live agent, printing every turn's full AgentResponse.
"""

from __future__ import annotations

import json
import time

from app.agent import run_agent_turn
from app.session import clear_session

_DELAY_SECONDS = 3

_SCENARIOS: list[tuple[str, str, list[str]]] = [
    ("Scenario A", "test-shipping", [
        "Do you ship internationally?",
        "What about Canada, and how long does it take?",
    ]),
    ("Scenario B", "test-order", [
        "Where is ORD-1007?",
        "When will it arrive?",
    ]),
    ("Isolation 1 (Canada, fresh session)", "test-isolation", [
        "What about Canada?",
    ]),
    ("Isolation 2 (pronoun, fresh session)", "test-isolation-2", [
        "When will it arrive?",
    ]),
]


def main() -> None:
    for index, (label, session_id, messages) in enumerate(_SCENARIOS):
        if index > 0:
            time.sleep(_DELAY_SECONDS)
        clear_session(session_id)
        print(f"===== {label} (session={session_id!r}) =====")
        for turn_number, message in enumerate(messages, start=1):
            print(f"--- turn {turn_number}: {message}")
            print(run_agent_turn(message, session_id=session_id).model_dump_json(indent=2))
            if turn_number < len(messages):
                time.sleep(_DELAY_SECONDS)
        print()


if __name__ == "__main__":
    main()
