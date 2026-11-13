"""Agent definitions: a name and the system prompt that goes with it.

Milestone 2 has one agent. Delegation between agents arrives in milestone 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    name: str
    system: str


WORKER = Agent(
    name="worker",
    system=(
        "You complete the user's task using the tools provided. All file paths are relative "
        "to the working directory, and you cannot reach outside it. The shell tool runs one "
        "allow-listed program with an argument list, without a shell, so pipes, redirection "
        "and globbing are not available. If a tool returns an error, read it and adjust "
        "rather than repeating the same call. When you are done, reply with the answer and "
        "no further tool calls."
    ),
)
