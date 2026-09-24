"""
Typed I/O for the behavior function f: B = f(P, E).

These three dataclasses are the *only* contract a task's behavior function must
speak, regardless of whether the underlying f is rule-based, LLM-based or hybrid.
They also line up 1:1 with the SocioVerse2 abc layer used at integration time:

    DecisionModel.decide_batch(obs: list[Observation], memories) -> list[Action]

so a task written against this kernel can be plugged into SocioVerse2 with zero glue.

Dependency-free on purpose (stdlib only) so `import socioverse_abm` never drags in
openai.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping


@dataclass
class Observation:
    """What a single agent perceives at step t — the (P, E) projection fed to f.

    state    : the agent's own persona/internal state (the P slice it can see).
    context  : the local environment slice (neighbors, field values, prices…) (E).
    rendered : optional natural-language rendering of `context`, used by the
               LLM behavior function (the companion ABM study's
               "psychological field" text).
    """
    agent_id: int
    state: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    rendered: str = ""


@dataclass
class Action:
    """A single agent's behavior output B at step t.

    kind     : discrete action label (e.g. "move", "cooperate", "infected").
    payload  : structured parameters of the action (e.g. {"to": (3, 4)}).
    raw      : the un-parsed model output, kept for parity tests / debugging.
    """
    agent_id: int
    kind: str
    payload: MutableMapping[str, Any] = field(default_factory=dict)
    raw: Any = None

    def matches(self, other: "Action") -> bool:
        """Action-level equality used by the consistency score (kind + payload)."""
        return self.kind == other.kind and dict(self.payload) == dict(other.payload)


@dataclass
class Decision:
    """A batch result: the actions plus optional per-agent diagnostics.

    Returned by hybrid/llm paths that want to surface which engine produced each
    action (for routing analysis) without changing the core Action contract.
    """
    actions: list[Action] = field(default_factory=list)
    source: list[str] = field(default_factory=list)  # "rule" | "llm" per action

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)
