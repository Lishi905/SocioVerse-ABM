"""
P — the SIR population: N network nodes and the interaction structure (graph).

A "persona" here is intentionally light (id + health + optional demographic attrs).
The attrs slot is the hook for `socioverse_abm.user_engine.sampling` to enrich agents
with demographics for the LLM behavior function; the rule path does not need it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import networkx as nx

# Health states (rumor analogy of Susceptible / Infected / Removed).
S, I, R = "S", "I", "R"
HEALTH_LABEL = {S: "Ignorant", I: "Spreader", R: "Stifler"}


@dataclass
class SIRPersona:
    agent_id: int
    health: str = S
    attrs: Dict[str, Any] = field(default_factory=dict)  # optional user_engine enrichment


@dataclass
class Population:
    graph: nx.Graph
    personas: List[SIRPersona]

    def __len__(self) -> int:
        return len(self.personas)

    def neighbors(self, agent_id: int):
        return self.graph.neighbors(agent_id)


def build_graph(n: int, topology: str, params: dict, seed: int) -> nx.Graph:
    if topology == "ER":
        g = nx.erdos_renyi_graph(n, params["er_mean_degree"] / (n - 1), seed=seed)
    elif topology == "BA":
        g = nx.barabasi_albert_graph(n, params["ba_m"], seed=seed)
    elif topology == "WS":
        g = nx.watts_strogatz_graph(n, params["ws_k"], params["ws_p"], seed=seed)
    else:
        raise ValueError(f"unknown topology '{topology}' (ER|BA|WS)")
    # Keep the largest connected component, relabel 0..n-1 (legacy behavior).
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        g = nx.convert_node_labels_to_integers(g)
    return g


def build_population(cfg: dict, seed: int | None = None) -> Population:
    seed = cfg["seed"] if seed is None else seed
    pcfg = cfg["population"]
    g = build_graph(pcfg["n"], pcfg["topology"], pcfg["topology_params"], seed)
    personas = [SIRPersona(agent_id=i) for i in g.nodes()]
    # Seed the initial spreaders deterministically.
    rng = random.Random(seed)
    k = min(pcfg["initial_spreaders"], g.number_of_nodes())
    for node in rng.sample(list(g.nodes()), k):
        personas[node].health = I
    return Population(graph=g, personas=personas)
