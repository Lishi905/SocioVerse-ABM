"""Chicago residential-segregation task (organization family).

Real-geography extension of the Schelling segregation model: 781 Chicago census
tracts (Queen contiguity), an archetype household population, and B = f(P, E) where
f is either the classical similarity-threshold rule or the validated LLM behavior
function. Wraps the vendored legacy `SegregationModel` (see ``legacy/PROVENANCE.md``)
without forking its dynamics.
"""
from . import model  # noqa: F401  — import side effect: registry.register(...)
