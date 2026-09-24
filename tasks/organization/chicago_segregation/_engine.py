"""ChicagoEngine — owns the vendored SegregationModel and the import seam.

This is the ONLY place that imports the vendored legacy code (``legacy/src``). It
builds the validated ``SegregationModel`` once, (optionally) swaps in a fake LLM
client for offline / parity runs, and assigns each HouseholdAgent a stable integer
id so the kernel's Observation/Action stream has a deterministic order.

The env (``env.py``) and the behavior functions (``llm_f.py`` / ``rule_f.py``) all
share ONE engine instance via ``set_active`` / ``get_active`` — they drive the same
model, exactly as legacy ``step()`` does, just split across the kernel's
observe → decide → apply → snapshot loop.

Design mirrors SocioVerse2's ``studies/chicago_schelling/adapter/engine_seam.py``,
which drives this same vendored copy when SocioVerse-ABM is checked out next to
SocioVerse2.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from socioverse_abm.scenario_engine.registry import TaskSetupError

# Vendored legacy project root: tasks/organization/chicago_segregation/legacy/
# (contains src/, config/, processed_data/, run_prototype.py — see PROVENANCE.md).
_LEGACY_ROOT = Path(__file__).resolve().with_name("legacy")
# SocioVerse-ABM repo root (…/SocioVerse-ABM) — used only for the optional .env.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def legacy_root() -> Path:
    """Locate the vendored Chicago code (the directory holding ``src/model.py``).

    Resolution order:
      1. $SV_CHICAGO_LEGACY — an explicit Chicago legacy directory;
      2. $SV_ABM_ROOT — a SocioVerse-ABM checkout root (its vendored legacy dir is
         used); for backward compatibility a value that already points at a legacy
         directory (one containing ``src/model.py``) is accepted as-is;
      3. the copy vendored next to this file.

    The vendored data (``legacy/processed_data/``, ``legacy/config/``) is not packaged, so
    a regular (non-editable) install has the code but not the data. In that case step 3
    raises ``TaskSetupError`` (a ``RuntimeError``) that says how to point at a checkout.
    """
    env = os.environ.get("SV_CHICAGO_LEGACY")
    if env:
        return Path(env)
    abm_root = os.environ.get("SV_ABM_ROOT")
    if abm_root:
        root = Path(abm_root)
        if (root / "src" / "model.py").exists():
            return root
        nested = root / "tasks" / "organization" / "chicago_segregation" / "legacy"
        if (nested / "src" / "model.py").exists():
            return nested
    if not (_LEGACY_ROOT / "processed_data" / "chicago_tracts.geojson").exists():
        raise TaskSetupError(
            "Chicago data not found: "
            f"{_LEGACY_ROOT / 'processed_data' / 'chicago_tracts.geojson'} is missing "
            "(a regular pip install does not ship the vendored data). Use a git checkout "
            "with an editable install (pip install -e \".[chicago]\"), or set "
            "SV_CHICAGO_LEGACY=<checkout>/tasks/organization/chicago_segregation/legacy."
        )
    return _LEGACY_ROOT


def _ensure_on_path() -> None:
    root = str(legacy_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_dotenv(root: Path = _REPO_ROOT) -> None:
    """Minimal dependency-free .env loader; only sets keys not already in os.environ."""
    p = root / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def default_llm_config(model: Optional[str] = None, **overrides: Any):
    """Build a live LLMConfig from the ENVIRONMENT (never hard-code secrets).

    Required: SV_LLM_API_KEY (or OPENAI_API_KEY). Optional: SV_LLM_BASE_URL (or
    OPENAI_BASE_URL; default: the official OpenAI endpoint) and SV_LLM_MODEL.
    Values may live in SocioVerse-ABM/.env (gitignored). For a no-token run, pass a
    DeterministicLLMClient via ``llm_client=`` instead.
    """
    _ensure_on_path()
    from src.llm_client import LLMConfig  # type: ignore

    _load_dotenv()
    api_key = os.environ.get("SV_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise TaskSetupError(
            "No LLM API key found. Set SV_LLM_API_KEY (or OPENAI_API_KEY) in your "
            "environment or in SocioVerse-ABM/.env. For a no-token run use "
            "--mode rule, or pass llm_client=DeterministicLLMClient() (see _fake_llm.py)."
        )
    params = dict(
        api_key=api_key,
        base_url=(
            os.environ.get("SV_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ),
        model=model or os.environ.get("SV_LLM_MODEL", "gpt-4o-2024-08-06"),
        temperature=0.7,
        max_tokens=512,
    )
    params.update(overrides)
    return LLMConfig(**params)


class ChicagoEngine:
    """Holds the SegregationModel + a stable agent index, shared by env + f."""

    def __init__(
        self,
        *,
        scale: str = "small",
        tract_ids: Optional[list[str]] = None,
        init_mode: str = "census",
        seed: int = 42,
        model_kwargs: Optional[dict] = None,
        llm_client: Any = None,   # a fake client (offline / parity); swapped in after build
        llm_config: Any = None,   # a real LLMConfig (live runs)
    ):
        self.scale = scale
        self._tract_ids_arg = tract_ids
        self.init_mode = init_mode
        self.seed = seed
        self.model_kwargs = dict(model_kwargs or {})
        self._llm_client = llm_client
        self._llm_config = llm_config

        self.model = None
        # Stable ordering: agents in creation order, each with an int id == its index.
        self.agent_order: list[Any] = []
        self._agent_by_id: dict[int, Any] = {}

    # --- build (idempotent so env.reset / build() can both trigger it) ---
    def ensure_built(self):
        if self.model is not None:
            return self.model
        _ensure_on_path()
        from src.model import SegregationModel  # type: ignore

        tract_ids = self._resolve_tract_ids()
        if self._llm_config is not None:
            llm_config = self._llm_config
        else:
            # rule mode and fake-client runs never call the LLM, so a dummy config is
            # enough to construct LLMClient (no network happens at construction time).
            from src.llm_client import LLMConfig  # type: ignore

            llm_config = LLMConfig(
                api_key="unused-offline", base_url="http://localhost", model="offline"
            )

        self.model = SegregationModel(
            llm_config=llm_config,
            tract_ids=tract_ids,
            seed=self.seed,
            init_mode=self.init_mode,
            **self.model_kwargs,
        )
        if self._llm_client is not None:
            # __init__ never calls the LLM (assess/evaluate happen in the phases), so
            # swapping the client in after construction is safe and total.
            self.model.llm = self._llm_client

        self._index_agents()
        return self.model

    @property
    def has_llm(self) -> bool:
        """True when built with a live LLMConfig or an injected (fake) client."""
        return self._llm_config is not None or self._llm_client is not None

    def _resolve_tract_ids(self) -> Optional[list[str]]:
        if self._tract_ids_arg is not None:
            return list(self._tract_ids_arg)
        if self.scale == "full":
            return None  # full city — SegregationModel uses all tracts
        _ensure_on_path()
        from run_prototype import select_prototype_tracts  # type: ignore

        return select_prototype_tracts(scale=self.scale)

    def _index_agents(self) -> None:
        self.agent_order = list(self.model.agents)
        self._agent_by_id = {}
        for i, agent in enumerate(self.agent_order):
            agent._sv_id = i               # int id for the kernel Observation/Action stream
            self._agent_by_id[i] = agent

    def agent(self, agent_id: int):
        return self._agent_by_id[agent_id]


# ── active-engine handshake ─────────────────────────────────────────────────
# The kernel calls the registered rule_f / llm_f as bare functions; they read the
# engine that build()/run() most recently activated (one study per process — the
# legacy globals are not parallel-safe anyway).
_ACTIVE: Optional[ChicagoEngine] = None


def set_active(engine: ChicagoEngine) -> None:
    global _ACTIVE
    _ACTIVE = engine


def get_active() -> ChicagoEngine:
    if _ACTIVE is None:
        raise RuntimeError(
            "chicago_segregation behavior function called before the engine was built. "
            "Drive the task via model.run()/build() (which activate the engine)."
        )
    return _ACTIVE
