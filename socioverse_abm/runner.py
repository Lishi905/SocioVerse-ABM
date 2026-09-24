"""
Unified CLI for the kernel:  `sv-abm <command>`

    sv-abm list                      list registered tasks (name, family, source,
                                     matching SocioVerse2 study)
    sv-abm run <task> --mode rule    run a task in rule | llm | hybrid mode

Tasks register themselves on import; the runner discovers them by importing the
`tasks` package (best-effort) and then consulting `scenario_engine.registry`.
Run it from the SocioVerse-ABM repository root (or with the repo installed via
`pip install -e .`) so that the `tasks` package is importable.

Exit status of `run`: 0 on success, 1 for an unknown task, 2 when the local setup is
incomplete (a missing optional extra, a missing LLM key for llm mode, or missing task
data); the message says what to install or set.

LLM key: `--mode llm` requires it up front. `--mode hybrid` does not, because its
default routing (`behavior_engine.hybrid.always_rule`) sends every agent to the rule
and never calls the LLM; without a key the runner prints a note and runs, and exits
with status 2 and the key hint if a custom `route_to_llm` policy does reach the LLM.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from socioverse_abm.scenario_engine import registry

# Optional-dependency groups: a missing top-level module -> the install hint to print.
_CHICAGO_MODULES = {"mesa", "geopandas", "libpysal", "pandas", "pyogrio", "shapely", "fiona"}
_CHICAGO_HINT = 'pip install -e ".[chicago]"   # geopandas, libpysal, mesa 3, pandas'
_CORE_HINT = "pip install -e .   # kernel dependencies: numpy, networkx, openai, pyyaml"
_CORE_MODULES = {"numpy", "networkx", "openai", "yaml"}

# Default LLM credentials for the classic tasks (socioverse_abm.behavior_engine.llm_f).
# A task may override them through its TaskSpec.meta:
#   "llm_key_envs":      env vars accepted as the API key, in priority order
#   "llm_base_url_envs": env vars accepted as the endpoint (optional)
#   "load_env":          a zero-argument callable that hydrates os.environ (e.g. from a .env)
_DEFAULT_KEY_ENVS = ("OPENAI_API_KEY",)
_DEFAULT_BASE_URL_ENVS = ("OPENAI_BASE_URL",)


def _discover_tasks() -> None:
    """Best-effort import of the tasks package so registrations populate."""
    try:
        importlib.import_module("tasks")
    except Exception:
        # `tasks` not importable (e.g. not run from the repo root); list shows nothing.
        pass


def _cmd_list(_args) -> int:
    _discover_tasks()
    tasks = registry.all_tasks()
    if not tasks:
        print("no tasks registered (run from the SocioVerse-ABM repository root)")
        return 0
    width = max(len(n) for n in tasks)
    for name in sorted(tasks):
        spec = tasks[name]
        print(f"{name:<{width}}  [{spec.family}]  {spec.source_abm}  -> {spec.socioverse2_study}")
    return 0


def _missing_llm_key(spec) -> tuple[str, ...] | None:
    """Return the accepted key names when none of them is set, else None."""
    meta = spec.meta or {}
    load_env = meta.get("load_env")
    if callable(load_env):
        try:
            load_env()
        except Exception:  # a broken .env must not mask the real message below
            pass
    names = tuple(meta.get("llm_key_envs") or _DEFAULT_KEY_ENVS)
    if any(os.environ.get(n) for n in names):
        return None
    return names


def _print_llm_key_help(spec, mode: str, key_names) -> None:
    meta = spec.meta or {}
    url_names = tuple(meta.get("llm_base_url_envs") or _DEFAULT_BASE_URL_ENVS)
    accepted = " or ".join(key_names)
    lines = [
        f"sv-abm: task '{spec.name}' in --mode {mode} needs an LLM API key; set {accepted}.",
        f"  export {key_names[0]}=...",
        f"  export {url_names[0]}=https://api.openai.com/v1   # optional; any OpenAI-compatible endpoint",
    ]
    if meta.get("load_env_hint"):
        lines.append(f"  ({meta['load_env_hint']})")
    lines.append(f"For a run without an API key use: sv-abm run {spec.name} --mode rule")
    print("\n".join(lines), file=sys.stderr)


def _is_missing_key_error(exc: BaseException) -> bool:
    """True for the OpenAI client's missing-credentials error, even when re-wrapped.

    Only consulted for a hybrid run started without a key. The client raises a bare
    ``openai.OpenAIError`` naming OPENAI_API_KEY (the wording differs across versions);
    llm_f.generate_and_parser re-raises errors as a plain Exception, hence the chain walk.
    """
    seen = 0
    while exc is not None and seen < 5:
        cls = type(exc)
        if cls.__name__ == "OpenAIError" and cls.__module__.startswith("openai"):
            return True
        if "OPENAI_API_KEY" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def _print_hybrid_no_key_note(spec, key_names) -> None:
    accepted = " or ".join(key_names)
    print(
        f"sv-abm: no LLM API key set ({accepted}). Running task '{spec.name}' in hybrid "
        "mode anyway: the default routing (always_rule) sends every agent to the rule, so "
        "the run makes no LLM call and matches --mode rule. A route_to_llm policy that "
        "sends agents to the LLM needs the key (see socioverse_abm/behavior_engine/hybrid.py).",
        file=sys.stderr,
    )


def _print_import_help(task: str, exc: ImportError) -> None:
    missing = (getattr(exc, "name", None) or "").split(".")[0]
    text = str(exc)
    lines = [f"sv-abm: task '{task}' could not run because an import failed: {text}"]
    if "socksio" in text or "SOCKS" in text:
        lines.append(
            "  A SOCKS proxy is set (ALL_PROXY / all_proxy). Either unset it or run: "
            'pip install "httpx[socks]"'
        )
    elif missing in _CHICAGO_MODULES or task == "chicago_segregation":
        lines.append("  This task needs the chicago extra. From the repository root run:")
        lines.append(f"    {_CHICAGO_HINT}")
    elif missing in _CORE_MODULES:
        lines.append(f"  '{missing}' is a kernel dependency. From the repository root run:")
        lines.append(f"    {_CORE_HINT}")
    elif missing:
        lines.append(f"  Install the missing package '{missing}' and retry.")
    print("\n".join(lines), file=sys.stderr)


def _cmd_run(args) -> int:
    _discover_tasks()
    try:
        spec = registry.get(args.task)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    if spec.run is None:
        print(f"task '{args.task}' has no run() entry registered", file=sys.stderr)
        return 1
    missing_key = _missing_llm_key(spec) if args.mode in ("llm", "hybrid") else None
    if missing_key and args.mode == "llm":
        _print_llm_key_help(spec, args.mode, missing_key)
        return 2
    if missing_key:  # hybrid: only a custom route_to_llm policy would call the LLM
        _print_hybrid_no_key_note(spec, missing_key)
    try:
        spec.run(mode=args.mode, config=args.config)
    except ImportError as exc:
        _print_import_help(args.task, exc)
        return 2
    except registry.TaskSetupError as exc:
        print(f"sv-abm: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if missing_key and _is_missing_key_error(exc):
            _print_llm_key_help(spec, args.mode, missing_key)
            return 2
        raise
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sv-abm", description="SocioVerse-ABM runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered tasks")

    run = sub.add_parser("run", help="run a task")
    run.add_argument("task", help="registered task name (see `sv-abm list`)")
    run.add_argument("--mode", choices=["rule", "llm", "hybrid"], default="rule")
    run.add_argument("--config", default=None, help="path to the task's config.yaml")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "run":
        return _cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
