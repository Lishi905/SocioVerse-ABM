#!/usr/bin/env python3
"""Headless evaluator for Boids in LLM mode.
Runs one or multiple simulations, records metrics each step, and outputs
step-by-step statistics plus averages; supports OpenAI or Qwen backends.
"""

import argparse
import csv
import math
import os
import random
import sys
import types
from typing import List, Optional

import numpy as np

from boids_core import BoidsEngine

_QWEN_PATCHED = False
_DEEPSEEK_PATCHED = False


def compute_additional_metrics(engine: BoidsEngine):
    if not engine.boids:
        return 0.0, 0.0
    xs = np.array([b.x for b in engine.boids], dtype=float)
    ys = np.array([b.y for b in engine.boids], dtype=float)
    cx = float(xs.mean())
    cy = float(ys.mean())
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mean_dist_to_centroid = float(dists.mean())
    speeds = np.array([math.hypot(b.vx, b.vy) for b in engine.boids], dtype=float)
    mean_speed = float(speeds.mean()) if speeds.size else 0.0
    return mean_dist_to_centroid, mean_speed


def compute_alignment_order(engine: BoidsEngine) -> float:
    if not engine.boids:
        return 0.0
    velocities = np.array([[b.vx, b.vy] for b in engine.boids], dtype=float)
    norms = np.linalg.norm(velocities, axis=1)
    norms[norms == 0] = 1.0
    directions = velocities / norms[:, None]
    avg_direction = directions.mean(axis=0)
    return float(np.linalg.norm(avg_direction))


def compute_mean_nnd(engine: BoidsEngine) -> float:
    positions = np.array([[b.x, b.y] for b in engine.boids], dtype=float)
    if len(positions) < 2:
        return 0.0
    min_dists: List[float] = []
    for i, pos in enumerate(positions):
        diff = positions - pos
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        min_dists.append(dist.min())
    return float(np.mean(min_dists))


def setup_qwen():
    global _QWEN_PATCHED
    if _QWEN_PATCHED:
        return

    qwen_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not qwen_key:
        print("✗ ERROR: QWEN_API_KEY or DASHSCOPE_API_KEY not found in environment!")
        return
    os.environ["QWEN_API_KEY"] = qwen_key
    os.environ["OPENAI_API_KEY"] = qwen_key

    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        os.system("pip install dashscope -q")
        import dashscope  # type: ignore
        from dashscope import Generation  # type: ignore

    dashscope.api_key = qwen_key  # type: ignore

    import openai

    def _qwen_wrapper(*args, **kwargs):
        kwargs['model'] = 'qwen3-235b-a22b-instruct-2507'
        messages = kwargs.get('messages', [])
        max_tokens = kwargs.get('max_tokens', 150)
        temperature = kwargs.get('temperature', 0.7)

        response = Generation.call(  # type: ignore
            model=kwargs['model'],
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response.status_code != 200:  # type: ignore
            raise RuntimeError(f"DashScope API error: {response.status_code} - {response.message}")  # type: ignore

        content = (
            response.output.text  # type: ignore
            if hasattr(response.output, 'text')  # type: ignore
            else str(response.output)  # type: ignore
        )

        class MockChoice:
            def __init__(self, content_text: str):
                self.message = types.SimpleNamespace(content=content_text)
                self.finish_reason = 'stop'

        class MockResponse:
            def __init__(self, content_text: str):
                self.choices = [MockChoice(content_text)]

        return MockResponse(content)

    openai.chat.completions.create = _qwen_wrapper
    _QWEN_PATCHED = True


def setup_deepseek():
    global _DEEPSEEK_PATCHED
    if _DEEPSEEK_PATCHED:
        return

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("✗ ERROR: DEEPSEEK_API_KEY not found in environment!")
        return
    os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    os.environ["OPENAI_API_KEY"] = deepseek_key

    import openai

    original_create = openai.chat.completions.create

    def _deepseek_wrapper(*args, **kwargs):
        kwargs['model'] = 'deepseek-chat'
        try:
            deepseek_client = openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1",
            )
            return deepseek_client.chat.completions.create(*args, **kwargs)
        except Exception as exc:
            print(f"Warning: DeepSeek client issue, falling back to original OpenAI call: {exc}")
            return original_create(*args, **kwargs)

    openai.chat.completions.create = _deepseek_wrapper
    _DEEPSEEK_PATCHED = True


def run_single(
    seed: Optional[int] = None,
    use_qwen: bool = False,
    use_deepseek: bool = False,
    label: Optional[str] = None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if use_qwen:
        setup_qwen()
    if use_deepseek:
        setup_deepseek()

    engine = BoidsEngine(
        width=800,
        height=600,
        num_boids=50,
        mode="llm",
        scenario="open_field",
    )
    engine.backend = "openai"
    engine.debug = True
    engine.debug_missing = False
    engine.freeze_on_missing_action = False
    engine.llm_decision_tick_interval = 20

    steps = 100
    os.makedirs("bench_exports", exist_ok=True)
    suffix = label or ("qwen" if use_qwen else "openai")
    export_path = os.path.join(
        "bench_exports",
        f"boids_llm_eval_seed_{seed if seed is not None else 'default'}_{suffix}.csv",
    )

    alignment_hist: List[float] = []
    nnd_hist: List[float] = []
    centroid_hist: List[float] = []
    speed_hist: List[float] = []

    with open(export_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "alignment_order", "mean_nnd", "mean_dist_to_centroid", "mean_speed"])

        for step in range(1, steps + 1):
            engine.step()
            alignment = compute_alignment_order(engine)
            mean_nnd = compute_mean_nnd(engine)
            mean_dist_to_centroid, mean_speed = compute_additional_metrics(engine)

            alignment_hist.append(alignment)
            nnd_hist.append(mean_nnd)
            centroid_hist.append(mean_dist_to_centroid)
            speed_hist.append(mean_speed)

            writer.writerow([
                step,
                f"{alignment:.3f}",
                f"{mean_nnd:.3f}",
                f"{mean_dist_to_centroid:.3f}",
                f"{mean_speed:.3f}",
            ])

        writer.writerow([
            "mean",
            f"{float(np.mean(alignment_hist)):.3f}",
            f"{float(np.mean(nnd_hist)):.3f}",
            f"{float(np.mean(centroid_hist)):.3f}",
            f"{float(np.mean(speed_hist)):.3f}",
        ])

    return {
        "alignment": float(np.mean(alignment_hist)),
        "nnd": float(np.mean(nnd_hist)),
        "centroid": float(np.mean(centroid_hist)),
        "speed": float(np.mean(speed_hist)),
        "export_path": export_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-driven Boids simulation")
    parser.add_argument("--runs", type=int, default=1, help="Number of random-seed runs")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen (DashScope) backend instead of OpenAI")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek backend")
    parser.add_argument("--seed", type=int, nargs="*", help="Specific seeds to run (overrides --runs)")
    args = parser.parse_args()

    seeds: List[int]
    if args.seed:
        seeds = args.seed
    else:
        seeds = [random.randint(0, 1_000_000) for _ in range(args.runs)]

    results = []
    if args.qwen and args.deepseek:
        parser.error("Please choose at most one backend flag (--qwen or --deepseek)")

    if args.qwen:
        label = "qwen"
    elif args.deepseek:
        label = "deepseek"
    else:
        label = "openai"

    print(f"Running {len(seeds)} evaluation run(s) using {label} backend...")
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n▶ Run {idx}/{len(seeds)} (seed={seed})")
        res = run_single(
            seed=seed,
            use_qwen=args.qwen,
            use_deepseek=args.deepseek,
            label=label,
        )
        results.append(res)
        print(
            f"  alignment={res['alignment']:.3f}, nn_dist={res['nnd']:.3f}, "
            f"centroid={res['centroid']:.3f}, speed={res['speed']:.3f}, file={res['export_path']}"
        )

    if results:
        avg_alignment = sum(r['alignment'] for r in results) / len(results)
        avg_nnd = sum(r['nnd'] for r in results) / len(results)
        avg_centroid = sum(r['centroid'] for r in results) / len(results)
        avg_speed = sum(r['speed'] for r in results) / len(results)
        print(
            f"\nSummary averages across {len(results)} run(s): "
            f"alignment={avg_alignment:.3f}, nn_dist={avg_nnd:.3f}, "
            f"centroid={avg_centroid:.3f}, speed={avg_speed:.3f}"
        )


if __name__ == "__main__":
    main()
