#!/usr/bin/env python3
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
每步必须使用LLM + 等待所有决策完成 + 分批并发策略
Created: 2025-12-04
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import types
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

from boids_core import BoidsEngine
from evaluate_boids_llm import (
    setup_qwen, setup_deepseek,
    compute_alignment_order, compute_mean_nnd, compute_additional_metrics
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def request_single_llm_decision(engine: BoidsEngine, boid_id: int) -> tuple:
    """
    为单个boid请求LLM决策（用于并发处理）
    同时限制speed_modulation范围，防止速度异常
    
    Returns:
        (boid_id, action_dict) 或 (boid_id, {"__error__": error_message})
    """
    boid = next((b for b in engine.boids if b.id == boid_id), None)
    if boid is None:
        return (boid_id, {"__error__": "Boid not found"})
    
    try:
        observation = engine._build_observation(boid)
        action = engine.llm_policy.decide(observation)
        
        # 限制speed_modulation范围：1.5-2.5
        if "speed_modulation" in action:
            speed_mod = action["speed_modulation"]
            if isinstance(speed_mod, (int, float)):
                action["speed_modulation"] = max(1.5, min(2.5, float(speed_mod)))
            else:
                action["speed_modulation"] = 2.0
        else:
            action["speed_modulation"] = 2.0
        
        return (boid_id, action)
    except Exception as e:
        return (boid_id, {"__error__": str(e)})


def request_llm_decision_batch(engine: BoidsEngine, boid_ids: List[int], batch_num: int, total_batches: int, max_workers: Optional[int] = None) -> Dict[int, Any]:
    """
    为一批boids请求LLM决策（真正的并发处理）
    
    Args:
        engine: BoidsEngine实例
        boid_ids: 要处理的boid ID列表
        batch_num: 当前批次编号
        total_batches: 总批次数
        max_workers: 最大并发线程数（默认使用batch_size）
    """
    results = {}
    
    if engine.llm_policy is None:
        # 导入OpenAIPolicy（在boids_core.py中定义）
        import sys
        import importlib
        boids_core_module = sys.modules.get('boids_core')
        if boids_core_module is None:
            import boids_core
            boids_core_module = boids_core
        OpenAIPolicy = boids_core_module.OpenAIPolicy
        engine.llm_policy = OpenAIPolicy()
        engine.llm_performance_stats["policy_created"] = True
    
    # 使用ThreadPoolExecutor实现真正的并发
    if max_workers is None:
        max_workers = len(boid_ids)  # 默认使用batch_size作为并发数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_boid = {
            executor.submit(request_single_llm_decision, engine, boid_id): boid_id
            for boid_id in boid_ids
        }
        
        # 收集结果
        for future in as_completed(future_to_boid):
            boid_id = future_to_boid[future]
            try:
                result_boid_id, action = future.result()
                results[result_boid_id] = action
                
                if isinstance(action, dict) and "__error__" in action:
                    engine.llm_performance_stats["errors_encountered"] += 1
                    if engine.debug:
                        print(f"[BATCH {batch_num}/{total_batches}] Error for boid {boid_id}: {action['__error__']}")
                else:
                    engine.llm_performance_stats["responses_received"] += 1
            except Exception as e:
                results[boid_id] = {"__error__": str(e)}
                engine.llm_performance_stats["errors_encountered"] += 1
                if engine.debug:
                    print(f"[BATCH {batch_num}/{total_batches}] Exception for boid {boid_id}: {e}")
    
    return results


def wait_for_all_llm_decisions(engine: BoidsEngine, batch_size: int = 20, max_workers_per_batch: Optional[int] = None) -> None:
    """
    等待所有boids的LLM决策完成（分批并发）
    
    Args:
        engine: BoidsEngine实例
        batch_size: 每批并发的数量（默认20，避免触发rate limit）
    """
    all_boid_ids = [b.id for b in engine.boids]
    total_boids = len(all_boid_ids)
    total_batches = (total_boids + batch_size - 1) // batch_size  # 向上取整
    
    all_results = {}
    
    # 分批处理
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_boids)
        batch_ids = all_boid_ids[start_idx:end_idx]
        
        if engine.debug:
            print(f"[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_ids)} boids...")
        
        # 并发处理当前批次（等待完成）
        batch_results = request_llm_decision_batch(engine, batch_ids, batch_idx + 1, total_batches, max_workers=max_workers_per_batch)
        all_results.update(batch_results)
        
        engine.llm_performance_stats["requests_sent"] += len(batch_ids)
    
    # 将所有结果存入llm_responses（不使用缓存）
    engine.llm_responses.clear()  # 清除旧结果
    engine.llm_responses.update(all_results)
    
    # 清除缓存，确保不使用缓存的决策
    engine.llm_decision_cache._store.clear()


def run_single_per_step(
    seed: Optional[int] = None,
    use_qwen: bool = False,
    use_deepseek: bool = False,
    label: Optional[str] = None,
    batch_size: int = 20,
    steps: int = 100,
    max_workers_per_batch: Optional[int] = None,
):
    """
    运行单次评估：每步都使用LLM，等待所有决策完成后再更新
    
    Args:
        seed: 随机种子
        use_qwen: 是否使用Qwen
        use_deepseek: 是否使用DeepSeek
        label: 标签
        batch_size: 每批并发的boids数量
        steps: 总步数
    """
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
    engine.debug = False
    engine.debug_missing = False
    engine.freeze_on_missing_action = True
    engine.llm_decision_tick_interval = 1
    
    for boid in engine.boids:
        boid.max_speed = 2.5
    
    engine.llm_decision_cache.set_persistence(0)
    
    os.makedirs("bench_exports", exist_ok=True)
    suffix = label or ("qwen" if use_qwen else "openai")
    export_path = os.path.join(
        "bench_exports",
        f"boids_llm_per_step_seed_{seed if seed is not None else 'default'}_{suffix}.csv",
    )
    
    alignment_hist: List[float] = []
    nnd_hist: List[float] = []
    centroid_hist: List[float] = []
    speed_hist: List[float] = []
    step_times: List[float] = []  # 记录每步的时间
    
    # 记录每个agent的LLM决策（用于分析离散性和一致性）
    all_llm_decisions: Dict[int, Dict[str, Any]] = {}  # {step: {agent_id: decision}}
    
    print(f"Running per-step LLM evaluation (seed={seed}, steps={steps}, batch_size={batch_size})...")
    print(f"  - Total API calls: {steps * len(engine.boids):,}")
    print(f"  - Estimated time: ~{steps * (len(engine.boids) / batch_size) * 2.0 / 60:.1f} minutes")
    
    start_time = time.time()
    
    with open(export_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "alignment_order", "mean_nnd", "mean_dist_to_centroid", "mean_speed", "step_time_sec"])
        
        # 使用tqdm显示进度
        iterator = range(1, steps + 1)
        if tqdm:
            iterator = tqdm(iterator, desc="Simulation", unit="step")
        
        for step in iterator:
            step_start = time.time()
            
            # 1. 等待所有LLM决策完成（分批并发）
            wait_for_all_llm_decisions(engine, batch_size=batch_size, max_workers_per_batch=max_workers_per_batch)
            
            # 1.5. 记录所有LLM决策（用于分析离散性和一致性）
            step_decisions = {}
            for boid_id, action in engine.llm_responses.items():
                if isinstance(action, dict) and "__error__" not in action:
                    step_decisions[str(boid_id)] = {
                        'velocity_change': action.get('velocity_change', [0.0, 0.0]),
                        'speed_modulation': action.get('speed_modulation', 1.0),
                        'confidence': action.get('confidence', 1.0)
                    }
            all_llm_decisions[step] = step_decisions
            
            # 2. 更新所有boids状态（纯LLM决策，无传统对齐力参与）
            for boid in engine.boids:
                engine._update_boid(boid)
            
            # 3. 更新模拟时间
            engine.time += engine.dt
            engine.step_count += 1
            
            # 4. 计算metrics
            alignment = compute_alignment_order(engine)
            mean_nnd = compute_mean_nnd(engine)
            mean_dist_to_centroid, mean_speed = compute_additional_metrics(engine)
            
            alignment_hist.append(alignment)
            nnd_hist.append(mean_nnd)
            centroid_hist.append(mean_dist_to_centroid)
            speed_hist.append(mean_speed)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # 5. 写入CSV
            writer.writerow([
                step,
                f"{alignment:.3f}",
                f"{mean_nnd:.3f}",
                f"{mean_dist_to_centroid:.3f}",
                f"{mean_speed:.3f}",
                f"{step_time:.2f}",
            ])
            
            # 6. 清除已使用的LLM响应，确保下一步使用新的决策
            engine.llm_responses.clear()
        
        # 写入平均值
        writer.writerow([
            "mean",
            f"{float(np.mean(alignment_hist)):.3f}",
            f"{float(np.mean(nnd_hist)):.3f}",
            f"{float(np.mean(centroid_hist)):.3f}",
            f"{float(np.mean(speed_hist)):.3f}",
            f"{float(np.mean(step_times)):.2f}",
        ])
    
    # 保存LLM决策到JSON文件（用于分析）
    decisions_json_path = export_path.replace('.csv', '_decisions.json')
    with open(decisions_json_path, 'w') as f:
        json.dump(all_llm_decisions, f, indent=2)
    print(f"  📊 LLM决策数据: {decisions_json_path}")
    
    total_time = time.time() - start_time
    
    # 计算关键metrics
    metrics = {
        "alignment": float(np.mean(alignment_hist)),
        "nnd": float(np.mean(nnd_hist)),
        "centroid": float(np.mean(centroid_hist)),
        "speed": float(np.mean(speed_hist)),
        "alignment_std": float(np.std(alignment_hist)),
        "speed_std": float(np.std(speed_hist)),
        "alignment_min": float(np.min(alignment_hist)),
        "alignment_max": float(np.max(alignment_hist)),
        "speed_min": float(np.min(speed_hist)),
        "speed_max": float(np.max(speed_hist)),
        "avg_step_time": float(np.mean(step_times)),
        "total_time": total_time,
        "total_api_calls": engine.llm_performance_stats["requests_sent"],
        "successful_calls": engine.llm_performance_stats["responses_received"],
        "failed_calls": engine.llm_performance_stats["errors_encountered"],
        "export_path": export_path,
    }
    
    return metrics


def extract_key_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从多个运行结果中提取关键metrics
    """
    if not results:
        return {}
    
    key_metrics = {
        "num_runs": len(results),
        "alignment_mean": np.mean([r["alignment"] for r in results]),
        "alignment_std": np.std([r["alignment"] for r in results]),
        "speed_mean": np.mean([r["speed"] for r in results]),
        "speed_std": np.std([r["speed"] for r in results]),
        "nnd_mean": np.mean([r["nnd"] for r in results]),
        "centroid_mean": np.mean([r["centroid"] for r in results]),
        "avg_total_time": np.mean([r["total_time"] for r in results]),
        "avg_step_time": np.mean([r["avg_step_time"] for r in results]),
        "total_api_calls": sum([r["total_api_calls"] for r in results]),
        "total_successful_calls": sum([r["successful_calls"] for r in results]),
        "total_failed_calls": sum([r["failed_calls"] for r in results]),
        "success_rate": sum([r["successful_calls"] for r in results]) / sum([r["total_api_calls"] for r in results]) if sum([r["total_api_calls"] for r in results]) > 0 else 0.0,
    }
    
    return key_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-driven Boids simulation (per-step LLM, wait for all decisions)")
    parser.add_argument("--runs", type=int, default=1, help="Number of random-seed runs")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen (DashScope) backend instead of OpenAI")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek backend")
    parser.add_argument("--seed", type=int, nargs="*", help="Specific seeds to run (overrides --runs)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for concurrent LLM calls (default: 20)")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent workers per batch (default: batch_size)")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps (default: 100)")
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
    
    print("="*80)
    print(f"Per-Step LLM Evaluation (wait for all decisions)")
    print("="*80)
    print(f"Backend: {label}")
    print(f"Runs: {len(seeds)}")
    print(f"Steps per run: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: Pure LLM (no traditional alignment force)")
    print(f"Expected API calls per run: {args.steps * 50:,}")
    print("="*80)
    
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n▶ Run {idx}/{len(seeds)} (seed={seed})")
        res = run_single_per_step(
            seed=seed,
            use_qwen=args.qwen,
            use_deepseek=args.deepseek,
            label=label,
            batch_size=args.batch_size,
            steps=args.steps,
            max_workers_per_batch=args.max_workers,
        )
        results.append(res)
        print(
            f"  ✅ alignment={res['alignment']:.3f}±{res['alignment_std']:.3f}, "
            f"speed={res['speed']:.3f}±{res['speed_std']:.3f}, "
            f"time={res['total_time']/60:.1f}min, "
            f"API calls={res['total_api_calls']}, "
            f"success={res['successful_calls']}/{res['total_api_calls']} "
            f"({res['successful_calls']/res['total_api_calls']*100:.1f}%)"
        )
        print(f"  📄 {res['export_path']}")
    
    if results:
        key_metrics = extract_key_metrics(results)
        print("\n" + "="*80)
        print("KEY METRICS SUMMARY")
        print("="*80)
        print(f"\n📊 Performance Metrics:")
        print(f"  Alignment (polarisation): {key_metrics['alignment_mean']:.3f} ± {key_metrics['alignment_std']:.3f}")
        print(f"  Speed: {key_metrics['speed_mean']:.3f} ± {key_metrics['speed_std']:.3f}")
        print(f"  Nearest neighbor distance: {key_metrics['nnd_mean']:.3f}")
        print(f"  Centroid distance: {key_metrics['centroid_mean']:.3f}")
        
        print(f"\n⏱️  Timing:")
        print(f"  Average total time per run: {key_metrics['avg_total_time']/60:.1f} minutes")
        print(f"  Average time per step: {key_metrics['avg_step_time']:.2f} seconds")
        
        print(f"\n🔌 API Usage:")
        print(f"  Total API calls: {key_metrics['total_api_calls']:,}")
        print(f"  Successful: {key_metrics['total_successful_calls']:,}")
        print(f"  Failed: {key_metrics['total_failed_calls']:,}")
        print(f"  Success rate: {key_metrics['success_rate']*100:.1f}%")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()

