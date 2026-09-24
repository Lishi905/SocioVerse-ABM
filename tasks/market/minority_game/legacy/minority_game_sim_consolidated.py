# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Minority Game 模拟实现 (ABM & LLM)

少数派游戏：N 个玩家每轮在 A(0) 和 B(1) 之间选择，少数派获胜。
每个玩家持有 S 个策略，基于最近 M 轮的获胜历史做决策。

用法:
    python minority_game_sim.py --N 101 --M 6 --S 5 --steps 100 --mode abm
    python minority_game_sim.py --N 1001 --M 6 --S 5 --steps 1000 --mode llm 
"""

import os
import json
import re
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免 Tkinter 多线程问题
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from socioverse.behavior_engine.ABM_agent import ABM_Agent
from socioverse.behavior_engine.LLM_agent import LLM_Agent
from socioverse.behavior_engine.LLM_based import parse_attributes, generate

# 并发请求的最大并发数
MAX_CONCURRENT_REQUESTS = 100


def extract_action_from_llm_response(content: str, fallback_action: int = 0) -> int:
    """
    从LLM响应中提取action值（0或1）。
    
    支持多种输出格式：
    1. 纯JSON: {"action": 0}
    2. Markdown代码块: ```json\n{"action": 0}\n```
    3. 混合文字和JSON: "Based on analysis...\n{"action": 0}"
    4. 长篇推理后的JSON
    5. 带有单引号的JSON: {'action': 0}
    
    Args:
        content: LLM原始响应字符串
        fallback_action: 解析失败时的默认动作
        
    Returns:
        提取的action值（0或1）
    """
    if not content or not isinstance(content, str):
        return fallback_action
    
    content = content.strip()
    
    # 方法1：处理 ```json ... ``` 格式
    json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    json_blocks = re.findall(json_block_pattern, content, re.DOTALL | re.IGNORECASE)
    for block in json_blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict) and "action" in data:
                action = data["action"]
                if action in (0, 1):
                    return action
        except (json.JSONDecodeError, TypeError):
            continue
    
    # 方法2：从文本中查找所有可能的JSON对象 {...}
    json_obj_pattern = r"\{[^{}]*\}"
    json_objs = re.findall(json_obj_pattern, content)
    for obj_str in json_objs:
        try:
            data = json.loads(obj_str)
            if isinstance(data, dict) and "action" in data:
                action = data["action"]
                if action in (0, 1):
                    return action
        except (json.JSONDecodeError, TypeError):
            # 尝试将单引号替换为双引号后再解析
            try:
                fixed_str = obj_str.replace("'", '"')
                data = json.loads(fixed_str)
                if isinstance(data, dict) and "action" in data:
                    action = data["action"]
                    if action in (0, 1):
                        return action
            except (json.JSONDecodeError, TypeError):
                continue
    
    # 方法3：直接正则匹配 "action": 0 或 "action": 1 模式
    action_pattern = r'["\']?action["\']?\s*[:=]\s*(\d+)'
    match = re.search(action_pattern, content, re.IGNORECASE)
    if match:
        try:
            action = int(match.group(1))
            if action in (0, 1):
                return action
        except ValueError:
            pass
    
    # 方法4：查找独立的0或1（在action关键字附近）
    # 优先匹配最后出现的action相关值
    action_context_pattern = r'action["\']?\s*[:=]?\s*["\']?\s*(0|1)'
    matches = re.findall(action_context_pattern, content, re.IGNORECASE)
    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass
    
    # 所有方法都失败，返回fallback
    return fallback_action


class MinorityGame_Agent:
    """
    Minority Game 智能体类
    
    支持 ABM 和 LLM 两种模式:
    - ABM: 使用传统策略表和虚拟分数进行决策
    - LLM: 调用大语言模型进行决策
    """
    
    def __init__(self, mode, N, M, S, steps, seed, output_pdf, llm_model="gpt-4o-mini", verbose=False):
        """
        初始化 Minority Game 模拟
        
        Args:
            mode: 模拟模式，"abm" 或 "llm"
            N: 玩家数量（建议使用奇数避免平局）
            M: 记忆长度（策略基于最近 M 轮的获胜历史）
            S: 每个玩家的策略数量
            steps: 模拟步数
            seed: 随机种子
            output_pdf: 结果输出路径
            llm_model: LLM 模型名称（仅在 llm 模式下使用）
            verbose: 是否在终端打印完整的LLM提示词和响应（仅在 llm 模式下使用）
        """
        self.N = N
        self.M = M
        self.S = S
        self.steps = steps
        self.seed = seed
        self.output_pdf = output_pdf
        self.llm_model = llm_model
        self.verbose = verbose
        
        self.mode = mode
        self.llm_agent = None
        self.abm_agent = None
    
    def init_strategies(self):
        """
        初始化策略表和虚拟分数（仅 LLM 模式使用）
        
        每个玩家有 S 个策略，每个策略是 2^M 大小的查找表
        注意：历史记录在策略表之前初始化，以避免在特定seed下产生极端初始历史
        """
        random.seed(self.seed)
        
        # 首先初始化历史记录（在策略表之前，确保初始历史更加多样化）
        # 这样避免了在特定seed下由于策略表生成消耗大量RNG状态后
        # 产生极端的初始历史（如全0或全1）
        self.history = [random.randint(0, 1) for _ in range(self.M)]
        
        table_size = 2 ** self.M
        
        self.strategies = []      # strategies[agent_id][strategy_id] = list of 0/1
        self.virtual_scores = []  # virtual_scores[agent_id][strategy_id] = float
        
        for _ in range(self.N):
            agent_strategies = []
            agent_scores = []
            for _ in range(self.S):
                table = [random.randint(0, 1) for _ in range(table_size)]
                agent_strategies.append(table)
                agent_scores.append(0.0)
            self.strategies.append(agent_strategies)
            self.virtual_scores.append(agent_scores)
    
    def history_to_index(self, history):
        """
        将历史记录转换为策略表索引
        
        Args:
            history: 最近 M 轮的获胜历史列表
            
        Returns:
            整数索引 (0 到 2^M - 1)
        """
        index = 0
        for i, bit in enumerate(history[-self.M:]):
            index += int(bit) * (2 ** (self.M - 1 - i))
        return index
    
    def select_best_strategy(self, agent_id):
        """
        为指定玩家选择虚拟分数最高的策略
        
        Args:
            agent_id: 玩家 ID
            
        Returns:
            (最佳策略索引, 推荐动作)
        """
        max_score = max(self.virtual_scores[agent_id])
        best_indices = [i for i, s in enumerate(self.virtual_scores[agent_id]) if s == max_score]
        best_idx = random.choice(best_indices)
        
        history_index = self.history_to_index(self.history)
        recommended_action = self.strategies[agent_id][best_idx][history_index]
        
        return best_idx, recommended_action
    
    def interation(self):
        """
        执行模拟迭代
        
        Returns:
            ABM 模式: iterations 列表，包含每步的统计数据
            LLM 模式: iterations 列表，包含每步的统计数据
        """
        if self.mode == "abm":
            # ABM 模式：通过 ABM_Agent 调用
            self.abm_agent = ABM_Agent(model="minority_game", scenario="minority_game")
            attributes = {
                "N": self.N,
                "M": self.M,
                "S": self.S,
                "steps": self.steps,
                "seed": self.seed
            }
            self.abm_agent.update_attributes(attributes)
            iterations = self.abm_agent.take_actions()
            return iterations
        
        elif self.mode == "llm":
            # LLM 模式：逐步调用 LLM 进行决策（同一步内并发执行）
            self.init_strategies()
            self.llm_agent = LLM_Agent(model=self.llm_model, scenario="minority_game")
            
            iterations = []
            self.conversation_history = []  # 存储完整的对话上下文历史
            
            for step_idx in tqdm(range(self.steps)):
                history_index = self.history_to_index(self.history)
                recent_winners = self.history[-self.M:]
                
                # 定义单个 agent 的 LLM 请求处理函数（用于并发执行）
                def process_agent_request(agent_id):
                    # 选择最佳策略并获取推荐动作
                    _, recommended_action = self.select_best_strategy(agent_id)
                    
                    # 构建 LLM 输入属性
                    attributes = {
                        "agent_id": agent_id,
                        "M": self.M,
                        "S": self.S,
                        "round_num": step_idx,
                        "recent_winners": recent_winners,
                        "recommended_action": recommended_action
                    }
                    self.llm_agent.update_attributes(attributes)
                    
                    # 调用 LLM 获取动作（最多重试10次）
                    prompt = parse_attributes("minority_game", attributes)
                    max_retries = 10
                    raw_response = None
                    last_error = None
                    
                    # 如果启用verbose模式，打印提示词
                    if self.verbose:
                        print(f"\n{'='*80}")
                        print(f"[Step {step_idx}] [Agent {agent_id}] LLM 提示词:")
                        print(f"{'-'*80}")
                        print(prompt)
                        print(f"{'='*80}")
                    
                    for retry in range(max_retries):
                        try:
                            raw_response = generate(self.llm_model, prompt)
                            
                            # 检查API是否返回了空响应（None或空字符串）
                            if raw_response is None or (isinstance(raw_response, str) and raw_response.strip() == ""):
                                raise ValueError("API返回空响应")
                            
                            # 如果启用verbose模式，打印响应
                            if self.verbose:
                                print(f"\n{'='*80}")
                                print(f"[Step {step_idx}] [Agent {agent_id}] LLM 响应:")
                                print(f"{'-'*80}")
                                print(raw_response)
                                print(f"{'='*80}\n")
                            
                            # 使用健壮的action提取函数解析响应
                            action = extract_action_from_llm_response(raw_response, recommended_action)
                            break  # 成功则跳出重试循环
                        except Exception as e:
                            last_error = e
                            if retry < max_retries - 1:
                                print(f"[Agent {agent_id}] LLM 调用错误 (重试 {retry + 1}/{max_retries}): {e}")
                            continue
                    else:
                        # 所有重试都失败
                        print(f"[Agent {agent_id}] LLM 调用错误（已重试{max_retries}次）: {last_error}，使用推荐动作")
                        action = recommended_action
                    
                    return {
                        "agent_id": agent_id,
                        "action": action,
                        "prompt": prompt,
                        "response": raw_response if raw_response else f"ERROR: {str(last_error)}"
                    }
                
                # 使用 ThreadPoolExecutor 并发处理所有玩家的 LLM 请求
                # 限制并发数为 MAX_CONCURRENT_REQUESTS (50)
                results = []
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                    # 提交所有 agent 的任务
                    futures = {
                        executor.submit(process_agent_request, agent_id): agent_id
                        for agent_id in range(self.N)
                    }
                    
                    # 收集所有结果（等待所有任务完成）
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            agent_id = futures[future]
                            print(f"[Agent {agent_id}] 并发执行异常: {e}")
                            # 使用推荐动作作为 fallback
                            _, recommended_action = self.select_best_strategy(agent_id)
                            results.append({
                                "agent_id": agent_id,
                                "action": recommended_action,
                                "prompt": "",
                                "response": f"ERROR: {str(e)}"
                            })
                
                # 按 agent_id 排序结果，确保顺序一致
                results.sort(key=lambda x: x["agent_id"])
                
                # 提取动作列表
                choices = [r["action"] for r in results]
                
                # 记录对话上下文
                for r in results:
                    self.conversation_history.append({
                        "step": step_idx,
                        "agent_id": r["agent_id"],
                        "prompt": r["prompt"],
                        "response": r["response"]
                    })
                
                # 统计 A(0) 和 B(1) 的数量
                count_A = sum(1 for c in choices if c == 0)
                count_B = self.N - count_A
                
                # 确定获胜方（少数派）
                if count_A < count_B:
                    winning_side = 0
                elif count_B < count_A:
                    winning_side = 1
                else:
                    winning_side = random.randint(0, 1)
                
                # 更新所有策略的虚拟分数（反事实推理）
                for agent_id in range(self.N):
                    for s_idx in range(self.S):
                        if self.strategies[agent_id][s_idx][history_index] == winning_side:
                            self.virtual_scores[agent_id][s_idx] += 1.0
                
                # 更新历史
                self.history.append(winning_side)
                
                # 记录数据
                iterations.append({
                    "step": step_idx,
                    "A_count": count_A,
                    "B_count": count_B,
                    "winning_side": winning_side
                })
                
                # 打印进度信息
                if step_idx < 3:
                    print(f"  Step {step_idx}: A={count_A}, B={count_B}, Winner={winning_side}")
            
            return iterations
    
    def visualization(self):
        """
        执行模拟并可视化结果，输出完整数据到Excel
        """
        iterations = self.interation()
        
        # 获取脚本所在目录作为基准路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 根据模式创建子文件夹，包含agent数量和step信息
        # ABM 模式：abm_N1001_steps1000
        # LLM 模式：llm_Qwen2.5-72B-Instruct_N101_steps100（提取模型名称最后部分）
        if self.mode == "llm":
            # 提取模型名称的最后一部分（去掉如 Qwen/ 的前缀）
            model_name_short = self.llm_model.split('/')[-1]
            output_dir = os.path.join(script_dir, f"llm_{model_name_short}_N{self.N}_steps{self.steps}")
        else:
            output_dir = os.path.join(script_dir, f"{self.mode}_N{self.N}_steps{self.steps}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        steps_idx = [it["step"] for it in iterations]
        A_counts = [it["A_count"] for it in iterations]
        B_counts = [it["B_count"] for it in iterations]
        winning_sides = [it["winning_side"] for it in iterations]
        
        # 计算统计量
        A_counts_arr = np.array(A_counts)
        mean_A = np.mean(A_counts_arr)
        std_A = np.std(A_counts_arr)
        var_A = np.var(A_counts_arr)
        
        # 生成时间戳用于文件命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文件名包含step和seed信息
        base_filename = f"minority_game_{self.mode}_M{self.M}_N{self.N}_S{self.S}_steps{self.steps}_seed{self.seed}_{timestamp}"
        
        # ========== 保存完整数据到Excel ==========
        # 1. 模拟参数信息
        params_data = {
            "参数名": ["mode", "N", "M", "S", "steps", "seed", "timestamp"],
            "参数值": [self.mode, self.N, self.M, self.S, self.steps, self.seed, timestamp]
        }
        params_df = pd.DataFrame(params_data)
        
        # 2. 每步迭代数据
        iterations_df = pd.DataFrame(iterations)
        
        # 3. 统计汇总数据
        stats_data = {
            "统计指标": ["mean_A_count", "std_A_count", "variance_A_count", "min_A_count", "max_A_count", 
                       "total_steps", "A_wins", "B_wins"],
            "统计值": [mean_A, std_A, var_A, np.min(A_counts_arr), np.max(A_counts_arr),
                     len(iterations), sum(1 for w in winning_sides if w == 0), sum(1 for w in winning_sides if w == 1)]
        }
        stats_df = pd.DataFrame(stats_data)
        
        # 保存到Excel（多个sheet）
        excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            params_df.to_excel(writer, sheet_name='参数配置', index=False)
            iterations_df.to_excel(writer, sheet_name='迭代数据', index=False)
            stats_df.to_excel(writer, sheet_name='统计汇总', index=False)
        
        print(f"\n[数据] Excel已保存至: {excel_path}")
        
        # ========== 绘制时间序列图 ==========
        plt.figure(figsize=(12, 6))
        plt.plot(steps_idx, A_counts, alpha=0.7, linewidth=0.8, color='steelblue', label='A count')
        plt.axhline(y=self.N/2, color='red', linestyle='--', alpha=0.5, label=f'N/2 = {self.N/2}')
        
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Number choosing A', fontsize=12)
        plt.title(f'Minority Game: A Count vs Step (Mode={self.mode}, M={self.M}, N={self.N}, S={self.S}, Steps={self.steps})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息文本框
        stats_text = f'Mean: {mean_A:.2f}\nStd: {std_A:.2f}\nVar: {var_A:.2f}'
        ax = plt.gca()
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        # 图片文件名包含step信息
        output_file = os.path.join(output_dir, f"{base_filename}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[图表] 已保存至: {output_file}")
        print(f"[统计] Mean A-count: {mean_A:.2f}, Std: {std_A:.2f}, Variance: {var_A:.2f}")
        
        # ========== LLM 模式：保存对话上下文到 JSON ==========
        if self.mode == "llm" and hasattr(self, 'conversation_history'):
            json_path = os.path.join(output_dir, f"{base_filename}_conversations.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"[对话] 对话上下文已保存至: {json_path}")
        
        return iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minority Game 模拟 (ABM & LLM)")
    parser.add_argument("--N", type=int, default=101, help="玩家数量，建议使用奇数 (default: 101)")
    parser.add_argument("--M", type=int, default=6, help="记忆长度 (default: 6)")
    parser.add_argument("--S", type=int, default=5, help="每个玩家的策略数量 (default: 5)")
    parser.add_argument("--steps", type=int, default=100, help="模拟步数 (default: 100)")
    parser.add_argument("--seed", type=str, default="42", help="随机种子，支持多个值用逗号分隔，如 1,2,3,4,5 (default: 42)")
    parser.add_argument("--out_pdf", type=str, default="./", help="结果输出路径")
    parser.add_argument("--mode", type=str, default="abm", choices=["llm", "abm"], help="模拟模式 (default: abm)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM 模型名称，仅在 llm 模式下使用 (default: gpt-4o-mini)")
    parser.add_argument("--verbose", action="store_true", help="在终端打印完整的LLM提示词和响应（仅在 llm 模式下有效）")
    
    args = parser.parse_args()
    
    # 解析多个seed值
    seed_list = [int(s.strip()) for s in args.seed.split(",")]
    
    print(f"[配置] 共有 {len(seed_list)} 个随机种子需要运行: {seed_list}")
    
    for idx, seed_val in enumerate(seed_list):
        print(f"\n{'='*60}")
        print(f"[进度] 运行第 {idx+1}/{len(seed_list)} 个实验，seed={seed_val}")
        print(f"{'='*60}")
        
        random.seed(seed_val)
        
        mg_agent = MinorityGame_Agent(
            mode=args.mode,
            N=args.N,
            M=args.M,
            S=args.S,
            steps=args.steps,
            seed=seed_val,
            output_pdf=args.out_pdf,
            llm_model=args.model,  # ← 传入模型名称
            verbose=args.verbose   # ← 传入verbose参数
        )
        
        # 执行模拟并保存结果（始终保存Excel和图表）
        mg_agent.visualization()
        print(f"[完成] 模拟结束，共 {mg_agent.steps} 步")
    
    print(f"\n{'='*60}")
    print(f"[完成] 所有 {len(seed_list)} 个实验已全部运行完毕！")
    print(f"{'='*60}")
