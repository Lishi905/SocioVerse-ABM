# -*- coding: utf-8 -*-
# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
NDlib 实现的 SIR 类比谣言传播（Ignorant->Spreader->Stifler）与可视化
- 智能体状态映射：Ignorant=Susceptible, Spreader=Infected, Stifler=Removed
- 近似：用 SIR 的 gamma 近似 MT 模型中的接触驱动“失去兴趣”
"""

import random
import networkx as nx
import matplotlib.pyplot as plt

from ndlib.models.epidemics.SIRModel import SIRModel
from ndlib.models.ModelConfig import Configuration

# ---------- 1) 场景/参数（可按需改动） ----------
N = 500                   # 智能体数量
topology = "WS"             # 选择: "ER"(随机图) | "BA"(无标度) | "WS"(小世界)
seed = 42                   # 随机种子（复现）
steps = 80                  # 仿真步数
initial_spreaders = 1       # 初始 Spreader 数量

# 谣言传播常用设定（来自文献与社区实践）
p = 0.6   # 转发概率：Spreader 接触 Ignorant 时对方变 Spreader 的意愿基准
q = 0.1   # 免疫概率：Ignorant 接触后直接变为 Stifler 的概率（识别为假消息/没兴趣）
# 近似映射到 SIR：
beta_eff = p * (1 - q)  # 有效传播率（Ignorant 不直接变 Stifler 的那部分）
gamma = 0.25            # 失去兴趣/停止传播（可理解为 MT 中接触驱动的等效平均效应）

# ---------- 2) 构图 ----------
random.seed(seed)

if topology == "ER":
    # Erdos-Rényi 随机图：平均度 ~ p_edge*(N-1)
    p_edge = 6 / (N - 1)   # 让平均度大约为 6
    G = nx.erdos_renyi_graph(N, p_edge, seed=seed)
elif topology == "BA":
    # Barabási–Albert 无标度图：平均度 ~ 2*m
    m = 3
    G = nx.barabasi_albert_graph(N, m, seed=seed)
else:
    # Watts–Strogatz 小世界图：k=近邻数，p=重连概率
    k = 6
    rewiring_p = 0.1
    G = nx.watts_strogatz_graph(N, k, rewiring_p, seed=seed)

# 确保连通（若不连通，取最大连通子图）
if not nx.is_connected(G):
    print("[提示] 网络不连通，提取最大连通子图")
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    N = G.number_of_nodes()

# ---------- 3) NDlib: SIR 类比谣言模型 ----------
model = SIRModel(G)

config = Configuration()
config.add_model_parameter('beta', beta_eff)
config.add_model_parameter('gamma', gamma)

# 精确定义初始 Spreader（NDlib: Infected）
seeds = random.sample(list(G.nodes()), k=min(initial_spreaders, N))
config.add_model_initial_configuration("Infected", seeds)

model.set_initial_status(config)

# ---------- 4) 迭代并记录 ----------
iterations = model.iteration_bunch(steps)

# 辅助：获取状态名称映射
status_map = model.get_status_map() 
# print("状态映射:", status_map) {'Susceptible': 0, 'Infected': 1, 'Removed': 2}

S_series, I_series, R_series = [], [], []
for it in iterations:
    # it['status']：每个节点当下的状态（整张图）
    # st = it.get('status', {})
    # 计数
    # cntS = sum(1 for v in st.values() if v == rev_map['Susceptible'])
    # cntI = sum(1 for v in st.values() if v == rev_map['Infected'])
    # cntR = sum(1 for v in st.values() if v == rev_map['Removed'])
    # S_series.append(cntS)
    # I_series.append(cntI)
    # R_series.append(cntR)
    
    counts = it.get('node_count', {})  # ← 全图计数（不是差分）
    cntS = counts.get(status_map['Susceptible'], 0)
    cntI = counts.get(status_map['Infected'], 0)
    cntR = counts.get(status_map['Removed'], 0)
    S_series.append(cntS)
    I_series.append(cntI)
    R_series.append(cntR)

    # print(f"Step {it['iteration']}: Ignorant={cntS}, Spreader={cntI}, Stifier={cntR}")
    # waitkey = input("Press Enter to continue...")

# ---------- 5) 指标：最终总传播规模 ----------
# “听说/转发过”的规模 = 最终的 Spreader+Stifler；此处近似用最终 (N - Ignorant_final)
final_reach = N - I_series[-1]

# ---------- 6) 可视化 ----------
plt.figure(figsize=(7.5, 4.5))
plt.plot(S_series, label='Ignorant (S)', linewidth=2)
plt.plot(I_series, label='Spreader (I)', linewidth=2)
plt.plot(R_series, label='Stifler (R)', linewidth=2)
plt.title(f'Rumor (SIR-like) on {topology} network | N={N}, β_eff={beta_eff:.2f}, γ={gamma:.2f}')
plt.xlabel('Step')
plt.ylabel('Count of agents')
plt.legend()
plt.tight_layout()
# plt.savefig(f'./res_figs_simple/SIR_rumor_diffusion_{topology}_{N}.png', dpi=300)
plt.savefig(f'./res_figs_simple/SIR_rumor_diffusion.png', dpi=300)

print(f"[结果] 最终总传播规模（听说/转发过的人数）≈ {final_reach} / {N}  "
      f"({final_reach/N:.1%})")

# ---------- 7) 终态着色网络图（可选） ----------
try:
    # 终态颜色：Ignorant=lightgray, Spreader=tab:orange, Stifler=tab:blue
    last_status = iterations[-1]['status']
    color_map = []
    for n in G.nodes():
        s = last_status[n]
        if s == status_map['Susceptible']:
            color_map.append('#d3d3d3')
        elif s == status_map['Infected']:
            color_map.append('#ff7f0e')
        else:
            color_map.append('#1f77b4')
    plt.figure(figsize=(6.2, 6.2))
    pos = nx.spring_layout(G, seed=seed)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=color_map, linewidths=0.0)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
    plt.axis('off')
    plt.title('Final states: gray=Ignorant, orange=Spreader, blue=Stifler')
    plt.savefig('./res_figs_simple/SIR_rumor_final_network.png', dpi=300)
except Exception as e:
    print("[可视化提示] 终态网络绘图跳过：", e)

# ---------- 8) （可选）替换为 NDlib 谣言模型：若你的 NDlib 版本包含 MT 模型 ----------
"""
try:
    from ndlib.models.epidemics.MakiThompsonModel import MakiThompsonModel

    model = MakiThompsonModel(G)
    config = Configuration()
    # 常见参数命名可能为（示例，具体以版本文档为准）：
    #   'p'：Spreader 接触 Ignorant 时对方转为 Spreader 的概率（≈ p*(1-q)）
    #   'q'：Ignorant 接触后直接变为 Stifler 的概率
    #   'r_ss'：Spreader 接触 Spreader 时转为 Stifler 的概率
    #   'r_sr'：Spreader 接触 Stifler 时转为 Stifler 的概率
    config.add_model_parameter('p', p*(1-q))
    config.add_model_parameter('q', q)
    config.add_model_parameter('r_ss', 0.30)
    config.add_model_parameter('r_sr', 0.30)
    config.add_model_initial_configuration("Infected", seeds)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(steps)
    # 后续计数与绘图与上面相同
except Exception as _:
    pass
"""
