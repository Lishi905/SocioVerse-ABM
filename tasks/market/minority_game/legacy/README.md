> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: Mesa 3 (see `requirements.txt`). `minority_game_sim_consolidated.py` also imports the pre-refactor `socioverse.behavior_engine` kernel, which is not shipped here. They do not run against the current kernel.
>
> Original authors: Chenyu Li ([@if111111111111111111111](https://github.com/if111111111111111111111)); `minority_game_sim_consolidated.py` by Jia Wang ([@JiaWANG-TJ](https://github.com/JiaWANG-TJ)).
>
> The original description below is in Chinese. In short: N players choose side A or B each round, the minority side wins, and each player picks among S strategy tables keyed on the last M outcomes by their virtual scores (built on Mesa).

# Minority Game Simulation

## 项目简介

少数派游戏（Minority Game）是一个经典的多智能体系统模型，用于研究复杂适应系统中的协调和竞争行为。在游戏中：

- **N** 个玩家每轮选择 A（0）或 B（1）两个选项之一
- 选择**少数派**的玩家获胜
- 玩家基于过去 **M** 轮的历史信息做出决策
- 每个玩家拥有 **S** 条策略，通过虚拟积分跟踪策略表现

该实现使用 [Mesa](https://mesa.readthedocs.io/) 框架构建，这是一个 Python 的基于智能体建模（ABM）框架。

## 项目结构

```
miniority_game/
├── README.md                    # 本文档
├── minority_game_agent.py       # Agent 和 Strategy 类定义
├── minority_game_model.py       # Model 类定义（Mesa Model）
├── minority_game.py             # 主程序（命令行接口）
└── run_experiments.py           # 自动运行两个实验的脚本
```

## 安装依赖

```bash
pip install mesa numpy matplotlib
```

或使用项目根目录的 `setup.py`：

```bash
cd SocioVerse-ABM
pip install -e .
```

## 使用方法

### 基本用法

```bash
python minority_game.py --N <agents> --M <memory> --S <strategies> --steps <steps>
```

### 参数说明

**必需参数：**
- `--N`: 智能体数量（建议使用奇数以避免平局）
- `--S`: 每个智能体的策略数量
- `--steps`: 仿真运行的步数

**可选参数：**
- `--M`: 记忆长度（默认 6）
- `--seed`: 随机种子，用于结果可重复性
- `--M-mix`: 混合记忆长度配置，格式如 `"1:100,2:100,3:100"`（M值:智能体数量）
- `--burn-in`: 预热期步数，统计时会忽略这些步骤（默认 0）
- `--plot`: 生成时间序列图
- `--dump-agent-stats`: 导出智能体统计数据到 CSV 文件

### 示例

**简单运行：**
```bash
python minority_game.py --N 1001 --M 6 --S 5 --steps 10000 --seed 42
```

**生成可视化图表：**
```bash
python minority_game.py --N 1001 --M 6 --S 5 --steps 20000 --burn-in 2000 --plot --seed 42
```

**混合人群实验：**
```bash
python minority_game.py --N 1001 --S 5 --steps 20000 --burn-in 2000 --seed 123 \
  --M-mix "1:100,2:100,3:100,4:100,5:100,6:100,7:100,8:100,9:100,10:101" \
  --dump-agent-stats
```

## 实验

本项目包含两个预设实验，用于验证少数派游戏的关键性质。

### 运行所有实验

最简单的方式是使用自动化脚本：

```bash
python run_experiments.py
```

该脚本会依次运行实验一和实验二，并生成所有需要的图表和数据文件。

### 实验一：不同记忆长度 M 的时间序列

**目的：** 展示记忆长度 M 越大，选择 A 的人数波动越小（方差越小）

**参数：**
- N = 1001（智能体数量）
- S = 5（每个智能体的策略数）
- steps = 20000
- burn-in = 2000
- seed = 42
- M ∈ {6, 8, 10}

**运行命令：**

```bash
# M = 6
python minority_game.py --N 1001 --M 6 --S 5 --steps 20000 --burn-in 2000 --plot --seed 42

# M = 8
python minority_game.py --N 1001 --M 8 --S 5 --steps 20000 --burn-in 2000 --plot --seed 42

# M = 10
python minority_game.py --N 1001 --M 10 --S 5 --steps 20000 --burn-in 2000 --plot --seed 42
```

**预期结果：**
- 所有曲线围绕 N/2 ≈ 500 波动
- 方差满足：Var(M=6) > Var(M=8) > Var(M=10)
- 生成三张时间序列图（PNG 文件）

**生成的文件：**
- `minority_game_M6_N1001_S5_steps20000_seed42.png`
- `minority_game_M8_N1001_S5_steps20000_seed42.png`
- `minority_game_M10_N1001_S5_steps20000_seed42.png`

---

### 实验二：混合人群的胜率 vs 记忆 M

**目的：** 在同场混合 M=1 到 M=10 的智能体，展示按 M 分组的平均胜率随 M 增大而上升，在 M≈6 后趋于平台

**参数：**
- N = 1001（总智能体数）
- S = 5
- steps = 20000
- burn-in = 2000
- seed = 123
- 人群构成：M=1 到 M=9 各 100 个智能体，M=10 有 101 个智能体

**运行命令：**

```bash
python minority_game.py --N 1001 --S 5 --steps 20000 --burn-in 2000 --seed 123 \
  --M-mix "1:100,2:100,3:100,4:100,5:100,6:100,7:100,8:100,9:100,10:101" \
  --dump-agent-stats
```

**预期结果：**
- 胜率随 M 整体上升
- M ≈ 6 之后增幅显著变小（进入平台期）
- 输出 M 分组的平均胜率统计

**生成的文件：**
- `agent_stats.csv`：包含每个智能体的详细统计（ID、M、胜场、胜率、切换次数等）
- `win_rate_by_M.png`：胜率 vs M 的折线图（带误差条）
- `minority_game_Mmixed_N1001_S5_steps20000_seed123.png`：时间序列图

## 输出说明

### 终端输出

程序会输出：
- 选择 A 的人数统计（均值、标准差、方差）
- 如果使用 `--M-mix`，会按 M 值分组显示胜率统计

### CSV 文件（`agent_stats.csv`）

当使用 `--dump-agent-stats` 时生成，包含以下列：

- `AgentID`：智能体唯一标识
- `M`：该智能体的记忆长度
- `wins`：总胜场数
- `moves`：总步数
- `win_rate`：胜率（wins / moves）
- `switches`：策略切换总次数
- `switch_freq`：切换频率（switches / moves）

### 图表文件

1. **时间序列图**：显示每一步选择 A 的人数
   - 蓝色曲线：A_count 随时间变化
   - 红色虚线：N/2 基准线
   - 文本框：均值、标准差、方差统计

2. **胜率图**（实验二）：显示不同 M 值的平均胜率
   - 绿色曲线：平均胜率随 M 变化
   - 误差条：标准差
   - 横轴：记忆长度 M
   - 纵轴：胜率

## 核心机制

### 策略（Strategy）

- 每条策略是一个长度为 2^M 的查找表
- 将历史模式（M 位二进制）映射到行动（0 或 1）
- 维护虚拟积分以跟踪表现

### 智能体（Agent）

- 拥有 S 条策略
- 每轮选择虚拟积分最高的策略
- 使用反事实推理更新所有策略的积分
- 当存在更优策略（积分 ≥ 当前策略 + 1）时切换

### 模型（Model）

每轮执行：
1. 所有智能体基于当前历史选择行动
2. 统计选择 A 和 B 的人数
3. 少数派获胜
4. 更新所有智能体的策略积分和真实得分
5. 更新历史记录

## 理论背景

少数派游戏最早由 Challet 和 Zhang (1997) 提出，用于研究：

- **自组织临界性**：系统自发达到临界状态
- **复杂性涌现**：简单规则产生复杂行为
- **适应性策略**：智能体如何学习和适应
- **记忆与性能**：更长的记忆如何影响决策质量

关键发现：
- M 越大，系统波动越小（更有序）
- 存在最优记忆长度，过短和过长都不利
- 混合人群中，适度记忆长度的智能体表现最好

## 技术实现

### Mesa 框架集成

本实现使用 Mesa 框架的以下组件：

- `mesa.Model`：模型基类
- `mesa.Agent`：智能体基类
- `mesa.time.SimultaneousActivation`：同步激活调度器
- `mesa.datacollection.DataCollector`：数据收集器

### 关键设计决策

1. **无空间结构**：少数派游戏是全局交互，不需要网格
2. **同步更新**：所有智能体同时行动，然后统一更新
3. **反事实学习**：每个策略都评估"如果使用该策略会怎样"
4. **随机种子控制**：使用 Mesa 的随机数生成器确保可重复性

## 扩展与修改

### 修改参数

编辑 `minority_game.py` 中的默认值或使用命令行参数。

### 添加新实验

在 `run_experiments.py` 中添加新的实验函数。

### 自定义策略

修改 `minority_game_agent.py` 中的 `Strategy` 类以实现不同的策略生成或选择机制。

### 可视化

可以使用 Mesa 的可视化模块创建交互式 Web 界面（类似 `civil_violence` 示例）。

## 常见问题

**Q: 为什么 N 应该是奇数？**  
A: 奇数可以避免平局。如果出现平局，程序会随机选择获胜方。

**Q: burn-in 期有什么用？**  
A: 系统需要一段时间才能达到稳定状态。burn-in 期的数据会从统计中排除。

**Q: 如何选择合适的步数？**  
A: 通常 10000-20000 步足以观察到稳定的统计特性。更长的运行可以获得更准确的结果。

**Q: 为什么实验二要混合不同的 M？**  
A: 这样可以在同一环境下公平比较不同记忆长度的表现。

## 参考文献

1. Challet, D., & Zhang, Y. C. (1997). Emergence of cooperation and organization in an evolutionary game. *Physica A*, 246(3-4), 407-418.

2. Arthur, W. B. (1994). Inductive reasoning and bounded rationality. *The American Economic Review*, 84(2), 406-411.

3. Johnson, N. F., Jefferies, P., & Hui, P. M. (2003). *Financial market complexity*. Oxford University Press.

## 许可证

本项目遵循与 SocioVerse-ABM 主项目相同的许可证。

## 作者

SocioVerse-ABM 项目组

---

**最后更新：** 2025-10-15

