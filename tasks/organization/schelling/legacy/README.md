> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: Mesa 2.x (`mesa.time`) and the `openai>=1` client (optionally `anthropic` / `aiohttp`). The `extends/` variants also import the pre-refactor `socioverse` kernel and a `simulations.Schelling_model` package, which are not shipped here. They do not run against the current kernel.
>
> Original author: Shijun Lei ([@ShijunLei-cn](https://github.com/ShijunLei-cn)).
>
> `schelling_model.py` is adapted from the Mesa Schelling example (Apache-2.0, Copyright Core Mesa Team and contributors; see the file header and `NOTICE`). Edits for the public release: `base.py` and `venue.py` were restored from the original author's code (they were empty in the snapshot), and the LLM base URL defaults in `config.py` / `llm_integration.py` now point at the official OpenAI endpoint instead of a third-party relay. The `extends/*_ve_variant.py` scripts still import their original package path. The original description below is partly in Chinese.

# Schelling Segregation Model

This implementation demonstrates the Schelling segregation model using the Mesa agent-based modeling framework.

## 模型概述

Schelling 分离模型展示了即使个体只有温和的偏好，也会导致宏观层面的高度分离现象。这个模型说明了"微观动机"与"宏观结果"之间可能存在的巨大差异。

## 核心特点

- **环境**: 网格空间（托拓扑）
- **主体**: 两种类型的代理（红色和蓝色）
- **行为规则**: 每个代理都有一个容忍度阈值，如果邻居中同类比例低于阈值则会移动
- **结果**: 即使温和的偏好也会导致高度分离的社区

## 文件结构

```
schelling_segregation/
├── schelling_agent.py           # 定义 Schelling 代理类
├── schelling_model.py           # 定义 Schelling 模型类
├── run_simulation.py            # 运行仿真并可视化结果
├── README.md                    # 本文件
├── simulation_steps/            # 🔄 每一步的过程图像（GIF动画源）
│   ├── step_000.png
│   ├── step_001.png
│   ├── ...
│   └── schelling_simulation.gif
└── output_figs/                 # 📊 最终结果图像
    └── schelling_results.png
```

## 使用方法

1. 确保已激活 conda 环境：
```bash
conda activate socioverse
```

2. 运行仿真：
```bash
cd raw_simulations/organization_model/schelling_segregation
python run_simulation.py
```

## 参数说明

- `steps`: 仿真步数（默认100）
- `height/width`: 网格尺寸（默认50x50）
- `density`: 网格占用率（0-1之间，默认0.8）
- `minority_pc`: 少数群体比例（默认0.5）
- `homophily`: 同质性阈值（期望的同类邻居数量，默认5）

## 输出说明

### simulation_steps/ 文件夹
- **用途**: 存储仿真的每一步快照
- **文件**: `step_000.png`, `step_001.png`, ... `step_XXX.png`
- **schelling_simulation.gif**: 由所有步骤图像合成的动画，显示分离过程的演变

### output_figs/ 文件夹
- **用途**: 存储最终的分析和统计图表
- **schelling_results.png**: 包含以下内容的两个子图：
  1. 最终代理分布（红色=多数派，蓝色=少数派，白色=空单元）
  2. 随时间变化的幸福代理数量趋势图

## 主要发现

模型展示了一个反直觉的结论：即使个体只有非常温和的偏好（例如，不希望自己成为绝对的少数派），宏观层面也会自发地演化出几乎完全隔离的社区。

这个模型的含义：
- **微观-宏观差异**: 个体的温和偏好不代表宏观结果也温和
- **自我强化机制**: 一旦开始分离，就会形成自我强化的循环
- **社会政策**: 理解集体行为的复杂性对政策制定至关重要