> **Legacy snapshot, not maintained.** This folder keeps the original scripts verbatim for provenance and as the reference behaviour for the refactored task in [`../`](../). It is not imported by the task at runtime and is not part of the test suite. The scripts need their own environment: Mesa 2.x (the `Agent(unique_id, model)` constructor). The LLM variants in `variants/` also import the pre-refactor `socioverse.behavior_engine` kernel, which is not shipped here. They do not run against the current kernel.
>
> Original author: Shijun Lei ([@ShijunLei-cn](https://github.com/ShijunLei-cn)).
>
> `agents.py` and `civil_violence_model.py` are adapted from the Mesa Epstein civil violence example (Apache-2.0, Copyright Core Mesa Team and contributors; see the file headers and `NOTICE`). The original description below is partly in Chinese; the `variants/` scripts are the LLM context/behaviour variants (OCM/LCM x TBF/LBF).

# Civil Violence Model

This implementation demonstrates the Epstein civil violence model using the Mesa agent-based modeling framework.

## 模型概述

Epstein 内战暴力模型展示了公民暴力如何随着合法性下降和不满情绪增加而出现。该模型包含市民和警察代理，探索了政府镇压与公民反抗之间的动态关系。

## 核心特点

- **环境**: 网格空间（托拓扑）
- **主体**: 市民代理和警察代理
- **行为规则**: 基于不满程度、风险规避和被捕概率的激励决策
- **结果**: 展示暴力如何通过社会动态演变

## 文件结构

```
civil_violence/
├── agents.py                    # 定义市民和警察代理
├── civil_violence_model.py      # 定义模型逻辑
├── run_simulation.py            # 运行仿真并可视化结果
├── README.md                    # 本文件
├── simulation_steps/            # 🔄 每一步的过程图像（GIF动画源）
│   ├── step_000.png
│   ├── step_001.png
│   ├── ...
│   └── civil_violence_simulation.gif
└── output_figs/                 # 📊 最终结果图像
    └── civil_violence_results.png
```

## 使用方法

1. 确保已激活 conda 环境：
```bash
conda activate socioverse
```

2. 运行仿真：
```bash
cd raw_simulations/organization_model/civil_violence
python run_simulation.py
```

## 参数说明

- `steps`: 仿真步数（默认200）
- `height/width`: 网格尺寸（默认40x40）
- `citizen_density`: 市民密度（0-1，默认0.2）
- `cop_density`: 警察密度（0-1，默认0.001）
- `legitimacy`: 政府合法性（0-1，默认0.5）
- `citizen_vision`: 市民视野范围（默认1）
- `cop_vision`: 警察视野范围（默认2）
- `threshold`: 激进的阈值（默认0.05）
- `max_jail_time`: 最长监禁时间（默认30步）

## 输出说明

### simulation_steps/ 文件夹
- **用途**: 存储仿真的每一步快照
- **文件**: `step_000.png`, `step_001.png`, ... `step_XXX.png`
- **civil_violence_simulation.gif**: 由所有步骤图像合成的动画

### output_figs/ 文件夹
- **用途**: 存储最终的分析和统计图表
- **civil_violence_results.png**: 包含以下内容的4个子图：
  1. 随时间变化的人口动态（静止/激进/被监禁）
  2. 激进 vs 被监禁市民的对比
  3. 最终代理位置的散点图
  4. 激进市民的百分比随时间的变化

## 主要发现

模型展示了几个关键现象：
1. **引爆点现象**: 市民暴力显示临界点行为
2. **连锁反应**: 当达到临界质量时的级联效应
3. **随机性影响**: 小的随机事件可能引发大规模暴动
4. **政策含义**: 合法性和警察覆盖的效果