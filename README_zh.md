<!-- ───────────────────────── Challenge banner ─────────────────────────
     Files: assets/challenge-banner.svg (dark card) + assets/challenge-banner-light.svg.
     Both are self-contained (text outlined to paths, no fonts/scripts/external refs),
     so GitHub renders them through <img>. <picture> follows the viewer's GitHub theme.
     Remove this block (and the News line) after the awards on 2026-11-08.            -->
<p align="center">
  <a href="https://socioverse.fudan-disc.com/challenge/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/challenge-banner.svg">
      <source media="(prefers-color-scheme: light)" srcset="assets/challenge-banner-light.svg">
      <img alt="SocioVerse Challenge 2026: AI4SS Challenge for Human-AI Collaboration and Social Governance. Three research tracks, $21,050 in prizes and API support, final submission October 31, 2026." src="assets/challenge-banner.svg" width="100%">
    </picture>
  </a>
</p>

<h1 align="center">SocioVerse-ABM</h1>

<p align="center">
  <b>LLM 智能体能否替代经典 ABM 中手写的规则？</b><br>
  把经典基于主体的模型（ABM）里的规则智能体换成 LLM 智能体，
  检验 LLM 驱动的智能体能否复现规则产生的群体行为。
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.24911"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.24911-b31b1b.svg"></a>
  <a href="https://github.com/sii-research/SocioVerse2"><img alt="SocioVerse2" src="https://img.shields.io/badge/SocioVerse2-runtime-6f42c1.svg"></a>
  <a href="https://socioverse.fudan-disc.com/"><img alt="Homepage" src="https://img.shields.io/badge/Homepage-socioverse.fudan--disc.com-e67e22.svg"></a>
  <a href="https://socioverse.fudan-disc.com/docs/"><img alt="Docs" src="https://img.shields.io/badge/Docs-user%20manual-2563eb.svg"></a>
  <a href="https://socioverse.fudan-disc.com/challenge/"><img alt="Challenge 2026" src="https://img.shields.io/badge/Challenge%202026-open-f97316.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white">
</p>

<p align="center">
  <a href="https://github.com/sii-research/SocioVerse2">SocioVerse2</a> ·
  <a href="https://socioverse.fudan-disc.com/">主页</a> ·
  <a href="https://socioverse.fudan-disc.com/docs/">使用手册</a> ·
  <a href="https://arxiv.org/abs/2609.24911">技术报告</a> ·
  <a href="https://socioverse.fudan-disc.com/challenge/">Challenge 2026</a> ·
  <a href="README.md">English</a>
</p>

---

## 动态

- **2026-09** SocioVerse-ABM v0.2.0 开源，作为 [SocioVerse2](https://github.com/sii-research/SocioVerse2) 配套的 ABM 基准。
- **2026-09-21** SocioVerse2 技术报告发布于 arXiv：[SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm](https://arxiv.org/abs/2609.24911)。本仓库支撑其中的 Case Study 1（*Reproducing Canonical ABMs with LLM Agents*）和 Case Study 3（*Chicago Segregation with Real Census Data*）。
- **2026-09-15** [SocioVerse Challenge 2026](https://socioverse.fudan-disc.com/challenge/)（AI4SS Challenge for Human–AI Collaboration and Social Governance）开放报名：三个赛道，奖金与 API 支持共 $21,050，提案 2026-10-09 截止，最终提交 2026-10-31 截止。

## 概览

这里的每个模型都写成 Lewin 的 **B = f(P, E)**：行为 `B` 是人群 `P` 与环境 `E` 的函数 `f`，
与 SocioVerse2 运行时使用同一锚点。每个任务都在**相同的观测与类型化动作**上把 `f` 实现两次：

- **规则 `f`**：源模型的经典手写规则（参照基准）；
- **LLM `f`**：LLM 接收同样的观测，并必须返回同一类动作（移动或停留、合作或背叛、传播或忽略、朝向、价格立场）。

两种行为函数在同一 `(P, E)` 上、同一动作空间内运行，因此可以逐步比较，并以规则模型为参照打分
（`socioverse_abm.eval` 提供一致性评分，各任务的 `evaluate.py` 提供结果指标）。
`hybrid` 模式按路由策略把每一步的智能体分给规则或 LLM，路由策略就是
[`socioverse_abm/behavior_engine/hybrid.py`](socioverse_abm/behavior_engine/hybrid.py)
中 `hybrid_decide` 的 `route_to_llm` 参数。它的默认策略是 `always_rule`，随仓库提供的任务都沿用这一默认值，
所以 `sv-abm run <task> --mode hybrid` 会让所有智能体走规则，不调用 LLM，不需要 API key，
结果与 `--mode rule` 相同。若要混合规则智能体与 LLM 智能体，请在任务 `model.py`（`_behavior_fn`）
调用 `hybrid_decide` 的地方传入策略：`always_llm`，或任意接收 `Observation`、对需要询问 LLM 的智能体
返回 `True` 的函数（例如按 `agent_id` 划分）。这样的运行需要与 `--mode llm` 相同的 API key。

## 任务

四个家族共 12 个任务。其中 10 个构成 SocioVerse2 技术报告的基准集（Case Study 1）；
`lux_marchesi` 是基准集之外的附加任务；`chicago_segregation` 是真实地理案例（Case Study 3），
在 SocioVerse2 中对应 `chicago_schelling` 研究。

| 家族 | 任务（`sv-abm` 名称） | 源模型 | SocioVerse2 研究 | 报告中 |
|---|---|---|---|---|
| flow | [`nasch`](tasks/flow/nasch/) | Nagel & Schreckenberg (1992) 交通元胞自动机 | `abm_nasch` | Case Study 1 |
| flow | [`boids`](tasks/flow/boids/) | Reynolds (1987) 群集 | `abm_boids` | Case Study 1 |
| flow | [`social_force`](tasks/flow/social_force/) | Helbing 社会力模型（行人疏散） | `abm_social_force` | Case Study 1 |
| market | [`sugarscape`](tasks/market/sugarscape/) | Epstein & Axtell (1996) Sugarscape | `abm_sugarscape` | Case Study 1 |
| market | [`minority_game`](tasks/market/minority_game/) | Challet & Zhang (1997) 少数者博弈 | `abm_minority_game` | Case Study 1 |
| market | [`axelrod`](tasks/market/axelrod/) | Axelrod (1984) 重复囚徒困境锦标赛 | `abm_axelrod` | Case Study 1 |
| market | [`lux_marchesi`](tasks/market/lux_marchesi/) | Lux & Marchesi (1999) 交互主体金融市场 | `abm_lux_marchesi` | 附加任务，不在基准集内 |
| organization | [`schelling`](tasks/organization/schelling/) | Schelling (1971) 居住隔离 | `abm_schelling` | Case Study 1 |
| organization | [`civil_violence`](tasks/organization/civil_violence/) | Epstein (2002) 公民暴力 | `abm_civil_violence` | Case Study 1 |
| organization | [`chicago_segregation`](tasks/organization/chicago_segregation/) | 2010 年芝加哥普查区上的 Schelling 模型（vendored Chicago segregation model @ a1964b8） | `chicago_schelling` | Case Study 3 |
| diffusion | [`sir`](tasks/diffusion/sir/) | 以谣言传播解读的 SIR 过程 | `abm_sir` | Case Study 1 |
| diffusion | [`hegselmann_krause`](tasks/diffusion/hegselmann_krause/) | Hegselmann & Krause (2002) 有界信任 | `abm_hegselmann_krause` | Case Study 1 |

> [!NOTE]
> 各任务默认的 `config.yaml` 是小规模、快速的**演示配置**，不是技术报告表格中的设定
> （例如 NaSch 默认 50 辆车而非 200 辆，少数者博弈默认 101 个智能体、记忆长度 5，而非 301 个、记忆长度 3，
> Axelrod 默认 20 名选手而非 64 名）。复现报告设定时，请修改任务的 `config.yaml`，或用 `--config` 传入自己的配置文件。
> 报告调用 GPT-4o、DeepSeek-V3 和 Qwen3-235B，温度 0.7，最多 256 个输出 token；每个任务规则对照组跑 10 次，每个 LLM 跑 3 次。

## 快速开始

```bash
git clone https://github.com/Lishi905/SocioVerse-ABM.git
cd SocioVerse-ABM

conda create -n sv-abm python=3.11 -y && conda activate sv-abm   # 或：python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

sv-abm list                        # 12 个任务及其对应的 SocioVerse2 研究 id
sv-abm run sir --mode rule         # 规则模式运行，无需 API key
pytest tasks/ socioverse_abm/ -q   # 未安装 [chicago] extra 时，chicago 测试会自动跳过
```

SocioVerse-ABM 的使用方式是克隆仓库后按上面的方式做可编辑安装，本仓库不发布到 PyPI。
普通安装（`pip install .` 或 `pip install git+https://...`）可以运行 11 个经典任务，但不包含芝加哥数据文件；
此时 `chicago_segregation` 需要把 `SV_CHICAGO_LEGACY` 指向某个仓库克隆中的 `tasks/organization/chicago_segregation/legacy/` 目录。

所有任务都可以离线以规则模式运行：`sv-abm run <task> --mode rule`。芝加哥任务需要地理计算依赖：

```bash
pip install -e ".[chicago]"        # geopandas、libpysal、mesa 3、pandas
sv-abm run chicago_segregation --mode rule
```

### LLM 模式

经典任务通过 `socioverse_abm.behavior_engine.llm_f` 调用 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY=...                       # 必填
export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选；任意 OpenAI 兼容接口
sv-abm run schelling --mode llm
```

所用模型由任务 `config.yaml` 中的 `behavior.llm_model` 决定（随仓库提供的文件中为 `gpt-4o`）。
要换用其他模型，请修改这一项，或用 `--config` 传入自己的配置文件。`sv-abm run` 总是使用这一项，
因此 `SV_LLM_MODEL` 对它不起作用；`SV_LLM_MODEL`（默认 `gpt-4o`）只在你从自己的代码中调用任务的
`llm_f`、且没有调用 `llm_f.configure(model)` 时决定初始模型。

`chicago_segregation` 读取 `SV_LLM_API_KEY` 和 `SV_LLM_BASE_URL`（未设置时回退到
`OPENAI_API_KEY` / `OPENAI_BASE_URL`），可以来自环境变量，也可以写在仓库根目录下被 gitignore 的 `.env` 中；
它的模型同样取自 `behavior.llm_model`（仅当该值为空时才回退到 `SV_LLM_MODEL`）。
LLM 运行会产生费用，建议先用小规模的 `config.yaml`。

## 与 SocioVerse2 配合使用

SocioVerse2 的 11 个 `abm_*` 研究和 `chicago_schelling` 研究会把本仓库作为同级目录加载，不需要在那里 pip 安装本仓库。

```bash
git clone https://github.com/sii-research/SocioVerse2.git
git clone https://github.com/Lishi905/SocioVerse-ABM.git   # 目录名必须恰好是 SocioVerse-ABM
cd SocioVerse2
pip install -e ".[dev,chicago,workbench]"
pytest -q                                                # ABM 与 Chicago 测试会真正运行，而不是跳过
```

这与 SocioVerse2 README 中的安装命令一致。`chicago` 用于运行芝加哥相关测试；
`workbench` 还会运行 `hisim_roe` 和 `germany_auto_market` 中缺少它时会被跳过的测试。

- 两个目录放在同一层，名称必须恰好是 `SocioVerse2` 和 `SocioVerse-ABM`。
- 不要设置 `SV_ABM_ROOT`；同级目录布局会被自动识别。
- 如果 shell 里设置了 SOCKS 代理（`ALL_PROXY=socks5://...`），请取消它，或者 `pip install "httpx[socks]"`：
  内置的芝加哥模型即使离线运行也会创建 OpenAI 客户端，而 httpx 在缺少该 extra 时会拒绝 SOCKS 代理。

## 目录结构

```
socioverse_abm/            共享内核（见 socioverse_abm/README.md）
  behavior_engine/         类型化 Observation / Action、hybrid 路由、LLM 辅助函数
  social_env_engine/       连续空间任务使用的 ContinuousField2D
  scenario_engine/         sv-abm 与 SocioVerse2 共用的任务注册表
  eval.py                  规则与 LLM 的一致性评分
  runner.py                sv-abm 命令行
tasks/<family>/<task>/     每个任务一个目录：
  config.yaml              全部超参数
  agents.py                P：人群
  env.py                   E：observe / apply / advance / snapshot
  rule_f.py, llm_f.py      f：规则与 LLM 行为函数
  evaluate.py              B：结果指标
  model.py                 组装、运行循环与 registry.register(...)
  test_*.py                离线测试（规则模式与确定性的假 LLM）
  legacy/                  原贡献者的脚本，原样保留以备溯源
```

`legacy/` 目录是历史快照：不再维护，运行时也不会被导入。其中的脚本需要各自的环境，具体依赖因目录而异
（例如 Mesa 2.x 或 3.x、ndlib、Axelrod 库或 1.0 之前的 `openai` API），各 legacy 目录的 README 列出了所需依赖。
唯一的例外是 `tasks/organization/chicago_segregation/legacy/`，它会在运行时被导入：它是芝加哥任务和
SocioVerse2 的 `chicago_schelling` 实际运行的内置芝加哥模型。

## 数据

芝加哥任务在 [`tasks/organization/chicago_segregation/legacy/`](tasks/organization/chicago_segregation/legacy/)
下附带 2010 年和 2000 年的芝加哥普查区数据。这些数据**不**在 Apache-2.0 代码许可的覆盖范围内。
它们组合了属于公有领域的美国人口普查局数据、City of Chicago Data Portal 数据、来自 OpenStreetMap 的计数（ODbL），
以及来自 Mapping Inequality census crosswalk 的 HOLC 红线字段；后者采用 CC BY-NC 许可，因此**仅限非商业用途**。
详见 [`DATA_LICENSE.md`](tasks/organization/chicago_segregation/legacy/DATA_LICENSE.md) 和
[`DATA_README.md`](tasks/organization/chicago_segregation/legacy/processed_data/DATA_README.md)。
其他任务的人群均为合成生成，不附带数据。

## 引用

如果使用了 SocioVerse-ABM，请引用 SocioVerse2 技术报告：

```bibtex
@misc{zhang2026socioverse2,
  title         = {SocioVerse2: A Longitudinal Dynamic Social Simulation Framework under a Human-AI Co-evolutionary Paradigm},
  author        = {Xinnong Zhang and Jiayu Lin and Jia Wang and Yixu Huang and Xinyi Mou and Yingqian Wu and Jingcong Liang and Shijun Lei and Jianing Shi and Guanying Li and Siyuan Wang and Hanjia Lyu and Zhenfei Yin and Yunlu Yin and Siming Chen and Yulan He and Jiebo Luo and Xuanjing Huang and Liyin Jin and Baohua Zhou and Hanqi Yan and Zhongyu Wei},
  year          = {2026},
  eprint        = {2609.24911},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.24911}
}
```

## 许可

代码以 [Apache License 2.0](LICENSE) 发布，另见 [`NOTICE`](NOTICE)。改编自 Mesa 示例的三个 legacy 文件
保留了对 Core Mesa Team and contributors 的 Apache-2.0 署名。内置的芝加哥数据适用其各自的条款（见[数据](#数据)）。

## 致谢

本仓库中各模型与共享内核的原始实现由以下贡献者完成：

- **Xinnong Zhang**（[@Lishi905](https://github.com/Lishi905)）：SIR、Hegselmann-Krause、Sugarscape、共享内核与芝加哥隔离模型。
- **Shijun Lei**（[@ShijunLei-cn](https://github.com/ShijunLei-cn)）：Schelling 与 Civil Violence。
- **Jianing Shi**（[@Brishian427](https://github.com/Brishian427)）：Boids、NaSch 与 Social Force。
- **Chenyu Li**（[@if111111111111111111111](https://github.com/if111111111111111111111)）：Axelrod 与少数者博弈。
- **Zijian Ling**（[@Georgelingzj](https://github.com/Georgelingzj)）：Lux-Marchesi。
- **Jia Wang**（[@JiaWANG-TJ](https://github.com/JiaWANG-TJ)）：少数者博弈的整合版模拟。

原始脚本保存在各任务的 `legacy/` 目录中。Schelling 与 Civil Violence 的 legacy 模型基于
[Mesa](https://github.com/projectmesa/mesa) 的示例（Apache-2.0，Core Mesa Team and contributors）；
Axelrod 的 legacy 代码使用了 [Axelrod 库](https://github.com/Axelrod-Python/Axelrod)。

问题与合作：contact@socioverse.fudan-disc.com。
