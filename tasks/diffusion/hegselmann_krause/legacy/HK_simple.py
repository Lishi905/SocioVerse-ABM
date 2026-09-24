# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
import os
import random
import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")

import ndlib.models.ModelConfig as mc
import ndlib.models.opinions as opn
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution


def cluster_opinions(values, tol=1e-3):
    vals = np.sort(np.asarray(values))
    clusters, cur = [], [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= tol:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return [float(np.mean(c)) for c in clusters]


def run_hk(N=100, epsilon=0.05, steps=200, seed=42, out_pdf="./res_figs_simple/hk_eps005.pdf"):
    # 固定随机种子（NDlib自身&我们自定义初始化都会用到）
    random.seed(seed)
    np.random.seed(seed)

    # 确保输出目录存在
    out_dir = os.path.dirname(out_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 完全图：所有智能体两两相连
    g = nx.complete_graph(N)

    model = opn.HKModel(g)

    # 设定 epsilon
    cfg = mc.Configuration()
    cfg.add_model_parameter("epsilon", float(epsilon))

    # 初始化：覆盖 NDlib 默认的，把意见设为 U[0,1]
    model.set_initial_status(cfg)
    init = {n: random.random() for n in g.nodes()}
    model.status = init.copy()
    model.initial_status = init.copy()

    # 进行 steps 次迭代；node_status=True 以便拿到每一步每个智能体的意见
    iterations = model.iteration_bunch(steps, node_status=True)

    # 画出“意见随时间演化”的轨迹图并保存
    viz = OpinionEvolution(model, iterations)
    viz.plot(out_pdf)

    # 统计最终簇中心用于判断：碎片化/极化/共识
    if isinstance(model.status, dict) and len(model.status) == g.number_of_nodes():
        last_vals = [float(model.status[n]) for n in g.nodes()]
    else:
        final_status = iterations[-1]["status"]
        # NDlib 可能把节点ID序列化成字符串
        fs = {int(k): float(v) for k, v in final_status.items()}
        last_vals = [fs[n] for n in g.nodes()]
    clusters = cluster_opinions(last_vals, tol=1e-3)
    return clusters


if __name__ == "__main__":
    c1 = run_hk(N=100, epsilon=0.01, steps=400, seed=1, out_pdf="./res_figs_simple/hk_eps001.png")
    print(f"epsilon=0.01: clusters={len(c1)}, centers={np.round(c1, 3)}")

    c2 = run_hk(N=100, epsilon=0.15, steps=400, seed=2, out_pdf="./res_figs_simple/hk_eps015.png")
    print(f"epsilon=0.15: clusters={len(c2)}, centers={np.round(c2, 3)}")

    c3 = run_hk(N=100, epsilon=0.25, steps=400, seed=2, out_pdf="./res_figs_simple/hk_eps025.png")
    print(f"epsilon=0.25: clusters={len(c3)}, centers={np.round(c3, 3)}")
