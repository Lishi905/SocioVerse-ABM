# NOTE (SocioVerse-ABM public release): legacy snapshot kept verbatim for provenance;
# not maintained (see the README in this legacy folder). Comments and messages may be in
# the original authors' language (Chinese).
"""
Sugarscape (G1) with Mesa
-------------------------
- Movement: each citizen looks up to `vision` cells in von Neumann directions,
  moves to the empty location with the most sugar (ties broken randomly, prefer
  closest), then harvests all sugar there.
- Metabolism: each step, citizen loses `metabolism` sugar. If sugar <= 0, dies;
  the model DO NOT immediately respawns a new citizen with random traits at a random
  empty cell to keep population constant.
- Environment: sugar patches regrow `regrowth_rate` per step up to `capacity`.

Open the URL printed in the console (default http://127.0.0.1:8521)
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional, Dict, Any

from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# --- Environment: Sugar Patch -------------------------------------------------
class SugarPatch(Agent):
    """Immobile patch that accumulates sugar up to a capacity."""
    def __init__(self, unique_id: int, model: "Sugarscape", capacity: int, regrowth_rate: int):
        super().__init__(unique_id, model)
        self.capacity = capacity
        self.regrowth_rate = regrowth_rate
        self.sugar = random.randint(0, capacity)  # random initial stock

    def step(self) -> None:
        if self.model.regrow_sugar:
            self.sugar = min(self.capacity, self.sugar + self.regrowth_rate)

# --- Citizens -----------------------------------------------------------------
class Citizen(Agent):
    def __init__(self, unique_id: int, model: "Sugarscape", metabolism: int, vision: int, sugar: Optional[int] = None):
        super().__init__(unique_id, model)
        self.metabolism = metabolism
        self.vision = vision
        self.sugar = sugar if sugar is not None else random.randint(5, 25)

    # Utility: check if a grid cell is free of other citizens
    def _cell_is_free(self, pos: Tuple[int, int]) -> bool:
        contents = self.model.grid.get_cell_list_contents(pos)
        for obj in contents:
            if isinstance(obj, Citizen):
                return False
        return True

    def _best_locations_in_vision(self) -> List[Tuple[int, int]]:
        """Return candidate locations (pos) that maximize patch sugar among visible empty cells.
        Von Neumann neighborhood along 4 cardinal directions up to `vision`.
        Ties: keep all with same max sugar; caller can break ties by distance/random.
        """
        candidates: List[Tuple[int, int]] = []
        max_sugar = -1
        x, y = self.pos
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in dirs:
            for r in range(1, self.vision + 1):
                nx = (x + dx * r) % self.model.grid.width
                ny = (y + dy * r) % self.model.grid.height
                pos = (nx, ny)
                if not self._cell_is_free(pos):
                    # If blocked by a citizen, stop scanning further in this direction
                    break
                # read sugar on that patch
                sugar_on_patch = 0
                for obj in self.model.grid.get_cell_list_contents(pos):
                    if isinstance(obj, SugarPatch):
                        sugar_on_patch = obj.sugar
                        break
                if sugar_on_patch > max_sugar:
                    max_sugar = sugar_on_patch
                    candidates = [pos]
                elif sugar_on_patch == max_sugar:
                    candidates.append(pos)
        # If no visible free cell (shouldn't happen), stay put
        if not candidates:
            candidates = [self.pos]
        return candidates

    def step(self) -> None:
        # 1) Move to best visible location (richest sugar; prefer nearest; break ties randomly)
        candidates = self._best_locations_in_vision()
        # prefer minimum Manhattan distance
        def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            dx = min(abs(a[0]-b[0]), self.model.grid.width - abs(a[0]-b[0]))
            dy = min(abs(a[1]-b[1]), self.model.grid.height - abs(a[1]-b[1]))
            return dx + dy
        best_dist = min(manhattan(self.pos, p) for p in candidates)
        nearest = [p for p in candidates if manhattan(self.pos, p) == best_dist]
        dest = random.choice(nearest)
        self.model.grid.move_agent(self, dest)

        # 2) Harvest sugar in the destination cell
        for obj in self.model.grid.get_cell_list_contents(dest):
            if isinstance(obj, SugarPatch):
                self.sugar += obj.sugar
                obj.sugar = 0
                break

        # 3) Metabolize and possibly die
        self.sugar -= self.metabolism
        if self.sugar <= 0:
            # Remove and respawn a new citizen to keep population constant
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            # Not respawn citizen setting
            # self.model.spawn_citizen()


# --- Helpers -------------------------------------------------------------------
def gini(values: List[int]) -> float:
    """Compute Gini coefficient for a list of non-negative values.
    Returns 0 for empty list and handles all-zero safely.
    """
    n = len(values)
    if n == 0:
        return 0.0
    # sort ascending
    vals = sorted(max(0, int(v)) for v in values)
    if vals[-1] == 0:
        return 0.0
    cum = 0
    weighted_sum = 0
    for i, v in enumerate(vals, start=1):
        cum += v
        weighted_sum += i * v
    total = cum
    # Gini = (2*sum(i*v)/n - (n+1)*mean) / (n*mean)
    mean = total / n
    return max(0.0, min(1.0, (2 * weighted_sum / n - (n + 1) * mean) / (n * mean)))


# --- Model --------------------------------------------------------------------
class Sugarscape(Model):
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        n_citizens: int = 200,
        patch_capacity_max: int = 4,
        regrowth_rate: int = 1,
        regrow_sugar: bool = True,
        metabolism_range: Tuple[int, int] = (1, 4),
        vision_range: Tuple[int, int] = (1, 6),
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.regrow_sugar = regrow_sugar
        self.regrowth_rate = regrowth_rate
        self.patch_capacity_max = patch_capacity_max
        self.n_citizens = n_citizens
        
        # --- Create sugar patches: two layered hills + diagonal connector --------------
        self.patches: List[SugarPatch] = []
        uid = 0

        # 两个山丘的圆心（对角分布，调这两个点就能整体平移）
        cx1, cy1 = int(width * 0.28), int(height * 0.28)   # 左下
        cx2, cy2 = int(width * 0.72), int(height * 0.72)   # 右上

        # 分层半径（单位：格子）。增大/减小它们来控制层的大小与“台地感”
        R0 = int(0.06 * min(width, height))   # 核心圈：容量=cap_max
        R1 = int(0.18 * min(width, height))   # 次核心：容量=cap_max-1（至少1）
        R2 = int(0.27 * min(width, height))   # 中环：容量=2（浅绿）
        R3 = int(0.45 * min(width, height))  # 外环：容量=1（浅绿）
        # 外部为0 → 纯白

        # 连接带宽度（单位：格子）。加粗/变细连接区域
        band_width = int(0.10 * min(width, height))

        # 点到线段距离（用于画对角连接带）
        def dist_point_to_segment(px, py, ax, ay, bx, by):
            apx, apy = px - ax, py - ay
            abx, aby = bx - ax, by - ay
            ab2 = abx*abx + aby*aby
            if ab2 == 0:
                return math.hypot(apx, apy), 0.0
            t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
            qx, qy = ax + t*abx, ay + t*aby
            return math.hypot(px - qx, py - qy), t

        for x in range(width):
            for y in range(height):
                # 到两座山的最近距离 → 分层圆丘
                d1 = math.hypot(x - cx1, y - cy1)
                d2 = math.hypot(x - cx2, y - cy2)
                d = min(d1, d2)

                if d <= R0:
                    level = self.patch_capacity_max
                elif d <= R1:
                    level = max(1, self.patch_capacity_max - 1)
                elif d <= R2:
                    level = max(1, self.patch_capacity_max - 2)
                elif d <= R3:
                    level = 1
                else:
                    level = 0  # 外围留白（糖=0 显示白色）

                # 对角连接带：让靠近两中心连线的格子至少为1
                dist_line, t = dist_point_to_segment(x, y, cx1, cy1, cx2, cy2)
                if dist_line <= band_width and 0.0 <= t <= 1.0:
                    level = max(level, 1)

                # 截断到合法范围并放置地块
                level = max(0, min(self.patch_capacity_max, int(level)))
                patch = SugarPatch(uid, self, capacity=level, regrowth_rate=self.regrowth_rate)
                uid += 1
                self.grid.place_agent(patch, (x, y))
                self.patches.append(patch)

        
        # Spawn citizens
        for _ in range(n_citizens):
            self.spawn_citizen(metabolism_range, vision_range)

        # Metrics
        self.datacollector = DataCollector(
            model_reporters={
                "MeanSugar": lambda m: (sum(a.sugar for a in m.citizens) / len(m.citizens)) if m.citizens else 0,
                "Alive": lambda m: len(m.citizens),
                "Gini": lambda m: gini([a.sugar for a in m.citizens]),
            },
            agent_reporters={"Sugar": lambda a: getattr(a, "sugar", None)},
        )

    @property
    def citizens(self) -> List[Citizen]:
        return [a for a in self.schedule.agents if isinstance(a, Citizen)]

    def random_empty_cell(self) -> Tuple[int, int]:
        while True:
            pos = (self.random.randrange(self.width), self.random.randrange(self.height))
            # cell is considered empty if it has no citizens (patches always exist)
            if all(not isinstance(o, Citizen) for o in self.grid.get_cell_list_contents(pos)):
                return pos

    def spawn_citizen(
        self,
        metabolism_range: Tuple[int, int] = (1, 4),
        vision_range: Tuple[int, int] = (1, 6),
    ) -> None:
        uid = self.next_id()
        metabolism = self.random.randint(metabolism_range[0], metabolism_range[1])
        vision = self.random.randint(vision_range[0], vision_range[1])
        agent = Citizen(uid, self, metabolism=metabolism, vision=vision)
        self.schedule.add(agent)
        self.grid.place_agent(agent, self.random_empty_cell())

    def step(self) -> None:
        # 1) Grow sugar on all patches
        for p in self.patches:
            p.step()
        # 2) Activate citizens in random order
        self.schedule.step()
        # 3) Collect metrics
        self.datacollector.collect(self)

# --- Visualization -------------------------------------------------------------
try:
    from mesa.visualization import CanvasGrid, ChartModule, ModularServer
    from mesa.visualization.modules import TextElement

    def citizen_portrayal(agent: Agent) -> Optional[Dict[str, Any]]:
        if isinstance(agent, Citizen):
            # 颜色仅反映财富：越富→越红；越穷→越白
            # R固定为255，G/B随财富下降（白→粉→红）
            try:
                citizens = agent.model.citizens
                if citizens:
                    mn = min(c.sugar for c in citizens)
                    mx = max(c.sugar for c in citizens)
                else:
                    mn, mx = 0, 1
            except Exception:
                mn, mx = 0, 1

            norm = 0.0 if mx == mn else max(0.0, min(1.0, (agent.sugar - mn) / (mx - mn)))
            gb = int(round(220 * (1.0 - norm)))        # 穷→255(白)，富→0(红)
            color = f"rgb(255,{gb},{gb})"

            return {
                "Shape": "circle",    # 小圆点；如果你想换成五角星，需要前端自定义形状支持
                "r": 0.5,               # 固定小半径（像素），明显小于单元格
                "Filled": "true",
                "Layer": 1,
                "Color": color,
            }

        if isinstance(agent, SugarPatch):
            # 保持原来的地块渲染（绿色深浅反映糖量）
            if agent.sugar <= 0:
                color = "rgb(255,255,255)"
            else:
                # green = int(255 * agent.sugar / max(1, agent.capacity))
                green = (1 - (agent.sugar-1) / max(1, agent.capacity)) * 255
                green = max(0, min(255, green))
                color = f"rgb(0,{green},0)"  # green intensity reflects sugar
            return {
                "Shape": "rect",
                "w": 1,
                "h": 1,
                "Filled": "true",
                "Layer": 0,
                "Color": color,
            }
        return None
    

    def make_server(width=50, height=50, n_citizens=200) -> ModularServer:
        grid_viz = CanvasGrid(citizen_portrayal, width, height, 600, 600)
        chart = ChartModule([
            {"Label": "MeanSugar", "Color": "#333"},
            {"Label": "Alive", "Color": "#999"},
        ])
        gini_chart = ChartModule([
            {"Label": "Gini", "Color": "#1f77b4"},
        ])
        
        class WealthHistogram(TextElement):
            def __init__(self, bins: int = 10, width: int = 40):
                self.bins = bins
                self.width = width
                super().__init__()

            def render(self, model: Sugarscape) -> str:
                vals = [a.sugar for a in model.citizens]
                if not vals:
                    return "Wealth Histogram: (no agents)"
                mn, mx = min(vals), max(vals)
                if mx == mn:
                    return f"Wealth Histogram: all {mx}"
                # build bins
                step = max(1, (mx - mn) // self.bins or 1)
                edges = [mn + i * step for i in range(self.bins)] + [mx + 1]
                counts = [0] * self.bins
                for v in vals:
                    # find bin index
                    idx = min(self.bins - 1, max(0, (v - mn) // step))
                    counts[idx] += 1
                # normalize to ascii bar
                mcount = max(counts)
                lines = []
                for i, c in enumerate(counts):
                    left = edges[i]
                    right = edges[i+1] - 1
                    bar = "#" * max(1, int(self.width * c / mcount))
                    lines.append(f"[{left:>3}–{right:>3}] {bar} {c}")
                g = gini(vals)
                return "<pre>Gini: {:.3f}\n{}\n</pre>".format(g, "\n".join(lines))

        wealth_hist = WealthHistogram(bins=10)
        
        server = ModularServer(
            Sugarscape,
            [grid_viz, chart, gini_chart, wealth_hist],
            "Sugarscape (Mesa)",
            {
                "width": width,
                "height": height,
                "n_citizens": n_citizens,
                "patch_capacity_max": 4,
                "regrowth_rate": 1,
                "regrow_sugar": True,
                "metabolism_range": (1, 4),
                "vision_range": (1, 6),
            },
        )
        return server
except Exception:
    # Visualization is optional; keep the headless model usable without viz deps
    ModularServer = None
    def make_server(*args, **kwargs):
        raise RuntimeError("Visualization components not available. Ensure mesa.visualization is installed.")

# --- BatchRunner example (deprecated) -------------------------------------------
try:
    from mesa.batchrunner import BatchRunner
except Exception:
    BatchRunner = None


def run_batch_example() -> None:
    if BatchRunner is None:
        print("BatchRunner not available; install mesa fully.")
        return
    fixed_params = {"width": 50, "height": 50, "n_citizens": 200}
    variable_params = {
        "regrowth_rate": [1, 2],
        "patch_capacity_max": [3, 4, 5],
    }
    br = BatchRunner(
        Sugarscape,
        variable_parameters=variable_params,
        fixed_parameters=fixed_params,
        iterations=3,
        max_steps=200,
        model_reporters={"MeanSugar": lambda m: (sum(a.sugar for a in m.citizens)/len(m.citizens)) if m.citizens else 0},
    )
    br.run_all()
    df = br.get_model_vars_dataframe()
    try:
        print(df.groupby(["regrowth_rate", "patch_capacity_max"]).MeanSugar.mean())
    except Exception:
        print(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sugarscape (Mesa) – interactive server or headless batch")
    parser.add_argument("--mode", choices=["server", "batch"], default="server")
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--agents", type=int, default=200)
    args = parser.parse_args()

    if args.mode == "server":
        if ModularServer is None:
            raise SystemExit("Visualization not available. Run with --mode batch or install mesa.")
        server = make_server(width=args.width, height=args.height, n_citizens=args.agents)
        server.port = 8521
        server.launch()
    else:
        # Headless example
        model = Sugarscape(width=args.width, height=args.height, n_citizens=args.agents)
        for _ in range(200):
            model.step()
        print("Final mean sugar:", sum(a.sugar for a in model.citizens) / len(model.citizens))
        # Or run a grid of experiments
        run_batch_example()
