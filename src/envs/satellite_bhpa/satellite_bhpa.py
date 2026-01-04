import torch
import numpy as np
import gym
from gym import spaces
from scipy.constants import speed_of_light, Boltzmann
from scipy.special import j1
from typing import Tuple, Dict, Any, Union, List

# 绘图依赖（可选，缺失时注释绘图相关代码）
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon, Patch
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    import os
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# 引入项目的多智能体环境抽象类
from envs.multiagentenv import MultiAgentEnv


# ======================================================================
# 内联工具函数（独立无依赖）
# ======================================================================
SCI_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
    "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148"
]


def calculate_sinr(signal_power: np.ndarray,
                   noise_power: float,
                   interference_power: np.ndarray) -> np.ndarray:
    """SINR = P * |h|^2 / (sigma^2 + I)"""
    return signal_power / (noise_power + interference_power + 1e-12)


def calculate_throughput(sinr: np.ndarray, bandwidth: float) -> np.ndarray:
    """Shannon capacity: B * log2(1 + SINR)"""
    return bandwidth * np.log2(1 + sinr)


def calculate_load_balance_diff(loads: np.ndarray) -> float:
    """Max(Load) - Min(Load)"""
    if len(loads) == 0:
        return 0.0
    return float(np.max(loads) - np.min(loads))


def calculate_delay_fairness_diff(delays: np.ndarray) -> float:
    """Max(Delay) - Min(Delay)，过滤无效延迟"""
    if len(delays) == 0:
        return 0.0
    valid_delays = delays[delays >= 0]
    if len(valid_delays) == 0:
        return 0.0
    return float(np.max(valid_delays) - np.min(valid_delays))


def generate_satellite_positions(
    num_satellites: int,
    dist: float,
    layout: str = "hex_grid",
    center: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """生成卫星星座位置"""
    if layout == "ring":
        if num_satellites == 1:
            return np.array([center])
        angle_step = 2 * np.pi / num_satellites
        angles = np.arange(num_satellites) * angle_step
        x = center[0] + dist * np.cos(angles)
        y = center[1] + dist * np.sin(angles)
        z = np.full(num_satellites, center[2])
        return np.column_stack((x, y, z))
    elif layout == "hex_grid":
        return _generate_hex_grid_satellites(num_satellites, dist, center)
    else:
        raise ValueError(f"Unknown layout: {layout}")


def _generate_hex_grid_satellites(num_satellites: int, dist: float, center: np.ndarray) -> np.ndarray:
    """内部函数：生成交错网格分布的卫星位置"""
    aspect_ratio = 1.0
    cols = int(np.sqrt(num_satellites * aspect_ratio))
    if cols == 0:
        cols = 1

    positions = []
    dx = dist
    dy = dist * np.sqrt(3) / 2

    count = 0
    row = 0
    while count < num_satellites:
        col_limit = cols
        for c in range(col_limit):
            if count >= num_satellites:
                break
            offset_x = (dx / 2.0) if (row % 2 == 1) else 0.0
            x = c * dx + offset_x
            y = row * dy
            z = 0.0
            positions.append([x, y, z])
            count += 1
        row += 1

    pos_array = np.array(positions)
    centroid = np.mean(pos_array, axis=0)
    pos_array = pos_array - centroid + center
    pos_array[:, 2] = center[2]
    return pos_array


def generate_hexagonal_grid(
    num_satellites: int,
    satellite_positions: np.ndarray,
    sat_service_radius: float,
    cell_radius: float,
) -> np.ndarray:
    """生成 pointy-topped 六边形小区网格"""
    del num_satellites  # 未使用，仅保持签名一致
    min_x = np.min(satellite_positions[:, 0]) - sat_service_radius
    max_x = np.max(satellite_positions[:, 0]) + sat_service_radius
    min_y = np.min(satellite_positions[:, 1]) - sat_service_radius
    max_y = np.max(satellite_positions[:, 1]) + sat_service_radius

    R = cell_radius
    dx = 3 * R
    dy = np.sqrt(3) / 2 * R

    cols = int((max_x - min_x) / dx) + 6
    rows = int((max_y - min_y) / dy) + 6

    start_x = min_x - 3 * dx
    start_y = min_y - 3 * dy

    cells = []
    for r in range(rows):
        for c in range(cols):
            offset = (dx / 2.0) if (r % 2 == 1) else 0.0
            cx = start_x + c * dx + offset
            cy = start_y + r * dy
            cz = 0.0
            point = np.array([cx, cy])
            dists = np.linalg.norm(satellite_positions[:, :2] - point, axis=1)
            if np.any(dists <= sat_service_radius + R * 0.8):
                cells.append([cx, cy, cz])
    return np.array(cells)


def map_cells_to_satellites(
    satellite_positions: np.ndarray,
    cell_positions: np.ndarray,
    sat_service_radius: float,
) -> list:
    """映射小区到卫星"""
    num_satellites = len(satellite_positions)
    mapping = [[] for _ in range(num_satellites)]
    dists = np.linalg.norm(
        satellite_positions[:, np.newaxis, :2] - cell_positions[np.newaxis, :, :2],
        axis=2,
    )
    for sat_id in range(num_satellites):
        covered_indices = np.where(dists[sat_id] <= sat_service_radius)[0]
        mapping[sat_id] = covered_indices.tolist()
    return mapping


def plot_hexagonal_topology(
    satellite_positions: np.ndarray,
    cell_positions: np.ndarray,
    mapping: list,
    sat_service_radius: float,
    cell_radius: float,
    save_path: str = None,
) -> None:
    """绘制六边形拓扑（可选，缺失matplotlib时跳过）"""
    if not PLOT_AVAILABLE:
        print("Warning: matplotlib not available, skip plotting")
        return
    
    del mapping  # 这里只使用几何关系着色

    num_satellites = len(satellite_positions)
    num_cells = len(cell_positions)

    fig, ax = plt.subplots(figsize=(12, 10))

    dists = np.linalg.norm(
        satellite_positions[:, np.newaxis, :2] - cell_positions[np.newaxis, :, :2],
        axis=2,
    )
    main_owners = np.argmin(dists, axis=0)

    patches = []
    face_colors = []
    for i in range(num_cells):
        x, y, _ = cell_positions[i]
        owner = main_owners[i]
        if dists[owner, i] <= sat_service_radius * 1.05:
            color = SCI_COLORS[owner % len(SCI_COLORS)]
        else:
            color = "#E0E0E0"
        hex_patch = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=cell_radius,
            orientation=np.pi / 6,
        )
        patches.append(hex_patch)
        face_colors.append(color)

    p_collection = PatchCollection(patches, match_original=False)
    p_collection.set_facecolors(face_colors)
    p_collection.set_edgecolor((0.3, 0.3, 0.3, 0.5))
    p_collection.set_linewidth(0.2)
    p_collection.set_alpha(0.85)
    ax.add_collection(p_collection)

    legend_elements = []
    for sat_id in range(num_satellites):
        sat_pos = satellite_positions[sat_id]
        c = SCI_COLORS[sat_id % len(SCI_COLORS)]
        circle = Circle(
            (sat_pos[0], sat_pos[1]),
            sat_service_radius,
            color=c,
            fill=False,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(circle)
        ax.scatter(
            sat_pos[0],
            sat_pos[1],
            s=350,
            marker="*",
            color=c,
            edgecolors="white",
            linewidth=1.0,
            zorder=10,
        )
        ax.text(
            sat_pos[0],
            sat_pos[1],
            f"{sat_id+1}",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
            fontsize=9,
            zorder=11,
        )
        if num_satellites <= 6:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor=c,
                    markersize=12,
                    label=f"Sat {sat_id+1} Coverage",
                )
            )

    if num_satellites > 6:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                markersize=15,
                label="Satellite",
            ),
            Patch(
                facecolor="gray", edgecolor="k", alpha=0.5, label="Served Beam Area"
            ),
            Line2D(
                [0], [0], linestyle="--", color="gray", linewidth=2, label="Service Boundary"
            ),
        ]
    else:
        legend_elements.append(
            Patch(
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.3,
                label="Unserved Area",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.25, 1),
        title="Network Entities",
        fancybox=True,
        shadow=True,
    )

    ax.set_aspect("equal")
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - 5, xlim[1] + 5)
    ax.set_ylim(ylim[0] - 5, ylim[1] + 5)
    ax.set_title("Multi-Beam Satellite Constellation Coverage", pad=20)
    ax.set_xlabel("X Coordinate (km)")
    ax.set_ylabel("Y Coordinate (km)")

    plt.tight_layout()
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

class ScalarNormalizer:
    """
    用于对标量指标进行动态归一化。
    维护一个指数移动平均 (EMA) 值，输出 value / (running_avg + eps)。
    这样可以将不同量纲的物理量转化为相对其历史平均水平的无量纲比率。
    """
    def __init__(self, alpha: float = 0.95, epsilon: float = 1e-6):
        self.alpha = alpha
        self.mean = 0.0
        self.epsilon = epsilon
        self.count = 0

    def normalize(self, x: float) -> float:
        # 仅处理非负的量级，取绝对值是个保险，虽然物理量通常非负
        val = float(x)
        if self.count == 0:
            self.mean = val
        else:
            # 更新移动平均：NewMean = alpha * OldMean + (1-alpha) * Current
            # 注意：这里alpha定义为历史权重的衰减系数。
            # 如果 alpha=0.95，表示历史平均占 95%，当前值占 5%。
            self.mean = self.alpha * self.mean + (1 - self.alpha) * val
        
        self.count += 1
        # 返回归一化后的值
        return val / (abs(self.mean) + self.epsilon)
    
    def reset(self):
        self.mean = 0.0
        self.count = 0

class TrafficGenerator:
    """模拟地面流量生成"""
    def __init__(self, num_cells, time_slots, seed=42):
        self.rng = np.random.RandomState(seed)
        self.num_cells = num_cells
        self.total_time = time_slots
        self.base_traffic = 10.0  # Mbps
        
    def get_traffic_at_time(self, t: int) -> np.ndarray:
        # 模拟时空分布不均：不同小区有不同的相位和幅度
        spatial_factor = self.rng.uniform(0.5, 2.0, self.num_cells)
        # 时间变化：正弦波模拟昼夜/忙闲
        temporal_factor = np.sin(2 * np.pi * t / 1000) + 1.5 
        noise = self.rng.normal(0, 0.5, self.num_cells)
        
        traffic = self.base_traffic * spatial_factor * temporal_factor + noise
        return np.maximum(traffic, 0.1)  # 保证非负

class SatelliteEnv(MultiAgentEnv):
    """
    修改后的卫星环境：支持联合波束跳变与功率分配 (Joint BH & PA)
    动作空间改为连续向量 (N, 2*C)，前C维控制波束选择，后C维控制功率。
    """
    def __init__(self, **kwargs):
        # 调用父类初始化以设置 self.args 等属性
        super().__init__(**kwargs)
        
        env_args = kwargs.get("env_args", {})
        self.seed_val = env_args.get("seed", 42)
        self.rng = np.random.RandomState(self.seed_val)
        
        # --- 系统参数 ---
        self.num_satellites: int = env_args.get("satellite_num", 6)
        self.beams_per_satellite: int = env_args.get("beam_per_satellite", 4)  # K
        self.cells_per_satellite: int = env_args.get("cell_per_satellite", 10) # C
        
        # --- 物理参数 ---
        self.fc = env_args.get("frequency", 20e9)
        self.bw = env_args.get("bandwidth", 500e6)
        self.lambda_ = speed_of_light / self.fc
        self.noise_power = Boltzmann * env_args.get("noise_temp", 300) * self.bw
        self.total_power_watt = 10 ** (env_args.get("total_power_dbw", 20) / 10) # P_total
        self.G_tx_max_db = env_args.get("max_transmit_gain_dbi", 35.0)
        self.G_rx_max_db = env_args.get("receive_gain_dbi", 30.0)
        
        # --- 仿真参数 ---
        self.max_steps = env_args.get("episode_limit", 50)
        self.sat_service_radius = env_args.get("service_radius", 150.0)
        self.cell_radius = env_args.get("cell_radius", 15.0)
        self.inter_sat_dist = env_args.get("inter_sat_dist", 160.0)
        self.time_slot_duration = env_args.get("time_slot_duration", 1e-3)

        # --- 初始化动态归一化器 ---
        # alpha=0.99 表示平均值变化较平滑，适合作为稳定的基准
        self.load_normalizer = ScalarNormalizer(alpha=0.99, epsilon=1e-2)
        self.delay_normalizer = ScalarNormalizer(alpha=0.99, epsilon=1e-2)
        self.th_normalizer = ScalarNormalizer(alpha=0.99, epsilon=1e-2)

        # --- 初始化拓扑 (保持原逻辑) ---
        print(f"[SatEnv] Initializing Topology (N={self.num_satellites}, K={self.beams_per_satellite}, C={self.cells_per_satellite})...")
        # 假设 generate_satellite_positions 等函数在外部定义可用
        from envs.satellite_bhpa.satellite_bhpa import generate_satellite_positions, generate_hexagonal_grid, map_cells_to_satellites, TrafficGenerator
        
        self.sat_positions_init = generate_satellite_positions(
            self.num_satellites, dist=self.inter_sat_dist, layout='hex_grid'
        )
        self.cell_positions = generate_hexagonal_grid(
            self.num_satellites, self.sat_positions_init, self.sat_service_radius, self.cell_radius
        )
        self.V = len(self.cell_positions)
        self.mapping_table = map_cells_to_satellites(
            self.sat_positions_init, self.cell_positions, self.sat_service_radius
        )
        
        # 构建映射矩阵 (N, C)
        self.mapping_matrix = np.full((self.num_satellites, self.cells_per_satellite), 0, dtype=int)
        for sat_index in range(self.num_satellites):
            cells = self.mapping_table[sat_index]
            valid = min(len(cells), self.cells_per_satellite)
            self.mapping_matrix[sat_index, :valid] = cells[:valid]
            # 填充剩余位置（避免索引越界，指向0号小区，但在mask中会被置0）
            if valid < self.cells_per_satellite:
                self.mapping_matrix[sat_index, valid:] = cells[0] if cells else 0

        # --- 预计算信道 ---
        self.H_tensor = self._precompute_channel_tensor()

        # --- 状态与动作空间重定义 ---
        self.traffic_gen = TrafficGenerator(self.V, self.max_steps, seed=self.seed_val)
        self.current_step_idx = 0
        
        self.local_queues = np.zeros((self.num_satellites, self.cells_per_satellite))
        self.local_delays = np.zeros((self.num_satellites, self.cells_per_satellite))
        self.last_bh_mask = np.zeros((self.num_satellites, self.cells_per_satellite))
        
        # [NEW] 动作空间：连续向量 (2 * C,)
        # 前 C 个：波束选择 Logits (Top-K 激活)
        # 后 C 个：功率分配 Logits (Softmax/Normalize 分配功率)
        self.act_dim = self.cells_per_satellite * 2
        self.action_spaces = [
            spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
            for _ in range(self.num_satellites)
        ]
        
        # 观测空间：(C*3,) 流量 + 信道 + 队列
        self.obs_dim = self.cells_per_satellite * 3
        self.obs_spaces = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.num_satellites)
        ]
        
        self.episode_limit = self.max_steps
        self.n_agents = self.num_satellites

    # ... _precompute_channel_tensor, _get_current_channel_matrix, _get_observation 保持原样 ...
    def _precompute_channel_tensor(self) -> np.ndarray:
        # (复用原代码逻辑)
        time_horizon, num_satellites, num_global_cells = self.max_steps, self.num_satellites, self.V
        time_vec = np.arange(time_horizon).reshape(-1, 1, 1)
        velocity = np.array([7500.0, 0, 0])
        sat_traj = self.sat_positions_init.reshape(1, num_satellites, 3) + (velocity * time_vec * self.time_slot_duration)
        cell_pos = self.cell_positions.reshape(1, 1, num_global_cells, 3)
        vec = cell_pos - sat_traj.reshape(time_horizon, num_satellites, 1, 3)
        dist = np.linalg.norm(vec, axis=3) + 1e-9
        G_tx = 10 ** (self.G_tx_max_db / 10)
        G_rx = 10 ** (self.G_rx_max_db / 10)
        path_loss = self.lambda_ / (4 * np.pi * dist)
        H_mag = np.sqrt(G_tx * G_rx) * path_loss
        return H_mag

    def _get_current_channel_matrix(self) -> np.ndarray:
        t = min(self.current_step_idx, self.max_steps - 1)
        return self.H_tensor[t]

    def _get_observation(self) -> np.ndarray:
        global_traf = self.traffic_gen.get_traffic_at_time(self.current_step_idx)
        local_traf = global_traf[self.mapping_matrix]
        H_global = self._get_current_channel_matrix()
        local_chan = np.take_along_axis(H_global, self.mapping_matrix, axis=1)
        local_q = self.local_queues
        return np.stack([local_traf, local_chan, local_q], axis=-1)
    
    def _calculate_interference_vectorized(self, active_mask, power_alloc) -> np.ndarray:
        # (复用原代码逻辑，注意 power_alloc 现已包含真实功率值)
        H_global = self._get_current_channel_matrix()
        power_global = np.zeros((self.num_satellites, self.V))
        np.put_along_axis(power_global, self.mapping_matrix, power_alloc, axis=1)
        
        beta = 0.05 
        avg_path_loss = np.mean(H_global ** 2)
        total_tx_power = np.sum(power_alloc)
        interference_floor = total_tx_power * avg_path_loss * beta
        interference = np.full((self.num_satellites, self.cells_per_satellite), interference_floor)
        
        # 这里可以加入更精确的互干扰计算（如果是计算密集型可省略）
        return interference

    def reset(self):
        self.current_step_idx = 0
        self.local_queues.fill(0)
        self.local_delays.fill(0)
        self.last_bh_mask.fill(0)
        self.traffic_gen = self.traffic_gen.__class__(self.V, self.max_steps, seed=self.rng.randint(10000))
        return self.get_obs()

    def step(self, actions: Union[np.ndarray, List[float]]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        [MODIFIED] 优化的 step 函数
        Args:
            actions: (N_agents, 2 * C) 的连续动作矩阵/列表
                     Front C: Beam Selection Logits
                     Back C: Power Allocation Params
        """
        # 0. 数据格式转换
        if isinstance(actions, list):
            actions = np.array(actions)
        # 确保维度为 (N, 2C)
        actions = actions.reshape(self.num_satellites, -1)
        
        C = self.cells_per_satellite
        K = self.beams_per_satellite
        
        # 1. 拆分动作
        beam_logits = actions[:, :C]       # (N, C)
        power_params = actions[:, C:]      # (N, C)
        
        # 2. 波束选择 (Beam Selection) - Top-K 策略
        # 每个卫星最多只能点亮 K 个波束
        bh_mask = np.zeros((self.num_satellites, C), dtype=int)
        # 对 logits 进行排序，取最大的 K 个索引
        # argsort 默认升序，取后 K 个
        top_k_indices = np.argsort(beam_logits, axis=1)[:, -K:]
        # 利用 numpy 高级索引置 1
        np.put_along_axis(bh_mask, top_k_indices, 1, axis=1)
        self.last_bh_mask = bh_mask
        
        # 3. 功率分配 (Power Allocation) - 归一化策略
        # 将输出 (-1, 1) 映射到非负空间 (例如 0~1)
        # 常用方法：(x+1)/2 或者 softplus/exp
        raw_power = (power_params + 1.0) / 2.0 + 1e-6
        
        # 仅对激活的波束分配功率
        masked_power = raw_power * bh_mask
        
        # 归一化：每个卫星的总功率 <= P_total
        sum_power_per_sat = np.sum(masked_power, axis=1, keepdims=True) + 1e-12
        # 计算缩放因子：如果总和 > P_total，则缩放；否则可以用满或者就用 raw
        # 通常为了性能最大化，我们会用满 P_total (Water-filling 思想)
        # 这里简单地按比例分配全功率 P_total
        scaling_factor = self.total_power_watt / sum_power_per_sat
        final_power_alloc = masked_power * scaling_factor
        
        # 4. 获取环境状态（流量、信道）
        global_traffic = self.traffic_gen.get_traffic_at_time(self.current_step_idx)
        local_arrival = global_traffic[self.mapping_matrix]  # Mbps
        H_local = np.take_along_axis(self._get_current_channel_matrix(), self.mapping_matrix, axis=1)
        
        # 5. 物理层计算 (SINR -> Capacity)
        interference = self._calculate_interference_vectorized(bh_mask, final_power_alloc)
        signal_p = final_power_alloc * (H_local ** 2)
        sinr = calculate_sinr(signal_p, self.noise_power, interference)
        capacity = calculate_throughput(sinr, self.bw) * bh_mask # Mbps
        
        # 6. 队列与时延更新
        # Capacity (Mbps) * Time (s) = Mbits
        service_bits = capacity * self.time_slot_duration 
        arrival_bits = local_arrival * self.time_slot_duration
        
        # 队列更新 (Mbits)
        old_queues = self.local_queues.copy()
        self.local_queues = np.maximum(0, old_queues - service_bits) + arrival_bits
        
        # 简单的时延估算：Little's Law 近似或累积滞留
        # 这里用一种启发式：如果有队列但没服务，时延+1；如果有服务，时延减小
        # 或者直接用队列长度作为时延的代理（Queue Stability <-> Low Delay）
        # 原代码有 calculate_delay_fairness_diff，这里维持原有的 queue 逻辑即可
        
        # 7. 计算指标 (Metrics)
        # (a) 负载均衡指标 (Load Balance): Max Queue - Min Queue
        # 考虑所有卫星的所有小区队列
        all_queues = self.local_queues.flatten()
        
        # (a) 负载均衡 (Load Balance Diff): 原始单位 Mbits
        load_diff = float(np.max(all_queues) - np.min(all_queues))
        
        # (b) 系统时延 (Avg Queue Length): 原始单位 Mbits (近似)
        avg_queue = float(np.mean(all_queues))
        
        # (c) 总吞吐量 (Total Throughput): 原始单位 Mbps
        total_throughput = float(np.sum(capacity))
        
        # 8. [核心修改] 动态量纲归一化
        # 将每个指标除以其历史平均值，使其成为无量纲的比率
        norm_load = self.load_normalizer.normalize(load_diff)
        norm_delay = self.delay_normalizer.normalize(avg_queue)
        norm_th = self.th_normalizer.normalize(total_throughput)
        
        # 9. 奖励计算
        # 现在权重 w 具有更纯粹的意义：相对重要性
        # 例如 w_load=0.4 意味着我们对负载不均衡的惩罚权重是基准值的 0.4 倍
        w_load = 0.4
        w_delay = 0.4
        w_th = 0.2
        
        # 奖励公式：- (负载比率 + 时延比率) + 吞吐量比率
        # 这样 reward 整体数值也会保持在 -1.0 ~ 1.0 附近的合理范围（取决于权重和波动）
        reward = - (w_load * norm_load + w_delay * norm_delay) + w_th * norm_th
        
        # 10. 状态更新与终止检查
        self.current_step_idx += 1
        terminated = self.current_step_idx >= self.max_steps
        
        info = {
            "raw/load_diff": load_diff,
            "raw/avg_queue": avg_queue,
            "raw/throughput": total_throughput,
            "norm/load_diff": norm_load,
            "norm/avg_queue": norm_delay,
            "norm/throughput": norm_th,
            "reward": reward,
            "mean_power_util": np.mean(np.sum(final_power_alloc, axis=1)) / self.total_power_watt
        }
        
        return float(reward), bool(terminated), info

    def get_env_info(self) -> Dict[str, Any]:
        """
        更新环境信息以适配连续动作空间
        """
        return {
            "n_agents": self.n_agents,
            "n_actions": self.act_dim,           # 2 * C
            "obs_shape": self.obs_dim,           # C * 3
            "state_shape": self.n_agents * self.obs_dim,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_spaces, # Box spaces
            "actions_dtype": np.float32,         # 连续值
            "normalise_actions": True            # 通知算法这是归一化的连续动作 (-1, 1)
        }

    def get_obs(self) -> List[np.ndarray]:
        """返回所有智能体的观测列表"""
        full_obs = self._get_observation()  # (N, C, 3)
        return [full_obs[i].astype(np.float32).reshape(-1) for i in range(self.num_satellites)]

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        """返回单个智能体的观测"""
        full_obs = self._get_observation()  # (N, C, 3)
        return full_obs[agent_id].astype(np.float32).reshape(-1)

    def get_obs_size(self) -> int:
        """返回单个智能体的观测维度"""
        return self.obs_dim

    def get_state(self) -> np.ndarray:
        """返回全局状态（展平为一维）"""
        full_obs = self._get_observation()  # (N, C, 3)
        return full_obs.astype(np.float32).reshape(-1)

    def get_state_size(self) -> int:
        """返回全局状态维度"""
        return self.n_agents * self.obs_dim

    def get_avail_actions(self) -> List[np.ndarray]:
        """
        返回所有智能体的可用动作
        对于连续动作空间，所有动作都可用，返回全1向量
        """
        # 对于连续动作空间，返回全1表示所有动作都可用
        return [np.ones(self.act_dim, dtype=np.int32) for _ in range(self.num_satellites)]

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        """返回单个智能体的可用动作"""
        # 对于连续动作空间，所有动作都可用
        return np.ones(self.act_dim, dtype=np.int32)

    def get_total_actions(self) -> int:
        """返回单个智能体的动作维度（连续动作空间）"""
        return self.act_dim

    def get_num_agents(self) -> int:
        """返回智能体数量"""
        return self.num_satellites

    def get_stats(self) -> Dict[str, Any]:
        """返回环境统计信息"""
        return {
            "current_step": self.current_step_idx,
            "episode_limit": self.episode_limit,
            "total_queues": np.sum(self.local_queues),
            "mean_queue": np.mean(self.local_queues),
            "max_queue": np.max(self.local_queues),
        }

    def render(self, save_path: str = None):
        """渲染环境（可选，需要matplotlib）"""
        if not PLOT_AVAILABLE:
            return
        try:
            plot_hexagonal_topology(
                self.sat_positions_init,
                self.cell_positions,
                self.mapping_table,
                self.sat_service_radius,
                self.cell_radius,
                save_path=save_path
            )
        except Exception as e:
            print(f"Warning: Failed to render environment: {e}")

    def close(self):
        """关闭环境，清理资源"""
        if PLOT_AVAILABLE:
            try:
                plt.close('all')
            except:
                pass

    def seed(self, seed: int):
        """设置随机种子"""
        self.seed_val = seed
        self.rng = np.random.RandomState(seed)
        try:
            torch.manual_seed(seed)
        except NameError:
            # torch 未导入时跳过
            pass