"""仿真数据生成器 - 冻土-基础设施沉降预测实验的兜底方案

基于 Stefan 方程 + 热力学模型生成逼真的祁连山/柴达木冻土 InSAR 形变+沉降数据。

数据格式：
    X:  (N, T, F) — N个监测节点, T个时间步, F=12个特征
    y:  (N, T_out) — 沉降预测标签
    adj_matrix: (N, N) — 空间邻近图邻接矩阵
    geo_features: (N, G=3) — FiLM conditioning先验 [ALT, k_s, I_t]

物理模型驱动：
    ALT = sqrt(2 * k_thaw * I_t * 86400 / L)   (Stefan方程)
    settlement = k_s * ALT * seasonal_factor + noise
    deformation velocity = dALT/dt 的积分
"""

import os
import numpy as np
from pathlib import Path


class SyntheticDataGenerator:
    """仿真数据生成器：基于 Stefan 方程 + 热力学模型"""

    # Stefan方程常数
    K_THAW_DEFAULT = 1.2      # W/m·K，融化期热导率
    L_DEFAULT = 3.34e7        # J/m³，潜热（水→冰）
    SEC_PER_DAY = 86400       # 秒/天

    # 气候参数（祁连山/柴达木典型值）
    MEAN_ANNUAL_TEMP = -3.0   # °C，年均地温
    SUMMER_AMPLITUDE = 8.0    # °C，夏季偏离均值
    WINTER_AMPLITUDE = -12.0  # °C，冬季偏离均值
    SUMMER_PEAK_MONTH = 6     # 6月融化峰值
    WINTER_TROUGH_MONTH = 1  # 1月冻结谷值

    # 地理参数范围
    ELEVATION_RANGE = (2800, 4800)  # m，海拔范围
    SLOPE_RANGE = (0, 30)           # °，坡度范围
    MOISTURE_RANGE = (15, 45)       # %，含水量范围

    # 特征名列表（12维，严格对应EXPERIMENT_DESIGN.md）
    FEATURE_NAMES = [
        "cumulative_deformation",   # 0: mm
        "deformation_velocity",     # 1: mm/year
        "coherence",                # 2: 0-1
        "ground_temp",              # 3: °C
        "thawing_index",            # 4: °C·day
        "freezing_index",           # 5: °C·day
        "moisture",                 # 6: %
        "elevation",                # 7: m
        "permafrost_type",          # 8: int (用slope_angle存，见下)
        "infrastructure_type",      # 9: int
        "seasonal_amplitude",       # 10: mm  (对应F idx 11)
    ]

    # 严格12维特征名
    FEATURE_NAMES_12 = [
        "cumulative_deformation",   # idx 0
        "deformation_velocity",     # idx 1
        "coherence",                # idx 2
        "ground_temp",              # idx 3
        "thawing_index",            # idx 4 (I_t)
        "freezing_index",           # idx 5
        "moisture",                 # idx 6
        "elevation",                # idx 7
        "slope_angle",              # idx 8
        "permafrost_type",          # idx 9
        "infrastructure_type",      # idx 10
        "seasonal_amplitude",       # idx 11
    ]

    # 冻土类型编码
    PERMAFROST_TYPES = {
        "continuous": 0,
        "discontinuous": 1,
        "sporadic": 2,
        "isolated": 3,
    }

    # 基础设施类型编码
    INFRASTRUCTURE_TYPES = {
        "railway": 0,
        "highway": 1,
        "pipeline": 2,
        "building": 3,
        "bridge": 4,
    }

    def __init__(self, config: dict):
        """初始化生成器

        Args:
            config: YAML配置字典（已做类型安全转换）
        """
        self.config = config
        self.N = int(config.get("N_nodes", 47))
        self.T = int(config.get("T_steps", 120))
        self.F = int(config.get("F_features", 12))
        self.G = int(config.get("G_geo_features", 3))
        self.T_in = int(config.get("T_in", 24))
        self.T_out = int(config.get("T_out", 1))
        self.seed = int(config.get("seed", 42))
        self.noise_level = float(config.get("noise_level", 0.05))
        self.k_thaw = float(config.get("k_thaw", self.K_THAW_DEFAULT))
        self.L = float(config.get("L", self.L_DEFAULT))
        self.adj_threshold_km = float(config.get("adj_threshold_km", 50))
        self.region = config.get("region", "Qaidam Basin Corridor")
        self.dataset_name = config.get("dataset_name", "qaidam_inSAR")
        self.output_dir = config.get("output_dir", "results")

        # 设置随机种子
        np.random.seed(self.seed)

    def _generate_node_coordinates(self) -> np.ndarray:
        """生成N个监测节点的地理坐标（沿走廊/铁路分布）

        Returns:
            coords: (N, 2) — [longitude, latitude] 经纬度
        """
        # 祁连山走廊：大致从东到西分布
        # 经度范围 ~96°E - 103°E，纬度范围 ~37°N - 39°N
        base_lon = np.random.uniform(96.0, 103.0, self.N)
        base_lat = np.random.uniform(37.0, 39.0, self.N)

        # 模拟沿走廊分布：加入线性趋势
        order = np.argsort(base_lon)
        base_lon[order] = np.linspace(96.0, 103.0, self.N) + np.random.normal(0, 0.3, self.N)
        base_lat[order] = np.linspace(37.5, 38.5, self.N) + np.random.normal(0, 0.2, self.N)

        coords = np.stack([base_lon, base_lat], axis=1)
        return coords

    def _generate_monthly_thawing_index(self, t: int) -> float:
        """计算第t月的融化指数 I_t (°C·day)

        季节模型：7月约240°C·day/月，1月约0
        使用正弦模型拟合月度变化

        Args:
            t: 时间步索引（0-based，月度）

        Returns:
            I_t: 该月的融化指数 (°C·day)
        """
        month = t % 12  # 0=1月, 1=2月, ..., 5=6月, 6=7月
        # 融化指数正弦模型：6-8月高峰，1-2月为零
        # 峰值在6月（month=5）和7月（month=6）
        phase = (month - 5) / 12.0 * 2 * np.pi  # 6月为峰值
        I_t = max(0, 240.0 * np.cos(phase) * (1 if month >= 4 and month <= 9 else 0))
        # 更精确的模型
        if month in [4, 5, 6, 7, 8, 9]:  # 5-10月有融化
            I_t = 240.0 * np.sin((month - 3) / 6.0 * np.pi)  # 从4月到10月的半正弦
        else:
            I_t = 0.0
        return I_t

    def _generate_monthly_freezing_index(self, t: int) -> float:
        """计算第t月的冻结指数 (°C·day)

        Args:
            t: 时间步索引

        Returns:
            freezing_index: °C·day
        """
        month = t % 12
        if month in [10, 11, 0, 1, 2, 3]:  # 11-4月有冻结
            freezing_index = 200.0 * np.sin((month - 9) / 6.0 * np.pi)
        else:
            freezing_index = 0.0
        return max(0, freezing_index)

    def _compute_alt_stefan(self, I_t_annual: float) -> float:
        """用 Stefan 方程计算活动层厚度 ALT

        ALT = sqrt(2 * k_thaw * I_t * 86400 / L)

        Args:
            I_t_annual: 年度累积融化指数 (°C·day)

        Returns:
            ALT: 活动层厚度 (m)
        """
        if I_t_annual <= 0:
            return 0.01  # 极薄活动层（完全冻结）
        alt = np.sqrt(2.0 * self.k_thaw * I_t_annual * self.SEC_PER_DAY / self.L)
        return alt

    def _compute_seasonal_factor(self, t: int) -> float:
        """计算季节因子（冻融周期）

        6月峰值（融化），1月谷值（冻结）
        正弦模型：factor = 1 + 0.5 * sin(2π*(t-5)/12)

        Args:
            t: 时间步索引

        Returns:
            seasonal_factor: 季节波动因子
        """
        month = t % 12
        # 6月(month=5)为峰值，1月(month=0)为谷值
        phase = 2 * np.pi * (month - 5) / 12.0
        factor = 1.0 + 0.5 * np.sin(phase)
        return factor

    def _compute_ground_temp(self, t: int, mean_temp: float) -> float:
        """计算地温

        季节变化：夏+8°C，冬-12°C（年均-3°C基准）

        Args:
            t: 时间步
            mean_temp: 年均地温基准

        Returns:
            ground_temp: °C
        """
        month = t % 12
        phase = 2 * np.pi * (month - 5) / 12.0
        # 夏季偏离+8°C，冬季偏离-12°C，使用不对称振幅
        if month >= 4 and month <= 9:  # 夏半年
            amplitude = self.SUMMER_AMPLITUDE
        else:  # 冬半年
            amplitude = self.WINTER_AMPLITUDE

        temp = mean_temp + amplitude * np.sin(phase)
        return temp

    def generate(self) -> dict:
        """生成完整仿真数据集

        Returns:
            data_dict: 包含 X, y, adj_matrix, geo_features, coords, metadata 的字典
        """
        print(f"[SyntheticGenerator] 生成仿真数据集: {self.dataset_name}")
        print(f"  区域: {self.region}")
        print(f"  N={self.N}, T={self.T}, F={self.F}, G={self.G}")
        print(f"  T_in={self.T_in}, T_out={self.T_out}, seed={self.seed}")

        np.random.seed(self.seed)

        # ===== Step 1: 生成监测节点坐标 =====
        coords = self._generate_node_coordinates()

        # 每个节点的静态属性
        elevations = np.random.uniform(*self.ELEVATION_RANGE, self.N)
        slopes = np.random.uniform(*self.SLOPE_RANGE, self.N)
        moisture_base = np.random.uniform(*self.MOISTURE_RANGE, self.N)
        k_s_values = np.random.uniform(0.01, 0.05, self.N)  # 土壤-结构系数
        permafrost_types = np.random.choice(list(self.PERMAFROST_TYPES.values()), self.N)
        infrastructure_types = np.random.choice(list(self.INFRASTRUCTURE_TYPES.values()), self.N)

        # 每个节点有自己的年均地温（海拔越高越冷）
        mean_temps = self.MEAN_ANNUAL_TEMP - (elevations - 3500) / 1000.0 * 6.0  # 约-6°C/km

        # ===== Step 2: 计算每个节点的ALT时序 =====
        # 首先计算年度累积融化指数
        annual_I_t = np.zeros(self.N)
        for t_idx in range(12):  # 一个完整年度
            I_t_month = self._generate_monthly_thawing_index(t_idx)
            annual_I_t += I_t_month
        # 不同节点有不同I_t（海拔越高，I_t越低）
        node_I_t_annual = annual_I_t * (1.0 + np.random.normal(0, 0.2, self.N))
        node_I_t_annual = np.clip(node_I_t_annual, 200, 1800)

        # Stefan方程计算ALT
        alt_values = np.array([self._compute_alt_stefan(it) for it in node_I_t_annual])
        alt_values = np.clip(alt_values, 0.5, 5.0)  # 物理合理范围

        # ===== Step 3: 计算沉降时序 =====
        # settlement = k_s * ALT * seasonal_factor + noise
        # 累积沉降随时间增加
        settlements = np.zeros((self.N, self.T))
        for n in range(self.N):
            cumulative = 0.0
            for t in range(self.T):
                seasonal = self._compute_seasonal_factor(t)
                # 年度ALT变化（气候变暖趋势：每年ALT增加约2%）
                alt_year = alt_values[n] * (1.0 + 0.02 * (t // 12))
                settlement_increment = k_s_values[n] * alt_year * seasonal * 0.1  # mm/month
                settlement_increment += np.random.normal(0, self.noise_level * 10)  # 噪声
                cumulative += settlement_increment
                settlements[n, t] = cumulative

        # ===== Step 4: 生成InSAR累积形变 =====
        # cumulative_deformation = settlement的累积 + InSAR噪声 + 解缠误差
        cumulative_deformation = np.zeros((self.N, self.T))
        deformation_velocity = np.zeros((self.N, self.T))
        coherence = np.zeros((self.N, self.T))
        seasonal_amplitude = np.zeros((self.N, self.T))

        for n in range(self.N):
            for t in range(self.T):
                # 累积形变 = 累积沉降 + InSAR系统噪声
                cumulative_deformation[n, t] = settlements[n, t] + \
                    np.random.normal(0, self.noise_level * 5)
                # 形变速率 = 该月沉降增量 * 12 (转mm/year)
                if t > 0:
                    velocity = (settlements[n, t] - settlements[n, t - 1]) * 12.0
                else:
                    velocity = k_s_values[n] * alt_values[n] * 12.0 * 0.1
                deformation_velocity[n, t] = velocity + \
                    np.random.normal(0, self.noise_level * 2)

                # 相干性：冬季高（冻结稳定），夏季低（融化扰动）
                month = t % 12
                base_coherence = 0.85
                if month >= 4 and month <= 9:
                    base_coherence = 0.65  # 融化期相干性降低
                coherence[n, t] = base_coherence + np.random.normal(0, 0.05)
                coherence[n, t] = np.clip(coherence[n, t], 0.3, 0.95)

                # 季节振幅
                seasonal_amplitude[n, t] = k_s_values[n] * alt_values[n] * 0.5

        # ===== Step 5: 生成其他特征时序 =====
        ground_temp_ts = np.zeros((self.N, self.T))
        thawing_index_ts = np.zeros((self.N, self.T))
        freezing_index_ts = np.zeros((self.N, self.T))
        moisture_ts = np.zeros((self.N, self.T))

        for n in range(self.N):
            for t in range(self.T):
                ground_temp_ts[n, t] = self._compute_ground_temp(t, mean_temps[n])
                thawing_index_ts[n, t] = self._generate_monthly_thawing_index(t) * \
                    (1.0 + np.random.normal(0, 0.1))  # 节点间微差异
                freezing_index_ts[n, t] = self._generate_monthly_freezing_index(t) * \
                    (1.0 + np.random.normal(0, 0.1))
                # 含水量：融化期高，冻结期低
                month = t % 12
                moisture_ts[n, t] = moisture_base[n] * \
                    (1.0 + 0.3 * np.sin(2 * np.pi * (month - 5) / 12.0))

        # ===== Step 6: 组装特征矩阵 X =====
        # X: (N, T, F=12)
        X = np.zeros((self.N, self.T, self.F))
        X[:, :, 0] = cumulative_deformation
        X[:, :, 1] = deformation_velocity
        X[:, :, 2] = coherence
        X[:, :, 3] = ground_temp_ts
        X[:, :, 4] = thawing_index_ts
        X[:, :, 5] = freezing_index_ts
        X[:, :, 6] = moisture_ts
        X[:, :, 7] = elevations[:, np.newaxis] * np.ones((1, self.T))  # 海拔静态
        X[:, :, 8] = slopes[:, np.newaxis] * np.ones((1, self.T))      # 坡度静态
        X[:, :, 9] = permafrost_types[:, np.newaxis] * np.ones((1, self.T))
        X[:, :, 10] = infrastructure_types[:, np.newaxis] * np.ones((1, self.T))
        X[:, :, 11] = seasonal_amplitude

        # ===== Step 7: 组装标签 y =====
        # y: (N, T_out) — 每个时间步的沉降
        # 对于T_out=1，取每个时间步的沉降值
        # 滑动窗口构造样本：(N, T) -> (N*T_in有效样本, T_out)
        # 简化处理：y = settlements 的第 T_out 步

        # 构造滑动窗口样本
        num_windows = self.T - self.T_in - self.T_out + 1
        if num_windows <= 0:
            # 如果T太小，调整T_in
            self.T_in = max(1, self.T - self.T_out - 1)
            num_windows = self.T - self.T_in - self.T_out + 1

        # 每个节点产生 num_windows 个样本
        # 但为了保持 (N, T_out) 格式，我们取最后一个窗口的预测
        # 实际使用中会在训练阶段再做滑动窗口展开
        # 这里 y = 所有时间步的沉降值，方便后续滑动窗口切分
        y = settlements  # (N, T) — 全时序沉降

        # ===== Step 8: 构建地理先验 geo_features =====
        # geo_features: (N, G=3) — [ALT, k_s, I_t]
        geo_features = np.zeros((self.N, self.G))
        geo_features[:, 0] = alt_values       # ALT (m)
        geo_features[:, 1] = k_s_values       # k_s (无量纲)
        geo_features[:, 2] = node_I_t_annual  # I_t (°C·day)

        # ===== Step 9: 构建空间邻接矩阵 =====
        adj_matrix = self._build_adjacency_matrix(coords)

        # ===== Step 10: 时间分割 =====
        train_ratio = float(self.config.get("train_ratio", 0.8))
        split_idx = int(self.T * train_ratio)

        # 构造输出字典
        data_dict = {
            "X": X,
            "y": y,
            "adj_matrix": adj_matrix,
            "geo_features": geo_features,
            "coords": coords,
            "elevations": elevations,
            "settlements": settlements,
            "cumulative_deformation": cumulative_deformation,
            "alt_values": alt_values,
            "split_idx": split_idx,
            "feature_names": self.FEATURE_NAMES_12,
            "node_ids": np.arange(self.N),
            "metadata": {
                "dataset_name": self.dataset_name,
                "region": self.region,
                "N": self.N,
                "T": self.T,
                "F": self.F,
                "G": self.G,
                "T_in": self.T_in,
                "T_out": self.T_out,
                "seed": self.seed,
                "noise_level": self.noise_level,
                "k_thaw": self.k_thaw,
                "source": "simulation",
                "description": f"仿真数据: {self.region}, Stefan方程驱动",
            },
        }

        # 打印维度信息
        print(f"[SyntheticGenerator] 数据维度:")
        print(f"  X.shape = {X.shape}  (N, T, F)")
        print(f"  y.shape = {y.shape}  (N, T)")
        print(f"  adj_matrix.shape = {adj_matrix.shape}  (N, N)")
        print(f"  geo_features.shape = {geo_features.shape}  (N, G)")
        print(f"  coords.shape = {coords.shape}  (N, 2)")
        print(f"  split_idx = {split_idx}  (train: 0~{split_idx}, test: {split_idx}~{self.T})")

        # ALT统计
        print(f"[SyntheticGenerator] ALT统计:")
        print(f"  min ALT = {alt_values.min():.3f} m")
        print(f"  max ALT = {alt_values.max():.3f} m")
        print(f"  mean ALT = {alt_values.mean():.3f} m")

        return data_dict

    def _build_adjacency_matrix(self, coords: np.ndarray) -> np.ndarray:
        """从坐标构建空间邻接矩阵

        距离阈值：threshold_km 内的节点连线

        Args:
            coords: (N, 2) 经纬度坐标

        Returns:
            adj_matrix: (N, N) 邻接矩阵（0/1，对称）
        """
        N = coords.shape[0]
        adj = np.zeros((N, N), dtype=np.float32)

        # 经纬度距离近似（1° ≈ 111km）
        for i in range(N):
            for j in range(i + 1, N):
                dlat = coords[i, 1] - coords[j, 1]
                dlon = coords[i, 0] - coords[j, 0]
                # 简化距离计算（°→km）
                dist_km = np.sqrt((dlat * 111) ** 2 + (dlon * 111 * np.cos(np.radians(coords[i, 1]))) ** 2)
                if dist_km <= self.adj_threshold_km:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # 自环
        for i in range(N):
            adj[i, i] = 1.0

        # 打印邻接矩阵统计
        num_edges = (adj.sum() - N) / 2  # 去掉自环
        print(f"[SyntheticGenerator] 邻接矩阵: {num_edges:.0f} 条边 "
              f"(阈值={self.adj_threshold_km}km, 平均度={2*num_edges/N:.1f})")

        return adj

    def generate_and_save(self) -> str:
        """生成数据并保存为npz格式

        Returns:
            save_path: 保存文件路径
        """
        data_dict = self.generate()

        # 构造保存字典（npz兼容格式）
        save_dict = {
            "X": data_dict["X"],
            "y": data_dict["y"],
            "adj_matrix": data_dict["adj_matrix"],
            "geo_features": data_dict["geo_features"],
            "coords": data_dict["coords"],
            "feature_names": np.array(data_dict["feature_names"]),
            "node_ids": data_dict["node_ids"],
            "split_idx": data_dict["split_idx"],
            "alt_values": data_dict["alt_values"],
            "settlements": data_dict["settlements"],
        }

        # metadata 不能直接存npz，转为字符串数组
        meta = data_dict["metadata"]
        meta_keys = np.array(list(meta.keys()))
        meta_vals = np.array([str(v) for v in meta.values()])
        save_dict["metadata_keys"] = meta_keys
        save_dict["metadata_vals"] = meta_vals

        # 保存路径
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{self.dataset_name}.npz"

        np.savez(save_path, **save_dict)
        print(f"[SyntheticGenerator] 数据已保存至: {save_path}")
        print(f"  文件大小: {os.path.getsize(save_path) / 1024:.1f} KB")

        return str(save_path)


def generate_synthetic_data(config: dict) -> dict:
    """便捷函数：根据config生成仿真数据

    Args:
        config: YAML配置字典

    Returns:
        data_dict: 生成数据字典
    """
    generator = SyntheticDataGenerator(config)
    return generator.generate()


def generate_and_save_synthetic(config: dict) -> str:
    """便捷函数：根据config生成并保存仿真数据

    Args:
        config: YAML配置字典

    Returns:
        save_path: 保存文件路径
    """
    generator = SyntheticDataGenerator(config)
    return generator.generate_and_save()
