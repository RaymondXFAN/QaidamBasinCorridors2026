"""数据预处理模块 - 冻土-基础设施沉降预测实验

功能：
- load_config(): 读取YAML配置（类型安全转换）
- load_npz(path): 加载已预处理的npz数据
- normalize_features(X, method): 特征标准化
- build_adjacency_matrix(coords, threshold_km): 从坐标构建空间邻接矩阵
- split_time_series(data, train_ratio): 时间分割
- validate_data(X, y, adj, geo): 维度检查和校验
- save_processed(data_dict, path): 保存npz
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_config(yaml_path: str) -> dict:
    """读取YAML配置文件，并对所有数值键做类型安全转换

    关键修复：YAML可能把 1e-5 解析为字符串，AutoDL部署时出现过此bug。

    Args:
        yaml_path: YAML文件路径

    Returns:
        cfg: 类型安全的配置字典
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"YAML配置文件为空或格式错误: {yaml_path}")

    def sanitize(d):
        """递归对字典做类型安全转换"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                # 尝试转为int，再尝试转为float
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass  # 保持字符串
            elif isinstance(v, dict):
                v = sanitize(v)
            elif isinstance(v, list):
                # 列表中的元素也做类型转换
                v = [_sanitize_value(item) for item in v]
            result[k] = v
        return result

    def _sanitize_value(v):
        """对单个值做类型安全转换"""
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return v
        elif isinstance(v, dict):
            return sanitize(v)
        return v

    cfg = sanitize(cfg)
    print(f"[Preprocess] 加载配置: {yaml_path}")
    print(f"  dataset_name: {cfg.get('dataset_name', 'N/A')}")
    print(f"  N_nodes: {cfg.get('N_nodes', 'N/A')}, T_steps: {cfg.get('T_steps', 'N/A')}")

    return cfg


def load_npz(path: str) -> dict:
    """加载已预处理的npz数据文件

    Args:
        path: npz文件路径

    Returns:
        data_dict: 包含 X, y, adj_matrix, geo_features 等的字典
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    raw = np.load(str(path), allow_pickle=True)

    data_dict = {}
    for key in raw.files:
        data_dict[key] = raw[key]

    # 恢复metadata（如果存在）
    if "metadata_keys" in data_dict and "metadata_vals" in data_dict:
        keys = data_dict["metadata_keys"]
        vals = data_dict["metadata_vals"]
        metadata = {}
        for k, v in zip(keys, vals):
            # 尝试类型转换
            try:
                metadata[str(k)] = int(v)
            except (ValueError, TypeError):
                try:
                    metadata[str(k)] = float(v)
                except (ValueError, TypeError):
                    metadata[str(k)] = str(v)
        data_dict["metadata"] = metadata

    # 恢复feature_names（如果存在）
    if "feature_names" in data_dict:
        data_dict["feature_names"] = list(data_dict["feature_names"])

    # 打印维度信息
    print(f"[Preprocess] 加载npz数据: {path}")
    if "X" in data_dict:
        print(f"  X.shape = {data_dict['X'].shape}")
    if "y" in data_dict:
        print(f"  y.shape = {data_dict['y'].shape}")
    if "adj_matrix" in data_dict:
        print(f"  adj_matrix.shape = {data_dict['adj_matrix'].shape}")
    if "geo_features" in data_dict:
        print(f"  geo_features.shape = {data_dict['geo_features'].shape}")

    return data_dict


def normalize_features(X: np.ndarray, method: str = "zscore",
                       feature_range: Optional[Tuple] = None) -> Tuple[np.ndarray, dict]:
    """特征标准化

    Args:
        X: (N, T, F) 或 (N, F) 特征矩阵
        method: 标准化方法
            - 'zscore': 均值方差标准化 (x - μ) / σ
            - 'minmax': 最小最大标准化 (x - min) / (max - min)
            - 'robust': Robust标准化 (x - median) / IQR
        feature_range: minmax的目标范围，默认(0, 1)

    Returns:
        X_norm: 标准化后的特征
        norm_params: 标准化参数（用于逆变换）
    """
    if feature_range is None:
        feature_range = (0, 1)

    original_shape = X.shape
    # 展平为2D处理
    if X.ndim == 3:
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)  # (N*T, F)
    elif X.ndim == 2:
        X_flat = X.copy()
        F = X.shape[1]
    else:
        raise ValueError(f"X维度不支持: {X.shape}, 需要2D或3D")

    norm_params = {}

    if method == "zscore":
        mean = X_flat.mean(axis=0)
        std = X_flat.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)  # 防止除零
        X_flat = (X_flat - mean) / std
        norm_params = {"method": "zscore", "mean": mean, "std": std}

    elif method == "minmax":
        min_val = X_flat.min(axis=0)
        max_val = X_flat.max(axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val < 1e-8, 1.0, range_val)
        X_flat = (X_flat - min_val) / range_val
        # 映射到目标范围
        if feature_range != (0, 1):
            X_flat = X_flat * (feature_range[1] - feature_range[0]) + feature_range[0]
        norm_params = {
            "method": "minmax", "min": min_val, "max": max_val,
            "feature_range": feature_range,
        }

    elif method == "robust":
        median = np.median(X_flat, axis=0)
        q25 = np.percentile(X_flat, 25, axis=0)
        q75 = np.percentile(X_flat, 75, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < 1e-8, 1.0, iqr)
        X_flat = (X_flat - median) / iqr
        norm_params = {"method": "robust", "median": median, "iqr": iqr}

    else:
        raise ValueError(f"未知标准化方法: {method}")

    # 恢复原始形状
    if X.ndim == 3:
        X_norm = X_flat.reshape(original_shape)
    else:
        X_norm = X_flat

    print(f"[Preprocess] 特征标准化: method={method}, shape {original_shape} -> {X_norm.shape}")

    return X_norm, norm_params


def build_adjacency_matrix(coords: np.ndarray, threshold_km: float = 50.0) -> np.ndarray:
    """从坐标构建空间邻接矩阵

    Args:
        coords: (N, 2) 经纬度坐标 [lon, lat]
        threshold_km: 连线距离阈值(km)

    Returns:
        adj_matrix: (N, N) 邻接矩阵（对称，含自环）
    """
    N = coords.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            dlat = coords[i, 1] - coords[j, 1]
            dlon = coords[i, 0] - coords[j, 0]
            # 经纬度→km近似转换
            dist_km = np.sqrt(
                (dlat * 111.0) ** 2 +
                (dlon * 111.0 * np.cos(np.radians(np.mean([coords[i, 1], coords[j, 1]])))) ** 2
            )
            if dist_km <= threshold_km:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # 自环
    np.fill_diagonal(adj, 1.0)

    num_edges = int((adj.sum() - N) / 2)
    avg_degree = 2 * num_edges / N if N > 0 else 0
    print(f"[Preprocess] 邻接矩阵: {N}节点, {num_edges}条边, "
          f"平均度={avg_degree:.1f}, 阈值={threshold_km}km")

    return adj


def split_time_series(data: dict, train_ratio: float = 0.8) -> dict:
    """时间序列分割：train(前80%) / test(后20%)

    滑动窗口展开：
    - 输入窗口 T_in 个月 → 预测 T_out 个月沉降
    - 样本数 = T - T_in - T_out + 1 (每节点)

    Args:
        data: 原始数据字典（含 X, y, adj_matrix, geo_features）
        train_ratio: 训练集比例

    Returns:
        split_data: 包含 X_train, y_train, X_test, y_test, adj_matrix, geo_features
    """
    X = data["X"]        # (N, T, F)
    y = data["y"]        # (N, T) 沉降全时序
    adj = data["adj_matrix"]   # (N, N)
    geo = data["geo_features"] # (N, G)

    N, T, F = X.shape
    T_in = data.get("T_in", 24) if isinstance(data.get("T_in"), int) else 24
    T_out = data.get("T_out", 1) if isinstance(data.get("T_out"), int) else 1

    # 检查是否可以构造足够窗口
    num_windows = T - T_in - T_out + 1
    if num_windows <= 0:
        raise ValueError(f"时间步T={T}不足以构造窗口 T_in={T_in}, T_out={T_out}")

    # 滑动窗口展开
    X_windows = np.zeros((N, num_windows, T_in, F))
    y_windows = np.zeros((N, num_windows, T_out))

    for n in range(N):
        for w in range(num_windows):
            start = w
            end = w + T_in
            X_windows[n, w] = X[n, start:end]
            # 预测窗口end之后的T_out步沉降
            y_windows[n, w] = y[n, end:end + T_out]

    # 按时间分割（前80%时间窗口为训练，后20%为测试）
    split_w = int(num_windows * train_ratio)

    # ===== Graph Format (W, N, T_in, F) — GNN模型使用 =====
    # 每个时间窗口包含所有N个真实节点，adj_matrix是N×N
    X_graph = X_windows.transpose(1, 0, 2, 3)  # (num_windows, N, T_in, F)
    y_graph = y_windows.transpose(1, 0, 2)       # (num_windows, N, T_out)

    X_train_graph = X_graph[:split_w]   # (W_train, N, T_in, F)
    y_train_graph = y_graph[:split_w]   # (W_train, N, T_out)
    X_test_graph  = X_graph[split_w:]   # (W_test, N, T_in, F)
    y_test_graph  = y_graph[split_w:]   # (W_test, N, T_out)

    # ===== Flat Format (N*W, T_in, F) — ML模型使用 =====
    # 展平节点维度：(N, num_windows, ...) -> (N*num_windows, ...)
    X_all = X_windows.reshape(N * num_windows, T_in, F)
    y_all = y_windows.reshape(N * num_windows, T_out)

    X_train = X_all[:N * split_w]
    y_train = y_all[:N * split_w]
    X_test = X_all[N * split_w:]
    y_test = y_all[N * split_w:]

    split_data = {
        # Flat format (for ML baselines: RF, XGBoost)
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        # Graph format (for GNN models: Geo-FiLM ST-GAT, ST-GAT, ST-GCN, DCRNN)
        "X_train_graph": X_train_graph,
        "X_test_graph": X_test_graph,
        "y_train_graph": y_train_graph,
        "y_test_graph": y_test_graph,
        # Shared
        "adj_matrix": adj,
        "geo_features": geo,
        "feature_names": data.get("feature_names", []),
        "node_ids": data.get("node_ids", np.arange(N)),
        "metadata": data.get("metadata", {}),
        "num_windows": num_windows,
        "N_nodes": N,
        "split_window_idx": split_w,
    }

    print(f"[Preprocess] 时间分割: train_ratio={train_ratio}")
    print(f"  [Flat]  X_train.shape = {X_train.shape}  (N*W_train, T_in, F)")
    print(f"  [Flat]  y_train.shape = {y_train.shape}  (N*W_train, T_out)")
    print(f"  [Flat]  X_test.shape = {X_test.shape}   (N*W_test, T_in, F)")
    print(f"  [Flat]  y_test.shape = {y_test.shape}   (N*W_test, T_out)")
    print(f"  [Graph] X_train_graph.shape = {X_train_graph.shape}  (W_train, N, T_in, F)")
    print(f"  [Graph] y_train_graph.shape = {y_train_graph.shape}  (W_train, N, T_out)")
    print(f"  [Graph] X_test_graph.shape = {X_test_graph.shape}   (W_test, N, T_in, F)")
    print(f"  [Graph] y_test_graph.shape = {y_test_graph.shape}   (W_test, N, T_out)")
    print(f"  adj_matrix.shape = {adj.shape}")
    print(f"  geo_features.shape = {geo.shape}")
    print(f"  总窗口数: {num_windows}, 训练窗口: {split_w}, 测试窗口: {num_windows - split_w}")

    return split_data


def validate_data(X: np.ndarray, y: np.ndarray, adj: np.ndarray,
                  geo: np.ndarray, F_expected: int = 12,
                  G_expected: int = 3) -> bool:
    """数据维度检查和校验

    Args:
        X: 特征矩阵 (N, T, F) 或 (N_samples, T_in, F)
        y: 标签 (N, T) 或 (N_samples, T_out)
        adj: 邻接矩阵 (N_total, N_total)
        geo: 地理先验 (N_total, G)
        F_expected: 期望特征维度数
        G_expected: 期望地理先验维度数

    Returns:
        is_valid: 是否通过校验
    """
    print("[Preprocess] 数据校验:")
    is_valid = True
    errors = []

    # X维度检查
    print(f"  X.shape = {X.shape}")
    if X.ndim != 3:
        errors.append(f"X应为3维 (N, T, F), 实际 ndim={X.ndim}")
        is_valid = False
    else:
        F_actual = X.shape[2]
        if F_actual != F_expected:
            errors.append(f"X特征维度 F={F_actual}, 期望 F={F_expected}")
            is_valid = False

    # y维度检查
    print(f"  y.shape = {y.shape}")
    if y.ndim not in [1, 2]:
        errors.append(f"y应为1维或2维, 实际 ndim={y.ndim}")
        is_valid = False

    # adj维度检查
    print(f"  adj_matrix.shape = {adj.shape}")
    if adj.ndim != 2:
        errors.append(f"adj应为2维 (N, N), 实际 ndim={adj.ndim}")
        is_valid = False
    elif adj.shape[0] != adj.shape[1]:
        errors.append(f"adj应为方阵, 实际 shape={adj.shape}")
        is_valid = False

    # geo维度检查
    print(f"  geo_features.shape = {geo.shape}")
    if geo.ndim != 2:
        errors.append(f"geo应为2维 (N, G), 实际 ndim={geo.ndim}")
        is_valid = False
    elif geo.shape[1] != G_expected:
        errors.append(f"geo维度 G={geo.shape[1]}, 期望 G={G_expected}")
        is_valid = False

    # N一致性检查
    N_nodes = adj.shape[0]
    if X.ndim == 3 and X.shape[0] != N_nodes:
        # 滑动窗口展开后X的N可能 != adj的N，这是正常的
        print(f"  注意: X.shape[0]={X.shape[0]} != adj.shape[0]={N_nodes} "
              f"(可能是滑动窗口展开后的样本数)")
    if geo.shape[0] != N_nodes:
        errors.append(f"geo节点数 {geo.shape[0]} != adj节点数 {N_nodes}")
        is_valid = False

    if errors:
        print(f"  [ERROR] 数据校验失败:")
        for err in errors:
            print(f"    - {err}")
    else:
        print(f"  [OK] 数据校验通过 ✓")

    return is_valid


def save_processed(data_dict: dict, path: str) -> str:
    """保存处理后的数据为npz格式

    Args:
        data_dict: 数据字典
        path: 保存路径

    Returns:
        save_path: 实际保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 准备保存字典（metadata需要特殊处理）
    save_dict = {}
    for key, value in data_dict.items():
        if key == "metadata" and isinstance(value, dict):
            # metadata转为字符串数组保存
            save_dict["metadata_keys"] = np.array(list(value.keys()))
            save_dict["metadata_vals"] = np.array([str(v) for v in value.values()])
        elif key == "feature_names" and isinstance(value, list):
            save_dict["feature_names"] = np.array(value)
        elif isinstance(value, np.ndarray):
            save_dict[key] = value
        elif isinstance(value, (int, float, str)):
            save_dict[key] = np.array([value])
        # 忽略不支持的类型

    np.savez(str(path), **save_dict)
    file_size = os.path.getsize(str(path)) / 1024
    print(f"[Preprocess] 数据已保存至: {path} ({file_size:.1f} KB)")

    return str(path)
