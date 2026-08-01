"""数据下载模块 - 冻土-基础设施沉降预测实验

尝试下载公开数据集（TPDC/PANGAEA），失败时gracefully降级到仿真数据。

支持的数据集ID：
- D1: 青藏走廊InSAR形变 (TPDC)
- D2: CALM ALT (PANGAEA)
- A1: 柴木铁路 (TPDC)
- A5: 东北冻土 (NCDC)
- A8: 野牛沟 (TPDC)

路径策略：
- AutoDL环境：数据存放到 /root/autodl-tmp/datasets/raw/（数据盘）
- 本地开发：存放到 项目根目录/datasets/raw/
- 通过 DATA_ROOT 环境变量或 config['data_root'] 指定
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# AutoDL数据盘默认路径
AUTODL_DATA_DISK = '/root/autodl-tmp'


def _resolve_data_root(config: dict) -> str:
    """Resolve data root from config, env, or auto-detect.
    
    Priority:
      1. config['data_root'] (from CLI --data-root or YAML)
      2. Environment variable DATA_ROOT
      3. Auto-detect /root/autodl-tmp (AutoDL data disk)
      4. Current working directory (fallback)
    """
    # 1. Config override
    if config.get('data_root'):
        return config['data_root']
    
    # 2. Environment variable
    env_root = os.environ.get('DATA_ROOT')
    if env_root:
        return env_root
    
    # 3. Auto-detect AutoDL data disk
    if os.path.exists(AUTODL_DATA_DISK) and os.path.isdir(AUTODL_DATA_DISK):
        return AUTODL_DATA_DISK
    
    # 4. Fallback: current working directory
    return os.getcwd()

# 数据集元信息
DATASET_REGISTRY = {
    "D1": {
        "name": "Qinghai-Tibet Corridor 2D Deformation (2017-2022)",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/10.11888/cryos.tpdc.300400",
        "doi": "10.11888/cryos.tpdc.300400",
        "format": "geotiff",
        "description": "青藏工程走廊二维形变数据集",
    },
    "D2": {
        "name": "GTN-P CALM Active Layer Thickness",
        "source": "PANGAEA",
        "url": "https://doi.pangaea.de/10.1594/PANGAEA.972777",
        "doi": "10.1594/PANGAEA.972777",
        "format": "csv",
        "description": "Circumpolar Active Layer Monitoring (CALM) ALT数据",
    },
    "A1": {
        "name": "Chaimu Railway Deformation + Damage",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/f7be4a1d-2506",
        "doi": "f7be4a1d-2506",
        "format": "mixed",
        "description": "祁连山柴木铁路形变+病害数据",
    },
    "A5": {
        "name": "Northeast China Permafrost Engineering Stability",
        "source": "NCDC",
        "url": "https://data.ncdc.ac.cn/zh-hans/data/70c33bc7",
        "doi": "70c33bc7",
        "format": "mixed",
        "description": "东北冻土工程基础稳定性数据",
    },
    "A8": {
        "name": "Yeniugou Permafrost Deformation + ALT",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/dd24266c",
        "doi": "dd24266c",
        "format": "mixed",
        "description": "野牛沟冻土形变+ALT数据",
    },
}


def download_dataset(config: dict) -> Optional[str]:
    """根据config中的data_source尝试下载公开数据集

    Args:
        config: YAML配置字典（包含data_root等信息）

    Returns:
        data_path: 下载成功返回数据路径，失败返回None
    """
    data_source = config.get("data_source", "simulation")
    dataset_name = config.get("dataset_name", "qaidam_inSAR")

    # 如果data_source是simulation，直接降级
    if data_source == "simulation":
        print(f"[Download] data_source='simulation', 跳过下载，使用仿真数据")
        return None

    # 查找数据集ID
    dataset_id = None
    for key, info in DATASET_REGISTRY.items():
        if info["url"] == data_source or info["doi"] == data_source:
            dataset_id = key
            break

    if dataset_id is None:
        # 尝试直接用URL
        print(f"[Download] 未知数据源: {data_source}")
        print(f"[Download] 尝试直接下载...")
        dataset_id = "CUSTOM"

    print(f"[Download] 尝试下载数据集: {dataset_id}")
    print(f"  数据集: {DATASET_REGISTRY.get(dataset_id, {}).get('name', data_source)}")

    # Download to DATA_ROOT/datasets/raw/ (data disk on AutoDL)
    data_root = _resolve_data_root(config)
    download_dir = Path(data_root) / "datasets" / "raw"
    download_dir.mkdir(parents=True, exist_ok=True)
    target_path = download_dir / f"{dataset_name}_raw"

    try:
        _attempt_download(data_source, str(target_path))
        print(f"[Download] ✓ 下载成功: {target_path}")
        return str(target_path)
    except Exception as e:
        print(f"[Download] ✗ 下载失败: {e}")
        print(f"[Download] 降级到仿真数据...")
        return None


def _attempt_download(url: str, target_path: str) -> None:
    """尝试从URL下载数据

    实际下载需要网络访问和认证，这里仅做尝试。
    在有网络的环境中会真正下载。

    Args:
        url: 数据下载URL
        target_path: 本地保存路径
    """
    import urllib.request
    import socket

    # 设置超时
    socket.setdefaulttimeout(30)

    try:
        # 尝试简单HTTP下载
        urllib.request.urlretrieve(url, target_path)
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout) as e:
        raise RuntimeError(f"网络下载失败: {e}")
    except Exception as e:
        raise RuntimeError(f"下载异常: {e}")


def handle_download_failure(config: dict) -> str:
    """下载失败时自动调用synthetic_generator生成兜底数据

    Args:
        config: YAML配置字典

    Returns:
        save_path: 仿真数据保存路径
    """
    from .synthetic_generator import generate_and_save_synthetic

    print(f"[Download] ===== 降级到仿真数据（兜底方案） =====")
    print(f"[Download] 原因: 公开数据下载失败或不可用")
    print(f"[Download] 说明: 仿真数据基于Stefan方程+热力学模型，物理逼真")

    # 修改config中的data_source标记
    config["data_source"] = "simulation"
    config["metadata_note"] = "仿真兜底数据，仅供框架验证"

    save_path = generate_and_save_synthetic(config)

    print(f"[Download] 仿真数据已生成: {save_path}")
    print(f"[Download] 仿真数据仅用于验证实验框架可运行性，不代表真实结果")

    return save_path


def get_dataset_info(dataset_id: str) -> dict:
    """获取数据集信息

    Args:
        dataset_id: 数据集ID (D1, D2, A1, A5, A8)

    Returns:
        info: 数据集元信息字典
    """
    if dataset_id in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_id]
    else:
        return {"name": "Unknown", "source": "Unknown", "url": "", "doi": ""}


def list_available_datasets() -> Dict[str, dict]:
    """列出所有可用数据集

    Returns:
        registry: 数据集注册表
    """
    print("[Download] 可用数据集列表:")
    for id_, info in DATASET_REGISTRY.items():
        print(f"  {id_}: {info['name']} ({info['source']})")
    return DATASET_REGISTRY
