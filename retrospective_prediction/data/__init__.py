"""数据层模块 - 冻土-基础设施沉降预测实验"""

from .synthetic_generator import SyntheticDataGenerator
from .preprocess import (
    load_config,
    load_npz,
    normalize_features,
    build_adjacency_matrix,
    split_time_series,
    validate_data,
    save_processed,
)
from .download_data import download_dataset, handle_download_failure
