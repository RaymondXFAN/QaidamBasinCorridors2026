# Retrospective Prediction Experiment Design
## 方案A：用公开数据验证 Geo-FiLM ST-GAT 预测能力

---

## 1. 实验目标

**核心问题**：论文核心创新 Geo-FiLM ST-GAT 目前仅有 AirSim 仿真验证（91.3% mAP, 2.4m RMSE），
审稿人会质疑"仿真结果能否代表真实性能？"

**本实验解决**：
- 在至少1个公开真实数据集上做 retrospective prediction（用历史数据预测未来沉降）
- 对比 Geo-FiLM ST-GAT vs Standard ST-GAT vs 多个baseline（PINN/GNN/传统ML/纯物理）
- 输出论文可直接引用的实验结果表格

---

## 2. 统一数据格式规范

所有数据集预处理后统一输出为 `.npz` 格式：

```python
# 数据结构
{
    'X_train':     np.ndarray (N_train, T_in, F)     # 节点特征时序
    'X_test':      np.ndarray (N_test, T_in, F)      # 测试特征时序
    'y_train':     np.ndarray (N_train, T_out)        # 训练沉降标签（未来T_out步）
    'y_test':      np.ndarray (N_test, T_out)         # 测试沉降标签
    'adj_matrix':  np.ndarray (N_total, N_total)      # 图邻接矩阵（空间邻近）
    'geo_features':np.ndarray (N_total, G)            # FiLM conditioning 先验
                                               # G = [ALT, k_s, I_t, latitude, elevation, ...]
    'feature_names':list[str]                         # 特征名列表
    'node_ids':    np.ndarray (N_total)               # 节点标识
    'metadata':    dict                               # 数据集信息（名称、时间范围、来源等）
}
```

**特征维度 F（统一）**：
| idx | 特征名 | 说明 |
|-----|---------|------|
| 0   | cumulative_deformation | InSAR累积形变(mm) |
| 1   | deformation_velocity   | 形变速率(mm/year) |
| 2   | coherence              | InSAR相干性 |
| 3   | ground_temp            | 地温(°C) |
| 4   | thawing_index          | 融化指数I_t(°C·day) |
| 5   | freezing_index         | 冻结指数(°C·day) |
| 6   | moisture               | 土壤含水量(%) |
| 7   | elevation              | 海拔(m) |
| 8   | slope_angle            | 坡度(°) |
| 9   | permafrost_type        | 冻土类型编码 |
| 10  | infrastructure_type    | 基础设施类型编码 |
| 11  | seasonal_amplitude     | 形变季节振幅(mm) |

**FiLM conditioning 地理先验 G**：
| idx | 特征名 | 说明 |
|-----|---------|------|
| 0   | ALT     | Stefan方程推导的活动层厚度(m) |
| 1   | k_s     | 土壤-结构系数(无量纲) |
| 2   | I_t     | 融化指数(°C·day) |

---

## 3. 模型接口规范

所有模型继承 `BaseModel`，统一接口：

```python
class BaseModel:
    def __init__(self, config: dict):           # 从YAML配置初始化
    def fit(self, X, y, adj, geo, **kwargs):    # 训练
    def predict(self, X, adj, geo):             # 预测
    def evaluate(self, X, y, adj, geo):         # 评估 → metrics dict
    def get_name(self) -> str:                  # 返回模型名称
```

**关键约定**：
- `input_dim` **不从config硬编码读取**，在 `fit()` 中从 `X.shape[2]` 动态设置
- `geo_dim` 从 `geo.shape[1]` 动态设置
- 所有数值参数从YAML读取后做 `float()/int()` 类型安全转换
- GPU自动检测：`device = 'cuda' if torch.cuda.is_available() else 'cpu'`

---

## 4. Baseline 对比矩阵

| # | Model | Type | Key Feature | 论文地位 |
|---|-------|------|------------|---------|
| 1 | **Geo-FiLM ST-GAT** | PIML-GNN | FiLM conditioning on Stefan ALT | **Ours（核心创新）** |
| 2 | Standard ST-GAT | GNN | 无地质 conditioning | 消融对照（证明FiLM有用） |
| 3 | ST-GCN | GNN | 图卷积+时间卷积 | 其他GNN对比 |
| 4 | DCRNN | GNN | 扩散卷积RNN | 其他GNN对比 |
| 5 | PINN-Permafrost | PIML | Stefan方程作物理约束 | PINN对比（证明GNN+FiLM > PINN） |
| 6 | Stefan-Only | Physics | 纯Stefan方程预测沉降 | 纯物理模型对比 |
| 7 | Random Forest | ML | sklearn ensemble | 传统ML对比 |
| 8 | XGBoost | ML | gradient boosting | 传统ML对比 |

---

## 5. 评估指标

| Metric | 说明 | 对应论文 |
|--------|------|---------|
| RMSE (m) | 沉降预测均方根误差 | Table 7 12h settlement RMSE |
| MAE (m) | 平均绝对误差 | 补充 |
| R² | 决定系数 | 补充 |
| Physical Violation Rate (%) | 预测违反 Stefan 约束的比例 | Discussion 5.3 |
| FiLM Improvement (%) | Geo-FiLM vs Standard ST-GAT 的 RMSE 降幅 | 对应论文消融 |
| Cross-region Transfer R² | 跨区域验证的 R² | Transferability |

---

## 6. 数据集备选方案（多路并行获取）

### Primary：青藏走廊 InSAR 形变数据
| ID | 数据集 | 来源 | DOI/链接 | 优先级 |
|----|---------|------|----------|--------|
| **D1** | 青藏工程走廊二维形变数据集(2017-2022) | TPDC | 10.11888/cryos.tpdc.300400 | ⭐ Primary |
| **D2** | GTN-P CALM ALT | PANGAEA | 10.1594/PANGAEA.972777 | ⭐ Primary |
| **D3** | 青藏高原综合监测数据 | TPDC | 789e838e-16ac-4539-bb7e | ⭐ Primary |

### Alternative：多区域备选
| ID | 数据集 | 来源 | 链接 | 优先级 |
|----|---------|------|------|--------|
| **A1** | 祁连山柴木铁路形变+病害 | TPDC(牛富俊) | f7be4a1d-2506... | ⭐⭐ Alt-1 |
| **A2** | 青藏走廊冻土-工程水热变形综合观测 | ESSD | ESSD论文配套 | ⭐⭐ Alt-2 |
| **A5** | 东北冻土工程基础稳定性 | NCDC(金会军) | 70c33bc7... | ⭐ Alt-3 |
| **A8** | 野牛沟冻土形变+ALT | TPDC | dd24266c... | ⭐ Alt-4 |

### Fallback：仿真数据（始终可用）
| ID | 数据集 | 说明 | 优先级 |
|----|---------|------|--------|
| **SIM** | 仿真生成器 | 模拟InSAR形变+ALT+地基沉降，参数可调 | Fallback兜底 |

**数据获取策略**：
- 优先尝试D1/D2（论文已引用，审稿人认可）
- 同步注册TPDC获取A1/A2（开放获取，注册即下载）
- SIM仿真数据**始终可用**，确保实验框架至少能跑通
- 最终论文：优先展示真实数据结果，仿真作为补充

---

## 7. AutoDL 部署要点

| 要点 | 规范 |
|------|------|
| **input_dim** | 从数据动态设置 `input_dim = X_train.shape[2]`，覆盖config硬编码 |
| **维度检查** | setup_autodl.sh 打印实际特征数与config是否一致 |
| **YAML类型安全** | load_config后对所有数值键做 float()/int() 转换 |
| **GPU检测** | `device = 'cuda' if available else 'cpu'`；模型和数据 .to(device) |
| **依赖安装** | torch, torch_geometric, sklearn, rasterio, pandas, numpy, matplotlib |
| **操作指引** | README.md 提供"保姆级"详细步骤（SSH连接→环境配置→数据下载→运行→查看结果） |

---

## 8. 实验输出（论文可用）

### Table X: Retrospective Prediction on Public Datasets
| Model | RMSE (m) | MAE (m) | R² | Phys.Viol. (%) | FiLM Imp. (%) |
|-------|----------|---------|----|----------------|----------------|
| Geo-FiLM ST-GAT | — | — | — | — | — |
| Standard ST-GAT | — | — | — | — | reference |
| ST-GCN | — | — | — | — | — |
| DCRNN | — | — | — | — | — |
| PINN-Permafrost | — | — | — | — | — |
| Stefan-Only | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

### Figure X: Ablation of FiLM Conditioning
- Geo-FiLM vs Standard vs AdaIN vs Cross-Attention conditioning

### Figure X: Prediction Horizon Analysis
- 6h / 12h / 24h / 48h / 1week RMSE curves

---

*Generated: 2026-08-01 | Version: v1 | Author: 柳如烟*
