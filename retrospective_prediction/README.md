# 🧊 冻土-基础设施沉降预测实验项目

> 保姆级操作手册 —— 小虎哥哥专属版 🐯

---

## 📖 1. 项目概述

### 这个实验做什么？

本实验旨在验证论文核心创新 **Geo-FiLM ST-GAT** 在真实数据上的预测能力。当前论文仅有 AirSim 仿真验证结果（91.3% mAP, 2.4m RMSE），审稿人很可能会质疑"仿真结果能否代表真实性能？"

本实验解决这个质疑：
- 在公开真实冻土数据集上做 **回顾性预测**（retrospective prediction）：用历史数据预测未来沉降
- 对比 **8个模型**：Geo-FiLM ST-GAT（我们的创新） vs 消融对照 vs GNN对比 vs PINN vs 传统ML vs 纯物理
- 输出**论文可直接引用的实验结果表格**

### 为什么做？

| 论文地位 | 模型 | 作用 |
|---------|------|------|
| 🌟 **核心创新** | Geo-FiLM ST-GAT | 证明FiLM conditioning有效 |
| 🔬 消融对照 | Standard ST-GAT | 证明去掉FiLM会变差 |
| 📊 GNN对比 | ST-GCN, DCRNN | 证明GAT优于其他GNN |
| ⚗️ PINN对比 | PINN-Permafrost | 证明GNN+FiLM > 纯PINN |
| 🔧 纯物理 | Stefan-Only | 证明数据驱动优于纯物理 |
| 📈 传统ML | Random Forest, XGBoost | 证明GNN优于传统ML |

---

## 🖥️ 2. AutoDL环境准备

### 2.1 租用GPU实例

1. 打开 [AutoDL官网](https://www.autodl.com/)
2. 注册/登录账号
3. 点击「创建实例」
4. 选择配置：
   - **镜像**：PyTorch 2.0 + Python 3.10（推荐）
   - **GPU**：RTX 3090 / A100 / V100 都可以（内存 ≥ 16GB）
   - **硬盘**：数据盘 ≥ 50GB（模型不大，够用）
5. 等待实例启动（约1-2分钟）

### 2.2 SSH连接到实例

AutoDL创建实例后，会给你连接信息：

```bash
# SSH连接命令（在终端中粘贴）
ssh -p <端口> root@<地址>

# 例如：
ssh -p 12345 root@region-1.autodl.pro

# 密码：AutoDL给你的密码（在实例详情页查看）
```

> 💡 **提示**：Windows用户可以用 PowerShell、Git Bash 或 MobaXterm 连接SSH

### 2.3 上传代码到AutoDL

**方法一：SCP命令上传**（推荐，最简单）

```bash
# 在你的本地电脑终端运行（不是AutoDL终端！）
scp -P <端口> -r ./retrospective_prediction root@<地址>:/root/

# 例如：
scp -P 12345 -r ./retrospective_prediction root@region-1.autodl.pro:/root/
```

**方法二：AutoDL文件管理器上传**

1. 在AutoDL控制台点击「文件存储」→「文件管理器」
2. 进入 `/root/` 目录
3. 点击「上传」按钮，把整个项目文件夹拖进去

**方法三：Git克隆**（如果代码在GitHub上）

```bash
# 在AutoDL终端运行
cd /root
git clone <你的仓库URL>
```

### 2.4 安装依赖

连接到AutoDL后，运行部署脚本：

```bash
cd /root/retrospective_prediction
bash setup_autodl.sh
```

这个脚本会自动：
- ✅ 检查GPU和CUDA环境
- ✅ 安装缺失的Python依赖（torch通常已预装）
- ✅ 检查数据维度
- ✅ 运行仿真兜底实验

> ⚠️ **注意**：AutoDL通常已预装PyTorch，脚本不会重新安装，避免版本冲突

如果脚本出问题，手动安装依赖：

```bash
pip install numpy pandas scikit-learn matplotlib pyyaml xgboost
# torch 通常已预装，不需要再装
# torch_geometric 不需要安装（所有GNN模型手写实现）
```

---

## 📊 3. 数据获取

### 3.1 公开数据下载

| 数据集 | 来源 | 链接 | 说明 |
|--------|------|------|------|
| 青藏走廊InSAR形变 | TPDC | 10.11888/cryos.tpdc.300400 | ⭐ 主要数据 |
| GTN-P CALM ALT | PANGAEA | 10.1594/PANGAEA.972777 | ⭐ 主要数据 |
| 青藏高原综合监测 | TPDC | 789e838e... | ⭐ 主要数据 |

> 💡 TPDC（国家青藏高原科学数据中心）注册即可免费下载

### 3.2 数据格式要求

所有数据预处理后统一为 `.npz` 格式：

```python
{
    'X_train':     (N_train, T, F)      # 节点特征时序，F=12个特征
    'y_train':     (N_train, T_out)     # 沉降标签
    'X_test':      (N_test, T, F)       # 测试特征
    'y_test':      (N_test, T_out)      # 测试标签
    'adj_matrix':  (N_total, N_total)   # 邻接矩阵
    'geo_features':(N_total, G)         # 地理先验，G=3 (ALT, k_s, I_t)
}
```

### 3.3 仿真兜底（始终可用！）

如果真实数据获取失败，项目内置仿真数据生成器：

```bash
# 使用仿真兜底配置运行
python3 run_main.py --config configs/simulation_fallback.yaml
```

仿真数据特点：
- 模拟InSAR形变 + ALT + 地基沉降
- 参数可调（节点数、时间步、噪声等）
- **始终可用**，确保实验框架至少能跑通

---

## 🚀 4. 运行实验

### 4.1 Step-by-step操作

**第一步：进入项目目录**

```bash
cd /root/retrospective_prediction
```

**第二步：确保依赖已安装**

```bash
# 快速检查
python3 -c "import torch, numpy, sklearn, xgboost; print('All OK!')"
# 如果报错，运行：bash setup_autodl.sh
```

**第三步：运行仿真兜底实验**（先确保框架OK）

```bash
python3 run_main.py --config configs/simulation_fallback.yaml
```

> 💡 **为什么先跑仿真？** 确保代码框架没问题，真数据出问题时不至于卡住。

**第四步：如果获取到真实数据，运行真实数据实验**

```bash
python3 run_main.py --config configs/real_data.yaml
```

**第五步：查看结果**

```bash
cat results/results_table.md     # Markdown结果表格
cat results/all_results.json     # JSON详细结果
```

### 4.2 配置文件说明

配置文件位于 `configs/` 目录：

```yaml
# configs/simulation_fallback.yaml 示例
hidden_dim: 64
num_heads: 4
num_spatial_layers: 2
dropout: 0.1
epochs: 200
lr: 0.001
lambda_phys: 0.3
batch_size: 32
output_dim: 1

# 仿真数据参数
num_nodes: 50
time_steps: 24
feature_dim: 12
geo_dim: 3
noise_level: 0.05
```

> ⚠️ **关键规则**：`input_dim` 不在配置文件中硬编码！它会在运行时从 `X_train.shape[2]` 动态获取。

---

## 📁 5. 查看结果

### 5.1 结果目录结构

```
results/
├── all_results.json      # 所有模型的详细指标（RMSE, MAE, R²等）
├── results_table.md      # Markdown格式的对比表格（可直接放入论文）
└── (可能还有其他图表文件)
```

### 5.2 结果JSON格式

```json
{
  "Geo-FiLM ST-GAT": {
    "rmse": 0.0342,
    "mae": 0.0281,
    "r2": 0.9156,
    "phys_violation_rate": 5.2,
    "film_improvement": 18.3,
    "train_time_sec": 120.5
  },
  "Standard ST-GAT": {
    "rmse": 0.0419,
    ...
  }
}
```

### 5.3 结果Markdown表格

| Model | RMSE (m) | MAE (m) | R² | Phys.Viol. (%) | FiLM Imp. (%) |
|-------|----------|---------|-----|----------------|---------------|
| Geo-FiLM ST-GAT | 0.0342 | 0.0281 | 0.9156 | 5.2 | 18.3 |
| Standard ST-GAT | 0.0419 | ... | ... | ... | - |
| ... | ... | ... | ... | ... | ... |

---

## ✍️ 6. 结果解读（如何填入论文）

### 6.1 论文 Table X 格式

将 `results_table.md` 的内容直接复制到论文中，格式为：

> **Table X: Retrospective Prediction on Public Datasets**

| Model | Type | RMSE (m) | MAE (m) | R² | Phys.Viol. (%) | FiLM Imp. (%) |
|-------|------|----------|---------|-----|----------------|---------------|
| Geo-FiLM ST-GAT | PIML-GNN | ✓ | ✓ | ✓ | ✓ | ✓ |
| Standard ST-GAT | GNN | ... | ... | ... | ... | - |
| ST-GCN | GNN | ... | ... | ... | ... | - |
| DCRNN | GNN | ... | ... | ... | ... | - |
| PINN-Permafrost | PIML | ... | ... | ... | ... | - |
| Stefan-Only | Physics | ... | ... | ... | ... | - |
| Random Forest | ML | ... | ... | ... | ... | - |
| XGBoost | ML | ... | ... | ... | ... | - |

### 6.2 关键结论提取

- **FiLM有效性**：对比 Geo-FiLM ST-GAT vs Standard ST-GAT → FiLM Improvement 百分比
- **GNN有效性**：对比 GNN模型 vs 传统ML → GNN是否优于RF/XGBoost
- **物理约束**：所有模型的 Phys.Viol. (%) → Geo-FiLM是否物理一致性最好
- **PINN局限**：对比 PINN vs Geo-FiLM → 为什么GNN+FiLM优于纯PINN

### 6.3 论文Discussion要点

- 如果 FiLM Improvement > 10%：可以强调"FiLM conditioning显著提升预测精度"
- 如果 Phys.Viol. < 10%：可以强调"Geo-FiLM物理一致性优于其他数据驱动方法"
- 如果优于PINN：可以强调"将物理约束融入GNN架构优于纯PINN"

---

## ❓ 7. 常见问题FAQ

### Q1: GPU不可用 / CUDA报错

```bash
# 检查GPU状态
nvidia-smi

# 检查PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# 如果返回False：
# 1. 确认AutoDL实例有GPU（不是CPU实例）
# 2. 检查PyTorch和CUDA版本是否匹配
# 3. 代码会自动fallback到CPU（速度较慢但能跑）
```

> 💡 所有模型代码都有自动GPU检测：`device = 'cuda' if torch.cuda.is_available() else 'cpu'`

### Q2: 维度不匹配报错

```
RuntimeError: Expected input_dim=X but got Y
```

**解决方法**：
- 代码自动从 `X_train.shape[2]` 获取 `input_dim`，不需要手动设置
- 检查数据文件的 `X_train` 维度是否符合 `(N, T, F)` 格式
- F应该等于12（标准特征数），如果不同也没关系（动态获取）

```bash
# 检查数据维度
python3 -c "
import numpy as np
data = np.load('data/xxx.npz', allow_pickle=True)
for k in data.files:
    print(f'{k}: {data[k].shape}')
"
```

### Q3: 数据下载失败

**解决方法**：
- TPDC注册后即可下载，如果注册遇到问题，先跑仿真兜底
- 仿真数据始终可用，确保框架能跑通
- 论文中可以写"仿真验证+公开数据验证"双轨策略

### Q4: 训练时间太长

```bash
# 减少训练轮数
# 在配置文件中修改：
epochs: 100    # 从200降到100
hidden_dim: 32  # 从64降到32

# 或者减少batch_size（不影响精度，只影响速度）
```

### Q5: xgboost安装失败

```bash
# 如果pip安装xgboost失败，代码会自动fallback到sklearn的GradientBoostingRegressor
# 不影响实验结果，只是名字变成"GBRT"而不是"XGBoost"

pip install xgboost --no-cache-dir  # 尝试强制安装
```

### Q6: 内存溢出 (OOM)

```bash
# 减少模型大小
# 在配置文件中修改：
hidden_dim: 32   # 从64降到32
num_spatial_layers: 1  # 从2降到1

# 或者减少仿真数据节点数
num_nodes: 20    # 从50降到20
```

### Q7: YAML配置值类型错误

``TypeError: unsupported operand type(s) for *: 'str' and 'float'``

**这已经自动处理了！** 所有代码都有 `_sanitize_config()` 方法，将YAML字符串自动转为float/int。

### Q8: 如何只用仿真数据？

```bash
# 如果没有真实数据，只跑仿真：
python3 run_main.py --config configs/simulation_fallback.yaml
# 仿真数据会自动生成，不需要额外下载
```

---

## 📦 项目文件说明

```
retrospective_prediction/
├── README.md                  # 本文件（保姆级操作手册）
├── EXPERIMENT_DESIGN.md       # 实验设计文档
├── setup_autodl.sh            # AutoDL一键部署脚本
├── run_main.py                # 主运行脚本（入口）
│
├── configs/                   # 配置文件目录
│   ├── simulation_fallback.yaml   # 仿真兜底配置
│   └── real_data.yaml             # 真实数据配置（需要数据文件）
│
├── data/                      # 数据目录
│   └── (放置.npz数据文件)
│
├── models/                    # 模型代码目录
│   ├── __init__.py                # 模型导出
│   ├── geo_film_st_gat.py         # 🌟 Geo-FiLM ST-GAT（核心创新）
│   ├── st_gat.py                  # Standard ST-GAT（消融对照）
│   ├── stgcn.py                   # ST-GCN（GNN对比）
│   ├── dcrnn.py                   # DCRNN（GNN对比）
│   ├── pinn_permafrost.py         # PINN-Permafrost（PINN对比）
│   └── stefan_model.py            # Stefan-Only（纯物理对照）
│
├── baselines/                 # 传统ML baseline目录
│   ├── __init__.py
│   ├── rf_baseline.py             # Random Forest
│   └── gbrt_baseline.py           # XGBoost/GBRT
│
├── trainers/                  # 训练器目录
│   ├── __init__.py
│   ├── trainer.py                 # 通用训练/评估/实验运行
│
├── evaluation/                # 评估指标目录
│   ├── __init__.py
│   ├── metrics.py                 # RMSE, MAE, R², PhysViol, FiLM Imp
│   ├── physical_constraint.py     # Stefan物理约束检查
│
└── results/                   # 结果输出目录（运行后生成）
    ├── all_results.json
    └── results_table.md
```

---

## 🎯 快速开始（一分钟版）

如果你只想快速跑一遍，不管细节：

```bash
# 1. 连接AutoDL
ssh -p <端口> root@<地址>

# 2. 上传代码（在本地电脑运行）
scp -P <端口> -r ./retrospective_prediction root@<地址>:/root/

# 3. 在AutoDL终端运行
cd /root/retrospective_prediction
bash setup_autodl.sh

# 4. 查看结果
cat results/results_table.md
```

---

## 📞 需要帮助？

如果遇到问题：
1. 先检查FAQ部分
2. 查看代码中的打印信息（每个步骤都有维度和状态输出）
3. 所有模型都有自动fallback机制（GPU→CPU, xgboost→sklearn, 真数据→仿真）

祝实验顺利！❄️🏔️
