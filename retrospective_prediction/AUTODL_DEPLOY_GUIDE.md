# 🧊 冻土-基础设施沉降预测实验 — AutoDL 保姆级部署手册

> 📋 **目标读者**：第一次用AutoDL云端GPU的小虎哥哥 🐯
> 🎯 **最终目标**：在AutoDL RTX4090上跑完8模型×200 epochs实验，拿到txt结果发给我
> 🔧 **环境配置**：RTX4090 | CUDA 13.0(driver)/12.4(runtime) | PyTorch 2.4+cu24 | Python 3.12
> 💾 **数据盘**：/root/autodl-tmp (/dev/md0挂载点，大容量) ← ★ 所有数据和结果必须放这里！

---

## 📦 一、AutoDL租用与连接（5分钟）

### 1.1 租用GPU实例

1. 打开 [AutoDL官网](https://www.autodl.com/)
2. 注册/登录 → 点击「创建实例」
3. ⚡ 选择配置（匹配你的环境）：
   - **镜像**：PyTorch 2.4 + Python 3.12 + CUDA 12.4（搜索社区镜像 `pytorch2.4`）
   - **GPU**：RTX 4090（约3-5元/小时）
   - **数据盘**：勾选！非常重要！数据盘 = /root/autodl-tmp（大容量，/dev/md0）
4. 点击「创建」→ 等1-2分钟实例启动

> ⚠️ **环境兼容性说明**：
> - 你的驱动 CUDA 13.0 >= PyTorch CUDA Runtime 12.4 ✓（向下兼容，没问题）
> - RTX4090有24GB VRAM，batch_size=32完全够用
> - PyTorch 2.4+cu24 = CUDA 12.4 runtime，代码中所有tensor操作兼容
> - Python 3.12兼容所有依赖包（numpy, pandas, sklearn, scipy等）

### 1.2 连接实例

**推荐方式：JupyterLab（最简单）**
1. 在实例列表点击「JupyterLab」
2. 自动打开浏览器终端
3. 🎉 直接可以输入命令了！

**其他方式：SSH连接**
1. 在实例列表找到SSH连接信息：`ssh -p <端口> root@<地址>`
2. 用Mac终端/PuTTY/SecureCRT连接

### 1.3 关机与省钱

- 实验跑完 → AutoDL控制台点「关机」→ 只收存储费（几毛/天）
- 「无卡模式」开机 = 不花GPU钱，可以查看/下载结果
- 训练时开「有卡模式」→ 按小时收费
- ⭐ 数据在数据盘 `/root/autodl-tmp/`，关机后不会丢失！

---

## 📤 二、上传项目代码（3分钟）

### 2.1 JupyterLab上传（最简单）

1. 打开JupyterLab
2. 左侧文件浏览器 → 点「上传」按钮（⬆️图标）
3. 选择我发给你的 `retrospective_prediction.zip` 上传
4. 在终端中解压：
```bash
cd /root
unzip retrospective_prediction.zip
```

### 2.2 ★ 系统盘 vs 数据盘（重要！）

AutoDL有两个盘：
- **系统盘** `/root` — 小容量(~50GB)，放代码，关机后数据保留
- **数据盘** `/root/autodl-tmp` — 大容量(~数百GB)，放数据+输出，关机后数据保留

```
系统盘 /root/
├── retrospective_prediction/   ← 代码放这里（系统盘）
│   ├── configs/                ← YAML配置文件
│   ├── data/                   ← 数据处理代码
│   ├── models/                 ← 模型代码
│   ├── baselines/              ← 基线代码
│   ├── trainers/               ← 训练器代码
│   ├── evaluation/             ← 评估代码
│   ├── run_main.py             ← 主入口
│   ├── download_datasets.py    ← 数据下载脚本
│   └── setup_autodl.sh         ← 环境配置脚本
│
数据盘 /root/autodl-tmp/        ← 数据+输出放这里！
├── datasets/
│   ├── raw/                    ← 下载的原始数据
│   ├── processed/              ← 预处理后的数据(.npz)
│   └── download_status.txt     ← 下载状态报告
└── outputs/
    ├── experiment_results.txt  ← ★ 结果文件（发给我！）
    ├── experiment_results.csv  ← CSV格式结果
    ├── diagnostics.txt         ← 详细诊断信息
    └── run_log.txt             ← 运行日志
```

> ⚠️ **代码自动处理路径**：DATA_ROOT自动检测逻辑——发现 `/root/autodl-tmp` 存在就自动用数据盘，不需要手动设置路径！

### 2.3 验证上传成功

```bash
cd /root/retrospective_prediction
ls -la
```

应该看到 `run_main.py`, `download_datasets.py`, `setup_autodl.sh` 等文件。

---

## 🔬 三、数据下载与准备（先做！）

> ⭐ **数据与代码分离**：数据在数据盘 `/root/autodl-tmp/datasets/`，代码在系统盘，修改代码不会影响数据

### 3.1 先下载数据（独立脚本）

```bash
cd /root/retrospective_prediction
python3 download_datasets.py
```

脚本会自动：
- ✅ 发现 `/root/autodl-tmp` → DATA_ROOT自动设为数据盘
- ✅ 尝试下载4个候选数据集（A1柴木/A4青藏/A5东北/A8野牛沟）
- ✅ 每个数据集显示成功/失败状态
- ✅ 生成 `/root/autodl-tmp/datasets/download_status.txt` 报告
- ✅ 下载成功 → 预处理到 `/root/autodl-tmp/datasets/processed/`
- ✅ 全失败 → 自动生成仿真兜底数据

### 3.2 查看下载状态

```bash
cat /root/autodl-tmp/datasets/download_status.txt
```

这会告诉你哪些数据集可用。**如果全部失败也没关系**，仿真兜底数据会自动生成。

### 3.3 如果只想用已有数据（跳过下载）

```bash
python3 download_datasets.py --preprocess-only
```

### 3.4 查看已有数据（在数据盘上）

```bash
ls -la /root/autodl-tmp/datasets/processed/
ls -la /root/autodl-tmp/datasets/raw/
```

---

## 🚀 四、运行实验（核心步骤）

### 4.1 环境配置（一键脚本）

```bash
cd /root/retrospective_prediction
bash setup_autodl.sh
```

脚本会自动：
- ✅ 设置 DATA_ROOT=/root/autodl-tmp
- ✅ 检查GPU/CUDA/Python/PyTorch版本
- ✅ 验证CUDA Driver 13.0 >= Runtime 12.4 兼容性
- ✅ 检查数据盘容量
- ✅ 安装缺失依赖（numpy, pandas, sklearn, xgboost等）
- ✅ 运行数据下载脚本
- ✅ 运行实验（输出到数据盘）

### 4.2 手动运行实验 — 基础命令

```bash
cd /root/retrospective_prediction
python3 run_main.py --config configs/autodl_gpu.yaml
```

> ★ 运行时自动检测 DATA_ROOT——发现 `/root/autodl-tmp` 存在就自动用数据盘！无需手动加参数。

> ★ 如果想显式指定数据盘路径（更保险）：`python3 run_main.py --config configs/autodl_gpu.yaml --data-root /root/autodl-tmp`

### 4.3 调试参数（⭐ 方便追踪排错）

| 参数 | 作用 | 示例 |
|------|------|------|
| `--verbose` / `-v` | 默认，打印关键进度 | `python3 run_main.py -v` |
| `--debug` / `-d` | 详细维度/张量信息（排错用） | `python3 run_main.py -d` |
| `--quiet` / `-q` | 只打印最终结果 | `python3 run_main.py -q` |
| `--log-file` | 保存所有输出到文件（★默认已开启！） | `python3 run_main.py --log-file` |
| `--no-log` | 不保存日志，只打印到屏幕 | `python3 run_main.py --no-log` |
| `--epochs N` | 覆盖训练轮数 | `python3 run_main.py --epochs 5`（快速测试） |
| `--data-root PATH` | 指定数据盘路径 | `python3 run_main.py --data-root /root/autodl-tmp` |
| `--device cuda` | 强制使用GPU | `python3 run_main.py --device cuda` |
| `--skip-download` | 用本地缓存数据 | `python3 run_main.py --skip-download` |

**推荐操作流程**：

1️⃣ **先快速测试（5轮，确保不出错）**：
```bash
cd /root/retrospective_prediction
python3 run_main.py --config configs/autodl_gpu.yaml --epochs 5 --debug
```
> ★ 日志文件 `/root/autodl-tmp/outputs/run_log.txt` 默认自动创建！不再需要手动加 `--log-file`

2️⃣ **把调试日志发给我**：
```bash
cat /root/autodl-tmp/outputs/run_log.txt
# 复制内容发给我，我帮你看看有没有问题题
```

3️⃣ **确认OK后跑正式实验（200轮）**：
```bash
python3 run_main.py --config configs/autodl_gpu.yaml
```

### 4.4 切换不同数据集配置

```bash
# 柴木铁路（A1）— 30节点
python3 run_main.py --config configs/chaimu_railway.yaml

# 青藏走廊（A4）— 47节点 — 默认主配置
python3 run_main.py --config configs/qaidam_inSAR.yaml

# 东北冻土（A5）— 25节点
python3 run_main.py --config configs/northeast_frost.yaml

# 野牛沟（A8）— 15节点
python3 run_main.py --config configs/yeniugou_alt.yaml

# 仿真兜底（始终可用）— 40节点
python3 run_main.py --config configs/simulation_fallback.yaml

# AutoDL GPU正式实验（200轮）— 40节点
python3 run_main.py --config configs/autodl_gpu.yaml
```

### 4.5 预计耗时

| 设备 | 200 epochs × 8模型 |
|------|-------------------|
| RTX 4090 (24GB) | ~5分钟 |
| RTX 3090 (24GB) | ~8分钟 |
| A100 (80GB) | ~3分钟 |
| CPU-only | ~40分钟 |

---

## 📊 五、查看与发送结果（⭐ 格式已优化）

### 5.1 结果文件位置（在数据盘上！）

所有结果在 `/root/autodl-tmp/outputs/` 目录：

```bash
ls -la /root/autodl-tmp/outputs/
```

你应该看到：
```
/root/autodl-tmp/outputs/
├── experiment_results.txt  ← ⭐ 纯文本结果表（发给我这个！）
├── experiment_results.csv  ← CSV格式（也可以发）
├── diagnostics.txt         ← ⭐ 详细诊断信息（排错用）
└── run_log.txt             ← ★ 运行日志（默认自动创建！）
```

### 5.2 查看结果（纯文本，方便复制）

```bash
# 查看主结果表格
cat /root/autodl-tmp/outputs/experiment_results.txt

# 查看详细诊断信息
cat /root/autodl-tmp/outputs/diagnostics.txt

# 查看CSV结果（适合Excel打开）
cat /root/autodl-tmp/outputs/experiment_results.csv

# 查看运行日志
cat /root/autodl-tmp/outputs/run_log.txt
```

### 5.3 ⭐ 把结果发给我

**方法一：直接复制文本（推荐）**

```bash
cat /root/autodl-tmp/outputs/experiment_results.txt
# 复制全部内容 → 粘贴发给我
```

txt格式已经优化过了——表格清晰、文本量适中、方便复制粘贴。

**方法二：如果有报错，把debug日志发给我**

```bash
cat /root/autodl-tmp/outputs/run_log.txt
# 或只看最后几行
tail -50 /root/autodl-tmp/outputs/run_log.txt
```

> ⚠️ 不要发json/rar等格式！txt和csv是我能接收的格式。文本太长就只发最后50行。

---

## 🔧 六、常见问题排错

### 6.1 CUDA不可用

```bash
# 检查CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

如果显示 `False`：
- 检查镜像是否选了 PyTorch+CUDA 版本
- 确认GPU实例是否开机（不是"无卡模式"）
- 运行 `nvidia-smi` 检查GPU是否可见

### 6.2 数据维度不匹配

```bash
# 用--debug查看详细维度
python3 run_main.py --config configs/autodl_gpu.yaml --epochs 5 --debug
```

把debug输出发给我，我会帮你修改配置。

### 6.3 GPU内存不足

修改配置文件中的 `batch_size: 32` → `batch_size: 16` 或 `hidden_dim: 64` → `hidden_dim: 32`

RTX4090有24GB VRAM，正常配置不会爆内存。

### 6.4 数据盘不可写

```bash
# 检查数据盘
ls -la /root/autodl-tmp/
# 如果不存在，手动挂载
mkdir -p /root/autodl-tmp
```

### 6.5 数据下载失败

没关系！仿真兜底数据会自动生成。代码会在仿真数据上跑完所有8个模型。

### 6.6 运行中断

如果实验跑了一半断了：
```bash
# 查看是否有部分结果
ls -la /root/autodl-tmp/outputs/

# 重新运行（数据会从缓存加载，不用重新生成）
python3 run_main.py --config configs/autodl_gpu.yaml --skip-download
```

---

## 🗺️ 七、完整操作流程（从零开始）

```
Step 1:  AutoDL租GPU实例（RTX4090, PyTorch2.4+cu24镜像）
         → 打开JupyterLab终端

Step 2:  上传 retrospective_prediction.zip
         → unzip → cd /root/retrospective_prediction

Step 3:  bash setup_autodl.sh
         → 自动配置环境 + 下载数据 + 跑实验

Step 4:  如果Step 3失败，手动操作：
         → python3 download_datasets.py           # 先拿数据
         → python3 run_main.py --epochs 5 --debug # 快速测试
         → 把日志发给我确认OK
         → python3 run_main.py                    # 正式200轮

Step 5:  cat /root/autodl-tmp/outputs/experiment_results.txt
         → 复制结果发给我

Step 6:  AutoDL关机省钱！
```

---

## 📋 八、环境信息速查表

| 项目 | 你的配置 | 代码兼容要求 | 状态 |
|------|---------|-------------|------|
| GPU | RTX 4090 (24GB) | 任何NVIDIA GPU | ✅ |
| CUDA Driver | 13.0 (nvidia-smi) | >= 12.4 | ✅ 向下兼容 |
| PyTorch CUDA Runtime | 12.4 (cu24) | == 12.4 | ✅ |
| Python | 3.12 | >= 3.10 | ✅ |
| PyTorch | 2.4+ | >= 2.4.0 | ✅ |
| NVIDIA-SMI | 580.105.08 | - | ✅ |
| 数据盘 | /root/autodl-tmp | - | ✅ 自动检测 |
| 数据盘设备 | /dev/md0 | - | ✅ |

> ⭐ **关键安全措施**（已在代码中实现）：
> - DATA_ROOT自动检测 → 数据盘 `/root/autodl-tmp` 优先
> - input_dim从数据动态获取 → 覆盖config硬编码值
> - geo_dim从数据动态获取 → 覆盖config硬编码值
> - GPU自动检测 cuda/cpu
> - 所有config数值做类型安全转换（float()/int()）
> - Geo-FiLM einsum修复 → 显式广播+sum（无维度膨胀风险）

---

## 🎉 开始吧！

```bash
# 最简操作：一行命令搞定
cd /root/retrospective_prediction && bash setup_autodl.sh

# 然后把结果发给我：
cat /root/autodl-tmp/outputs/experiment_results.txt
```

加油小虎哥哥！🐯🧊
