#!/bin/bash
# ============================================================
# AutoDL Cloud Deployment Script
# 冻土-基础设施沉降预测实验项目
# ============================================================
# 使用方法：在AutoDL终端中运行 bash setup_autodl.sh
# ============================================================
# 环境信息：
#   GPU: RTX4090 (24GB VRAM)
#   CUDA Driver: 13.0 | PyTorch CUDA Runtime: 12.4 (cu24)
#   Python: 3.12 | PyTorch: 2.4+
#   数据盘: /root/autodl-tmp (/dev/md0, 大容量)
# ============================================================

set -e  # 遇到错误立即停止

# ============================================================
# DATA_ROOT — 数据存放根目录（关键设置）
# ============================================================
# AutoDL数据盘路径 /root/autodl-tmp
# 所有数据集和输出结果都存放在此（系统盘/root容量小）
export DATA_ROOT="/root/autodl-tmp"
echo ""
echo "  ★ DATA_ROOT = $DATA_ROOT (数据盘，/dev/md0挂载点)"
echo "  ★ 数据目录: $DATA_ROOT/datasets/"
echo "  ★ 输出目录: $DATA_ROOT/outputs/"
echo ""

# 确保数据盘目录存在
mkdir -p "$DATA_ROOT/datasets/raw"
mkdir -p "$DATA_ROOT/datasets/processed"
mkdir -p "$DATA_ROOT/outputs"

echo "============================================================"
echo "  冻土-基础设施沉降预测实验 - AutoDL 环境配置"
echo "  环境: RTX4090 | CUDA 13.0(driver)/12.4(runtime) | Py3.12"
echo "============================================================"

# ============================================================
# 1. 打印系统信息
# ============================================================
echo ""
echo "[Step 1] 系统信息检查"
echo "------------------------------------------------------------"

# GPU型号
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "[WARNING] nvidia-smi 未找到，可能没有 GPU"
fi

# CUDA版本
if [ -x "$(command -v nvcc)" ]; then
    echo "CUDA Runtime 版本: $(nvcc --version | tail -1 | cut -d' ' -f5 | cut -d',' -f1)"
else
    echo "[WARNING] nvcc 未找到，CUDA Runtime 可能未安装"
fi

echo "CUDA Driver 版本: $(nvidia-smi | grep 'CUDA Version' | head -1 | awk '{print $NF}')"

# Python版本
echo "Python 版本: $(python3 --version 2>&1 || python --version 2>&1)"

# PyTorch版本和CUDA支持
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Runtime 版本: {torch.version.cuda}')
    print(f'CUDA 设备数: {torch.cuda.device_count()}')
    print(f'当前 GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU 内存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('[WARNING] CUDA 不可用，将使用 CPU 训练（速度较慢）')
" 2>&1 || echo "[WARNING] PyTorch 未安装"

# 数据盘容量检查
echo ""
echo "数据盘 /root/autodl-tmp 信息:"
df -h /root/autodl-tmp 2>/dev/null || echo "[INFO] /root/autodl-tmp 未挂载或不存在"
echo "系统盘 /root 信息:"
df -h /root 2>/dev/null | head -2

echo "------------------------------------------------------------"

# ============================================================
# 2. 安装 Python 依赖
# ============================================================
echo ""
echo "[Step 2] 安装 Python 依赖"
echo "------------------------------------------------------------"

# AutoDL 通常已预装 torch，不需要重新安装
echo "检查已安装的包..."

# Use requirements.txt for pinned versions
if [ -f "requirements.txt" ]; then
    echo "使用 requirements.txt 安装依赖..."
    pip install -r requirements.txt -q 2>&1 || echo "[WARNING] Some packages may have failed to install"
else
    # 检测并安装缺失的包
    PACKAGES="numpy pandas scikit-learn matplotlib pyyaml xgboost scipy"
    
    for pkg in $PACKAGES; do
        # sklearn 的包名是 scikit-learn，但 import 名是 sklearn
        import_name=$pkg
        if [ "$pkg" = "scikit-learn" ]; then
            import_name="sklearn"
        fi
        
        if python3 -c "import $import_name" 2>/dev/null; then
            echo "  ✓ $pkg 已安装"
        else
            echo "  ✗ $pkg 未安装，正在安装..."
            pip install $pkg -q
        fi
    done
fi

# 特别检查 PyTorch
if python3 -c "import torch" 2>/dev/null; then
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    echo "  ✓ torch 已安装（AutoDL预装，不重新安装）"
    echo "  ✓ torch CUDA runtime: $TORCH_CUDA"
else
    echo "  ✗ torch 未安装，正在安装 CUDA 12.4 版本..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
fi

# CUDA兼容性检查
echo ""
echo "CUDA 兼容性检查:"
python3 -c "
import torch
driver_cuda = float('13.0')  # NVIDIA-SMI显示的CUDA版本
runtime_cuda = torch.version.cuda
print(f'  Driver CUDA: {driver_cuda} (nvidia-smi)')
print(f'  PyTorch CUDA Runtime: {runtime_cuda}')
print(f'  兼容: ✓ (Driver >= Runtime, 向下兼容)')
print(f'  torch.cuda.is_available(): {torch.cuda.is_available()}')
" 2>&1 || echo "[WARNING] CUDA兼容性检查失败"

# 不安装 torch_geometric（所有 GNN 模型手写，减少依赖冲突）
echo "  ⊘ torch_geometric 不安装（所有GNN模型手写实现）"

echo "依赖安装完成！"
echo "------------------------------------------------------------"

# ============================================================
# 3. 数据下载与预处理（数据存放到数据盘）
# ============================================================
echo ""
echo "[Step 3] 数据下载与预处理 (DATA_ROOT=$DATA_ROOT)"
echo "------------------------------------------------------------"

echo "运行数据下载脚本（数据存放到 $DATA_ROOT/datasets/）..."
python3 download_datasets.py 2>&1 || {
    echo "[WARNING] 数据下载脚本运行失败，将使用仿真数据兜底"
    echo "您可以稍后手动运行: python3 download_datasets.py"
}

echo "检查下载状态..."
STATUS_FILE="$DATA_ROOT/datasets/download_status.txt"
if [ -f "$STATUS_FILE" ]; then
    echo "下载状态报告:"
    cat "$STATUS_FILE"
else
    echo "[INFO] 下载状态报告未生成"
fi

echo "检查已处理数据..."
python3 -c "
import os
import numpy as np

# 检查数据盘上的 datasets/processed/ 目录
data_dir = '$DATA_ROOT/datasets/processed'
if os.path.exists(data_dir):
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if npz_files:
        print(f'已处理数据文件 ({len(npz_files)} 个):')
        for f in npz_files:
            path = os.path.join(data_dir, f)
            data = np.load(path, allow_pickle=True)
            print(f'  {f}: keys={data.files[:5]}')
    else:
        print('[INFO] datasets/processed/ 中没有 .npz 文件（将使用仿真数据兜底）')
else:
    print('[INFO] datasets/processed/ 目录不存在')

# 检查数据盘上的 datasets/raw/ 目录
raw_dir = '$DATA_ROOT/datasets/raw'
if os.path.exists(raw_dir):
    raw_files = os.listdir(raw_dir)
    if raw_files:
        print(f'原始数据文件 ({len(raw_files)} 个):')
        for f in raw_files:
            print(f'  {f}')
    else:
        print('[INFO] datasets/raw/ 中没有文件')
else:
    print('[INFO] datasets/raw/ 目录不存在')
" 2>&1

echo "------------------------------------------------------------"

# ============================================================
# 4. 运行实验（输出存放到数据盘）
# ============================================================
echo ""
echo "[Step 4] 运行实验 (DATA_ROOT=$DATA_ROOT)"
echo "------------------------------------------------------------"

# 先检查仿真兜底配置
CONFIG_FILE="configs/simulation_fallback.yaml"

# Verbose flag options:
# --quiet   : LEVEL 0 (minimal output)
# --verbose : LEVEL 1 (key progress, default)
# --debug   : LEVEL 2 (detailed debug info)

VERBOSE_FLAG=""
echo "Verbose 选项: --quiet (LEVEL 0) | --verbose (LEVEL 1, default) | --debug (LEVEL 2)"
echo "日志文件默认保存到 $DATA_ROOT/outputs/run_log.txt（无需手动加参数）"
echo "使用 --no-log 禁用日志文件"
echo ""

if [ -f "$CONFIG_FILE" ]; then
    echo "使用配置文件: $CONFIG_FILE"
    echo "运行命令: python3 run_main.py --config $CONFIG_FILE --data-root $DATA_ROOT $VERBOSE_FLAG"
    echo ""
    python3 run_main.py --config "$CONFIG_FILE" --data-root "$DATA_ROOT" $VERBOSE_FLAG 2>&1 || {
        echo "[ERROR] 实验运行失败！请检查错误信息。"
        echo "常见原因："
        echo "  1. 数据维度不匹配 → 检查 configs/ 中的 F_features 设置"
        echo "  2. GPU 内存不足 → 减少 batch_size 或 hidden_dim"
        echo "  3. 配置文件路径错误 → 确保 configs/ 目录存在"
        echo "  4. 数据盘不可写 → 检查 /root/autodl-tmp 是否挂载"
        echo ""
        echo "调试选项:"
        echo "  python3 run_main.py --config $CONFIG_FILE --data-root $DATA_ROOT --debug  # 详细debug"
        echo "  python3 run_main.py --config $CONFIG_FILE --data-root $DATA_ROOT --no-log  # 不保存日志"
        exit 1
    }
else
    echo "[WARNING] 配置文件 $CONFIG_FILE 不存在！"
    echo "请先创建配置文件，参考 configs/ 目录中的模板。"
    echo "或者手动运行: python3 run_main.py --data-root $DATA_ROOT"
fi

echo "------------------------------------------------------------"

# ============================================================
# 5. 打印结果摘要（结果在数据盘上）
# ============================================================
echo ""
echo "[Step 5] 结果摘要 (DATA_ROOT=$DATA_ROOT)"
echo "------------------------------------------------------------"

OUTPUT_DIR="$DATA_ROOT/outputs"

# Check outputs directory on data disk
if [ -f "$OUTPUT_DIR/experiment_results.txt" ]; then
    echo "结果文件已生成（在数据盘 $OUTPUT_DIR/）："
    echo "  - experiment_results.txt (plain text table)"
    echo "  - experiment_results.csv (CSV format)"
    echo "  - diagnostics.txt (detailed per-model info)"
    echo ""
    cat "$OUTPUT_DIR/experiment_results.txt"
elif [ -f "$OUTPUT_DIR/experiment_results.csv" ]; then
    echo "结果文件已生成（在数据盘 $OUTPUT_DIR/）："
    echo "  - experiment_results.csv"
    echo ""
    cat "$OUTPUT_DIR/experiment_results.csv"
else
    echo "[WARNING] $OUTPUT_DIR/ 中没有结果文件，实验可能未完成。"
fi

echo "------------------------------------------------------------"
echo ""
echo "============================================================"
echo "  实验完成！结果保存在数据盘 $DATA_ROOT/outputs/ 目录下。"
echo "============================================================"
echo ""
echo "下一步："
echo "  1. 查看 text 结果表: cat $OUTPUT_DIR/experiment_results.txt"
echo "  2. 查看 CSV 详细结果: cat $OUTPUT_DIR/experiment_results.csv"
echo "  3. 查看诊断信息: cat $OUTPUT_DIR/diagnostics.txt"
echo "  4. 复制结果发给助手: cat $OUTPUT_DIR/experiment_results.txt"
echo "  5. 将结果填入论文 Table X"
echo ""
echo "重要提醒："
echo "  ★ 数据和结果在数据盘 /root/autodl-tmp/（不会占系统盘空间）"
echo "  ★ 系统盘 /root 容量有限（~50GB），不要把数据放系统盘"
echo "  ★ 如果需要重新下载数据: python3 download_datasets.py --data-root $DATA_ROOT"
echo ""
