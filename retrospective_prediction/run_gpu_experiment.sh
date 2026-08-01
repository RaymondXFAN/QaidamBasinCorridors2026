#!/bin/bash
# ============================================================
# AutoDL GPU 正式实验一键运行脚本
# 冻土-基础设施沉降预测实验
# ============================================================
# 📋 保姆级使用方法：
#   1. 把整个 retrospective_prediction 项目上传到 AutoDL
#   2. 打开 AutoDL 终端（JupyterLab → Terminal）
#   3. cd 到项目目录
#   4. 运行: bash run_gpu_experiment.sh
#   5. 等约10分钟，查看 results/ 目录的结果文件
# ============================================================

set -e

PROJECT_DIR="/root/retrospective_prediction"
cd "$PROJECT_DIR" || { echo "❌ 项目目录不存在: $PROJECT_DIR"; exit 1; }

echo ""
echo "============================================================"
echo "  🚀 冻土-基础设施沉降预测 — AutoDL GPU 正式实验"
echo "============================================================"

# ---- Step 1: 系统信息 ----
echo ""
echo "[Step 1/6] 🔍 系统信息检查"
echo "------------------------------------------------------------"

# GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU 检测到:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "⚠️  未检测到GPU，将使用CPU（速度较慢）"
fi

# CUDA & PyTorch
python3 -c "
import torch
print(f'✅ PyTorch版本: {torch.__version__}')
print(f'✅ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU内存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA不可用，将使用CPU训练')
" 2>&1 || echo "❌ PyTorch未安装"

# Python版本
echo "✅ Python版本: $(python3 --version)"

echo "------------------------------------------------------------"

# ---- Step 2: 安装依赖 ----
echo ""
echo "[Step 2/6] 📦 安装Python依赖"
echo "------------------------------------------------------------"

PACKAGES="numpy pandas scikit-learn matplotlib pyyaml xgboost"
for pkg in $PACKAGES; do
    import_name=$pkg
    if [ "$pkg" = "scikit-learn" ]; then import_name="sklearn"; fi
    if python3 -c "import $import_name" 2>/dev/null; then
        echo "  ✓ $pkg 已安装"
    else
        echo "  📥 安装 $pkg..."
        pip install $pkg -q
    fi
done

# torch (AutoDL通常预装)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  ✓ PyTorch CUDA版已安装"
else
    echo "  ⚠️  PyTorch CUDA版未安装或CUDA不可用"
    echo "     AutoDL镜像通常自带PyTorch+CUDA，请检查镜像选择"
fi

echo "  ⊘ torch_geometric 不安装（所有GNN模型手写实现）"

echo "依赖安装完成！"
echo "------------------------------------------------------------"

# ---- Step 3: 数据维度校验 ----
echo ""
echo "[Step 3/6] 📊 数据维度校验"
echo "------------------------------------------------------------"

python3 -c "
import os, numpy as np
data_dir = 'data'
if os.path.exists(data_dir):
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if npz_files:
        for f in npz_files:
            path = os.path.join(data_dir, f)
            data = np.load(path, allow_pickle=True)
            print(f'  数据文件: {f}')
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    print(f'    {key}: shape={arr.shape}, dtype={arr.dtype}')
    else:
        print('  [INFO] 无.npz文件，将生成仿真数据')
else:
    print('  [INFO] data目录不存在，将生成仿真数据')

# 校验配置文件
config_path = 'configs/autodl_gpu.yaml'
if os.path.exists(config_path):
    print(f'  ✓ 配置文件 {config_path} 存在')
else:
    print(f'  ❌ 配置文件 {config_path} 不存在！')
" 2>&1

echo "------------------------------------------------------------"

# ---- Step 4: 运行正式实验（200 epochs） ----
echo ""
echo "[Step 4/6] 🚀 运行正式实验（200 epochs × 8模型）"
echo "------------------------------------------------------------"
echo "预计耗时: ~10分钟（GPU） / ~40分钟（CPU）"
echo ""

CONFIG_FILE="configs/autodl_gpu.yaml"

if [ -f "$CONFIG_FILE" ]; then
    echo "使用配置: $CONFIG_FILE"
    echo "命令: python3 run_main.py --config $CONFIG_FILE"
    echo ""
    python3 run_main.py --config "$CONFIG_FILE" 2>&1 || {
        echo ""
        echo "❌ 实验运行失败！常见原因:"
        echo "  1. GPU内存不足 → 编辑 configs/autodl_gpu.yaml 减小 batch_size=16 或 hidden_dim=32"
        echo "  2. 数据维度不匹配 → 代码已内置动态input_dim覆盖"
        echo "  3. YAML类型错误 → 代码已内置float/int类型安全转换"
        exit 1
    }
else
    echo "⚠️  配置文件不存在，使用默认配置"
    python3 run_main.py 2>&1
fi

echo "------------------------------------------------------------"

# ---- Step 5: 结果文件说明 ----
echo ""
echo "[Step 5/6] 📋 结果文件说明"
echo "------------------------------------------------------------"

python3 -c "
import os, json
results_dir = 'results'
if os.path.exists(results_dir):
    files = os.listdir(results_dir)
    print(f'  结果目录: {results_dir}/')
    print(f'  文件数量: {len(files)}')
    for f in sorted(files):
        path = os.path.join(results_dir, f)
        size = os.path.getsize(path)
        if size > 1024*1024:
            print(f'    📄 {f} ({size/1024/1024:.1f} MB)')
        else:
            print(f'    📄 {f} ({size/1024:.1f} KB)')
    
    # 读取并展示结果摘要
    json_path = os.path.join(results_dir, 'simulation_fallback_results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        print()
        print('  📊 结果摘要:')
        for model_name, metrics in data.items():
            if model_name.startswith('_'):
                continue
            if isinstance(metrics, dict) and 'rmse' in metrics:
                rmse = metrics.get('rmse', 'N/A')
                r2 = metrics.get('r2', 'N/A')
                viol = metrics.get('phys_violation_rate', 'N/A')
                print(f'    {model_name}: RMSE={rmse}, R²={r2}, PhysViol={viol}%')
else:
    print('  ⚠️  结果目录不存在')
" 2>&1

echo "------------------------------------------------------------"

# ---- Step 6: 下一步指引 ----
echo ""
echo "[Step 6/6] 📝 下一步指引"
echo "------------------------------------------------------------"
echo ""
echo "  ✅ 实验完成！结果已保存到 results/ 目录"
echo ""
echo "  📝 论文使用建议:"
echo "    1. 查看 results/simulation_fallback_results.md — Markdown结果表格"
echo "    2. 查看 results/simulation_fallback_results.json — 详细JSON结果"
echo "    3. 将结果填入论文 Table X（实验对比表格）"
echo "    4. ⚠️ 仿真数据仅供框架验证，论文需补充真实数据结果"
echo ""
echo "  🔬 切换真实数据集（有数据时）:"
echo "    编辑 configs/autodl_gpu.yaml 的 data_source 和 dataset_name"
echo "    或创建新配置如 configs/qaidam_inSAR.yaml"
echo "    然后运行: python3 run_main.py --config configs/qaidam_inSAR.yaml"
echo ""
echo "  🛠️ 调整实验参数:"
echo "    编辑 configs/autodl_gpu.yaml:"
echo "    - epochs: 200 → 500（更充分训练）"
echo "    - hidden_dim: 64 → 128（更大模型）"
echo "    - lambda_phys: 0.3 → 0.5（更强物理约束）"
echo ""
echo "============================================================"
echo "  🎉 全部完成！感谢使用冻土-基础设施沉降预测实验框架"
echo "============================================================"
