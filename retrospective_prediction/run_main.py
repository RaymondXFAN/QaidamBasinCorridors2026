#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_main.py — Retrospective Prediction Experiment 主入口

实验目标：在公开真实/仿真数据上验证 Geo-FiLM ST-GAT 的冻土-基础设施沉降预测能力

流程：
  1. 加载YAML配置（类型安全转换）
  2. 获取数据（优先公开数据 → 仿真兜底）
  3. 预处理（标准化 + 时间分割 + 维度校验）
  4. 运行所有模型对比实验（8个模型）
  5. 输出结果（txt + csv + 物理约束检查）

Verbose/Debug 控制：
  --verbose / -v : LEVEL 1 (default) — 打印关键进度消息
  --debug / -d   : LEVEL 2 — 打印详细shapes, tensor信息, model internals
  --quiet / -q   : LEVEL 0 — 最小输出（仅最终结果）
  --log-file     : 可选，保存所有输出到 outputs/run_log.txt

AutoDL关键约束（从过往bug修复经验提炼）：
  - input_dim 从 X_train.shape[2] 动态获取，覆盖config硬编码值
  - geo_dim 从 geo_features.shape[1] 动态获取
  - 所有config数值做 float()/int() 类型安全转换
  - GPU自动检测 cuda/cpu
"""

import sys
import os
import io

# 项目根目录加入 sys.path（确保包导入正确）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import time
import numpy as np
from datetime import datetime


# ============================================================
# DATA_ROOT — 数据/输出存放根目录
# ============================================================
# AutoDL环境：数据盘 /root/autodl-tmp（大容量，/dev/md0挂载点）
# 本地开发：项目根目录（sandbox等）
# 优先级：CLI --data-root > 环境变量 DATA_ROOT > 自动检测 > 默认PROJECT_ROOT
AUTODL_DATA_DISK = '/root/autodl-tmp'


def resolve_data_root(project_root: str, cli_data_root: str = None) -> str:
    """
    Resolve DATA_ROOT for data and output storage.
    
    Priority:
      1. CLI argument --data-root (explicit override)
      2. Environment variable DATA_ROOT
      3. Auto-detection: /root/autodl-tmp exists (AutoDL cloud)
      4. Default: project root (local development/sandbox)
    
    Returns:
        DATA_ROOT: absolute path for datasets/ and outputs/
    """
    if cli_data_root:
        data_root = os.path.abspath(cli_data_root)
        os.makedirs(data_root, exist_ok=True)
        return data_root
    
    env_data_root = os.environ.get('DATA_ROOT')
    if env_data_root:
        data_root = os.path.abspath(env_data_root)
        os.makedirs(data_root, exist_ok=True)
        return data_root
    
    # Auto-detect AutoDL data disk
    if os.path.exists(AUTODL_DATA_DISK) and os.path.isdir(AUTODL_DATA_DISK):
        # Verify it's actually writable (not just a symlink)
        test_file = os.path.join(AUTODL_DATA_DISK, '.data_root_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return AUTODL_DATA_DISK
        except (OSError, PermissionError):
            pass  # Not writable, fall through
    
    # Default: project root (sandbox/local development)
    return project_root


class TeeLogger:
    """Logger that tees output to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def print_banner(verbose=1):
    """打印实验启动横幅"""
    if verbose <= 0:
        return
    print()
    print("=" * 70)
    print("  Retrospective Prediction Experiment")
    print("  Geo-FiLM ST-GAT Validation on Public Permafrost Data")
    print("=" * 70)


def vprint(msg="", verbose=1, level=1):
    """Conditional print based on verbose level.
    
    Args:
        msg: message to print
        verbose: current verbose level (0=quiet, 1=progress, 2=debug)
        level: minimum verbose level required to print this message
    """
    if verbose >= level:
        print(msg)


def main():
    # ===== Argument parsing =====
    parser = argparse.ArgumentParser(
        description='Retrospective Prediction Experiment: Geo-FiLM ST-GAT Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例命令:
  python run_main.py                                    # 默认仿真兜底 (verbose=1)
  python run_main.py --config configs/qaidam_inSAR.yaml # 青藏走廊配置
  python run_main.py --skip-download                    # 用本地缓存数据
  python run_main.py --epochs 50                        # 快速调试（减少epoch）
  python run_main.py --verbose                          # LEVEL 1: 关键进度消息
  python run_main.py --debug                            # LEVEL 2: 详细debug信息
  python run_main.py --quiet                            # LEVEL 0: 仅最终结果
  python run_main.py --log-file                         # 保存输出到 outputs/run_log.txt
        """
    )
    parser.add_argument('--config', type=str, default='configs/simulation_fallback.yaml',
                        help='YAML配置文件路径（相对于项目根目录）')
    parser.add_argument('--skip-download', action='store_true',
                        help='跳过数据下载，使用本地缓存')
    parser.add_argument('--skip-train', action='store_true',
                        help='跳过训练，仅做数据生成+维度检查')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的训练epoch数（调试用）')
    parser.add_argument('--device', type=str, default=None,
                        help='指定设备 cpu/cuda/cuda:0')
    # Data root (AutoDL data disk)
    parser.add_argument('--data-root', type=str, default=None,
                        help='数据/输出存放根目录（AutoDL默认/root/autodl-tmp）')
    # Verbose/Debug/Quiet controls
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='LEVEL 1 (default): print key progress messages')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='LEVEL 2: print detailed shapes, tensor info, model internals')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='LEVEL 0: minimal output (only final results)')
    parser.add_argument('--log-file', nargs='?', const='default', default='default',
                        help='Save all output to a log file (default path: DATA_ROOT/outputs/run_log.txt)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable log file output (print to console only)')
    args = parser.parse_args()
    
    # Determine verbose level
    if args.quiet:
        verbose = 0
    elif args.debug:
        verbose = 2
    else:
        verbose = 1  # default or --verbose
    
    # ===== Resolve DATA_ROOT =====
    DATA_ROOT = resolve_data_root(PROJECT_ROOT, args.data_root)
    
    # Set up log file — DEFAULT is now ON (use --no-log to disable)
    log_file_path = None
    if not args.no_log and args.log_file:
        # Resolve DATA_ROOT early for log file path
        _data_root_for_log = resolve_data_root(PROJECT_ROOT, args.data_root)
        if args.log_file == 'default':
            log_file_path = os.path.join(_data_root_for_log, 'outputs', 'run_log.txt')
        else:
            log_file_path = args.log_file
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        tee_logger = TeeLogger(log_file_path)
        sys.stdout = tee_logger
    
    print_banner(verbose)
    vprint(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", verbose, 1)
    vprint(f"  Config: {args.config}", verbose, 1)
    vprint(f"  Verbose level: {verbose}", verbose, 1)
    vprint(f"  DATA_ROOT: {DATA_ROOT}", verbose, 1)
    vprint(f"  Project root: {PROJECT_ROOT}", verbose, 1)
    if DATA_ROOT != PROJECT_ROOT:
        vprint(f"  → Data/outputs on AutoDL data disk: {DATA_ROOT}", verbose, 1)
    vprint()

    # ===== Step 1: Load YAML config (type-safe) =====
    vprint("[Step 1/5] Loading YAML config...", verbose, 1)
    from data import load_config, load_npz, normalize_features, split_time_series
    from data import validate_data, save_processed, SyntheticDataGenerator
    from data import download_dataset, handle_download_failure

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = load_config(config_path)
    
    # Inject verbose level into config for trainer/model use
    config['verbose'] = verbose

    # CLI overrides
    if args.epochs is not None:
        config['epochs'] = int(args.epochs)
        vprint(f"  epochs overridden to {args.epochs}", verbose, 1)

    # GPU auto-detect
    if args.device:
        config['device'] = args.device
    else:
        import torch
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print config summary
    vprint(f"  Dataset: {config.get('dataset_name', 'N/A')}", verbose, 1)
    vprint(f"  Region: {config.get('region', 'N/A')}", verbose, 1)
    vprint(f"  Device: {config['device']}", verbose, 1)
    vprint(f"  Dimensions: N={config.get('N_nodes')}, T={config.get('T_steps')}, "
          f"F={config.get('F_features', 12)}, G={config.get('G_geo_features', 3)}", verbose, 1)
    vprint(f"  Data source: {config.get('data_source', 'simulation')}", verbose, 1)

    # ===== Step 2: Data acquisition =====
    vprint("\n[Step 2/5] Data acquisition...", verbose, 1)
    # Data stored on DATA_ROOT (AutoDL data disk or project root)
    data_dir = os.path.join(DATA_ROOT, 'datasets', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    data_filename = f"{config['dataset_name']}.npz"
    data_path = os.path.join(data_dir, data_filename)

    data_dict = None
    data_source_type = "unknown"

    # Priority: cached > public download > simulation fallback
    if args.skip_download and os.path.exists(data_path):
        vprint(f"  Using cached data: {data_path}", verbose, 1)
        data_dict = load_npz(data_path)
        data_source_type = "cached"

    elif config.get('data_source', 'simulation') == 'simulation':
        vprint(f"  Generating simulation data (Stefan equation driven)...", verbose, 1)
        generator = SyntheticDataGenerator(config)
        data_dict = generator.generate()
        save_processed(data_dict, data_path)
        vprint(f"  Simulation data saved: {data_path}", verbose, 1)
        data_source_type = "simulation"

    else:
        vprint(f"  Attempting download: {config.get('data_source')}", verbose, 1)
        try:
            download_dataset(config)
            if os.path.exists(data_path):
                data_dict = load_npz(data_path)
                data_source_type = "download"
                vprint(f"  Public data download successful!", verbose, 1)
            else:
                raise FileNotFoundError(f"File not found after download: {data_path}")
        except Exception as e:
            vprint(f"  Download failed: {e}", verbose, 1)
            vprint(f"  -> Falling back to simulation data...", verbose, 1)
            data_dict = handle_download_failure(config)
            if data_dict is not None:
                save_processed(data_dict, data_path)
                data_source_type = "simulation_fallback"
                vprint(f"  Simulation fallback data saved: {data_path}", verbose, 1)
            else:
                # Direct simulation generation as last resort
                vprint(f"  -> Direct simulation data generation...", verbose, 1)
                generator = SyntheticDataGenerator(config)
                data_dict = generator.generate()
                save_processed(data_dict, data_path)
                data_source_type = "simulation_fallback_direct"

    if data_dict is None:
        print("  ERROR: No data available! Check config file.")
        sys.exit(1)

    # Print raw data dimensions
    vprint(f"  Data source type: {data_source_type}", verbose, 1)
    if verbose >= 2:
        for key in ['X', 'y', 'adj_matrix', 'geo_features']:
            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                print(f"     [DEBUG] {key}.shape = {data_dict[key].shape}, dtype = {data_dict[key].dtype}")
    elif verbose >= 1:
        for key in ['X', 'y', 'adj_matrix', 'geo_features']:
            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                print(f"     {key}.shape = {data_dict[key].shape}")

    # ===== Step 3: Preprocess =====
    vprint("\n[Step 3/5] Data preprocessing...", verbose, 1)

    # 3a. Feature normalization
    norm_method = config.get('normalization_method', 'zscore')
    vprint(f"  Feature normalization: method={norm_method}", verbose, 1)

    X_norm, norm_params = normalize_features(data_dict['X'], method=norm_method)
    data_dict['X'] = X_norm
    data_dict['norm_params'] = norm_params

    # 3b. Time series split
    T_in = int(config.get('T_in', 24))
    T_out = int(config.get('T_out', 1))
    train_ratio = float(config.get('train_ratio', 0.8))

    # Inject T_in/T_out into data_dict for split_time_series
    data_dict['T_in'] = T_in
    data_dict['T_out'] = T_out

    vprint(f"  Time split: T_in={T_in}, T_out={T_out}, train_ratio={train_ratio}", verbose, 1)
    split_data = split_time_series(data_dict, train_ratio=train_ratio)

    # 3c. Dimension validation
    vprint("\n  Dimension validation:", verbose, 1)
    is_valid = validate_data(
        split_data['X_train'], split_data['y_train'],
        split_data['adj_matrix'], split_data['geo_features']
    )
    if not is_valid:
        print("  ERROR: Data dimension validation failed!")
        sys.exit(1)

    # ===== CRITICAL: Dynamic input_dim override (核心安全措施) =====
    actual_input_dim = split_data['X_train'].shape[2]  # F from flat format (N*W, T_in, F)
    actual_geo_dim = split_data['geo_features'].shape[1]  # G from (N, G)
    N_nodes = split_data.get('N_nodes', split_data['geo_features'].shape[0])
    split_data['N_nodes'] = N_nodes  # ensure N_nodes is available for trainer

    # Override config hardcoded values! Key rule from AutoDL bug fix experience
    config['input_dim'] = actual_input_dim
    config['geo_dim'] = actual_geo_dim

    config_F = config.get('F_features', 12)
    config_G = config.get('G_geo_features', 3)

    vprint("", verbose, 1)
    if verbose >= 1:
        print("  ===========================================")
        print("  Dynamic dimension override (AutoDL safety)")
        print("  ===========================================")
        print(f"  config F_features = {config_F}  ->  actual input_dim = {actual_input_dim}")
        print(f"  config G_geo_features = {config_G}  ->  actual geo_dim = {actual_geo_dim}")
        print("  ===========================================")

    if verbose >= 2:
        print(f"  [DEBUG] X_train shape: {split_data['X_train'].shape}")
        print(f"  [DEBUG] X_train_graph shape: {split_data['X_train_graph'].shape}")
        print(f"  [DEBUG] y_train shape: {split_data['y_train'].shape}")
        print(f"  [DEBUG] adj_matrix shape: {split_data['adj_matrix'].shape}")
        print(f"  [DEBUG] geo_features shape: {split_data['geo_features'].shape}")

    if actual_input_dim != config_F:
        vprint(f"  WARNING: Feature dimension mismatch! Overridden {actual_input_dim} -> config {config_F}", verbose, 1)
    else:
        vprint(f"  Feature dimension consistent ({actual_input_dim})", verbose, 1)

    if actual_geo_dim != config_G:
        vprint(f"  WARNING: geo dimension mismatch! Overridden {actual_geo_dim} -> config {config_G}", verbose, 1)
    else:
        vprint(f"  geo dimension consistent ({actual_geo_dim})", verbose, 1)

    # ===== Step 4: Run all experiments =====
    if args.skip_train:
        vprint("\n[Step 4/5] Skipping training (--skip-train mode)", verbose, 1)
        vprint("  Data ready, dimension validation passed.", verbose, 1)
    else:
        vprint("\n[Step 4/5] Running all model comparison experiments...", verbose, 1)
        vprint(f"  Models: Geo-FiLM ST-GAT | Standard ST-GAT | ST-GCN | DCRNN | "
              f"PINN | Stefan | RF | XGBoost", verbose, 1)

        from trainers import run_all_experiments
        from evaluation.metrics import format_results_txt, format_results_csv
        from evaluation.physical_constraint import PhysicalConstraintChecker

        start_time = time.time()

        output_dir = os.path.join(DATA_ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)

        results = run_all_experiments(
            config, split_data,
            output_dir=output_dir
        )

        elapsed = time.time() - start_time
        vprint(f"\n  All experiments completed in {elapsed:.1f}s ({elapsed/60:.1f}min)", verbose, 1)

        # ===== Step 5: Save and output results =====
        vprint("\n[Step 5/5] Saving results...", verbose, 1)

        # 5a. Plain text results table
        txt_table = format_results_txt(results)
        txt_path = os.path.join(output_dir, 'experiment_results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Retrospective Prediction Experiment Results\n")
            f.write(f"={'='*69}\n\n")
            f.write(f"Dataset: {config.get('dataset_name')}\n")
            f.write(f"Region: {config.get('region')}\n")
            f.write(f"Data source: {data_source_type}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Dimensions: input_dim={actual_input_dim}, geo_dim={actual_geo_dim}\n")
            if data_source_type.startswith('simulation'):
                f.write(f"WARNING: Results based on simulation data (Stefan equation driven),\n")
                f.write(f"         for framework validation only. Replace with real data for publication.\n")
            f.write("\n")
            f.write(txt_table)
            f.write("\n\n")
            # FiLM improvement
            geo_film_name = "Geo-FiLM ST-GAT"
            standard_name = "Standard ST-GAT"
            if geo_film_name in results and standard_name in results:
                gf_rmse = results[geo_film_name].get('rmse', float('inf'))
                std_rmse = results[standard_name].get('rmse', float('inf'))
                if std_rmse > 0 and gf_rmse < std_rmse:
                    improvement = (std_rmse - gf_rmse) / std_rmse * 100
                    f.write(f"FiLM Conditioning improvement: Geo-FiLM RMSE {improvement:.1f}% lower than Standard ST-GAT\n")
                    f.write(f"({gf_rmse:.4f} vs {std_rmse:.4f})\n")
        vprint(f"  Plain text results: {txt_path}", verbose, 1)

        # 5b. CSV results
        csv_text = format_results_csv(results)
        csv_path = os.path.join(output_dir, 'experiment_results.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(csv_text)
        vprint(f"  CSV results: {csv_path}", verbose, 1)

        # 5c. Physical constraint check
        vprint("\n  Physical constraint check (Stefan equation validation):", verbose, 1)
        checker = PhysicalConstraintChecker(
            k_thaw=float(config.get('k_thaw', 1.2)),
            L_latent=float(config.get('L_latent', 3.34e7))
        )
        phys_reports = {}
        geo_full = split_data['geo_features']  # (N, G) — real node geo features
        W_test = split_data.get('W_test', split_data.get('N_nodes', 40))
        # For flat-format y_pred (N*W_test), expand geo_features to match
        N_nodes_phys = split_data.get('N_nodes', geo_full.shape[0])
        for model_name, metrics in results.items():
            if 'y_pred' in metrics and isinstance(metrics['y_pred'], np.ndarray):
                y_pred_arr = metrics['y_pred'].flatten()
                # Expand geo_features to match y_pred length: (N*W, G)
                n_pred = len(y_pred_arr)
                n_repeat = max(1, n_pred // N_nodes_phys)
                geo_expanded = np.repeat(geo_full, n_repeat, axis=0)
                try:
                    report = checker.check(y_pred_arr, geo_expanded)
                    phys_reports[model_name] = report
                    viol_pct = report.get('violation_rate_pct', 'N/A')
                    vprint(f"    {model_name}: violation rate {viol_pct}%", verbose, 1)
                except Exception as e:
                    vprint(f"    {model_name}: check failed ({e})", verbose, 1)

        # ===== Final summary =====
        if verbose >= 1:
            print()
            print("=" * 70)
            print("  Experiment Results Summary")
            print("=" * 70)
            print(txt_table)

        # FiLM improvement highlight
        geo_film_name = "Geo-FiLM ST-GAT"
        standard_name = "Standard ST-GAT"
        if geo_film_name in results and standard_name in results:
            gf_rmse = results[geo_film_name].get('rmse', float('inf'))
            std_rmse = results[standard_name].get('rmse', float('inf'))
            if std_rmse > 0 and gf_rmse < std_rmse:
                improvement = (std_rmse - gf_rmse) / std_rmse * 100
                vprint(f"\n  FiLM Conditioning improvement: Geo-FiLM vs Standard ST-GAT "
                      f"RMSE reduced by {improvement:.1f}%", verbose, 1)
                vprint(f"     ({gf_rmse:.4f} vs {std_rmse:.4f})", verbose, 1)

        vprint()
        vprint(f"  Result files:", verbose, 1)
        vprint(f"     TXT: {txt_path}", verbose, 1)
        vprint(f"     CSV: {csv_path}", verbose, 1)
        vprint(f"     Data: {data_path}", verbose, 1)
        vprint()
        vprint(f"  Next steps:", verbose, 1)
        vprint(f"     1. Check outputs/experiment_results.txt", verbose, 1)
        vprint(f"     2. Use CSV data for paper Table X", verbose, 1)
        vprint(f"     3. If real data available, re-run: python run_main.py --config configs/qaidam_inSAR.yaml", verbose, 1)
        vprint()

    print("=" * 70)
    print("  Experiment completed")
    print("=" * 70)
    
    # Close log file if opened
    if log_file_path is not None:
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        vprint(f"  Log saved to: {log_file_path}", verbose, 1)


if __name__ == '__main__':
    main()
