#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_datasets.py — Standalone data download and preprocessing script

Usage:
    python3 download_datasets.py                  # Download all + preprocess
    python3 download_datasets.py --preprocess-only # Only preprocess existing raw data
    python3 download_datasets.py --status          # Print download status only

Downloads ALL candidate datasets (A1 chaimu, A4 qaidam, A5 northeast, A8 yeniugou),
saves raw data into datasets/raw/, preprocesses into datasets/processed/ in .npz format.
If no real data available, generates simulation fallback data.
"""

import sys
import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# DATA_ROOT — 数据存放根目录
# ============================================================
# AutoDL环境：数据盘 /root/autodl-tmp（大容量，/dev/md0挂载点）
# 本地开发：项目根目录（sandbox等）
AUTODL_DATA_DISK = '/root/autodl-tmp'


def resolve_data_root(project_root: str, cli_data_root: str = None) -> str:
    """Resolve DATA_ROOT (same logic as run_main.py)."""
    if cli_data_root:
        data_root = os.path.abspath(cli_data_root)
        os.makedirs(data_root, exist_ok=True)
        return data_root
    env_data_root = os.environ.get('DATA_ROOT')
    if env_data_root:
        data_root = os.path.abspath(env_data_root)
        os.makedirs(data_root, exist_ok=True)
        return data_root
    if os.path.exists(AUTODL_DATA_DISK) and os.path.isdir(AUTODL_DATA_DISK):
        test_file = os.path.join(AUTODL_DATA_DISK, '.data_root_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return AUTODL_DATA_DISK
        except (OSError, PermissionError):
            pass
    return project_root

import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path


# Dataset registry with download URLs
DATASET_REGISTRY = {
    "A1": {
        "name": "Chaimu Railway Deformation + Damage (柴木铁路)",
        "config_file": "configs/chaimu_railway.yaml",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/f7be4a1d-2506",
        "doi": "f7be4a1d-2506",
        "format": "mixed",
        "description": "祁连山柴木铁路形变+病害数据",
    },
    "A4": {
        "name": "Qaidam Basin Corridor InSAR (柴达木走廊InSAR)",
        "config_file": "configs/qaidam_inSAR.yaml",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/10.11888/cryos.tpdc.300400",
        "doi": "10.11888/cryos.tpdc.300400",
        "format": "geotiff",
        "description": "青藏工程走廊二维形变数据集",
    },
    "A5": {
        "name": "Northeast China Permafrost Engineering (东北冻土)",
        "config_file": "configs/northeast_frost.yaml",
        "source": "NCDC",
        "url": "https://data.ncdc.ac.cn/zh-hans/data/70c33bc7",
        "doi": "70c33bc7",
        "format": "mixed",
        "description": "东北冻土工程基础稳定性数据",
    },
    "A8": {
        "name": "Yeniugou Permafrost Deformation + ALT (野牛沟)",
        "config_file": "configs/yeniugou_alt.yaml",
        "source": "TPDC",
        "url": "https://data.tpdc.ac.cn/zh-hans/data/dd24266c",
        "doi": "dd24266c",
        "format": "mixed",
        "description": "野牛沟冻土形变+ALT数据",
    },
}


def print_banner():
    """Print download script banner."""
    print()
    print("=" * 60)
    print("  Dataset Download & Preprocessing Script")
    print("  Permafrost Settlement Prediction Experiment")
    print("=" * 60)


def attempt_download(dataset_id: str, dataset_info: dict, raw_dir: str) -> dict:
    """
    Attempt to download a dataset.
    
    Args:
        dataset_id: Dataset identifier (A1, A4, A5, A8)
        dataset_info: Dataset metadata dictionary
        raw_dir: Path to datasets/raw/ directory
    
    Returns:
        status_dict: Download status information
    """
    url = dataset_info["url"]
    dataset_name = dataset_id.lower()
    target_path = os.path.join(raw_dir, f"{dataset_name}_raw")
    
    status = {
        "dataset_id": dataset_id,
        "name": dataset_info["name"],
        "url": url,
        "source": dataset_info["source"],
        "downloaded": False,
        "raw_path": None,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }
    
    print(f"\n  [{dataset_id}] {dataset_info['name']}")
    print(f"  Source: {dataset_info['source']}")
    print(f"  URL: {url}")
    
    try:
        import urllib.request
        import socket
        socket.setdefaulttimeout(30)
        
        print(f"  Attempting download...")
        urllib.request.urlretrieve(url, target_path)
        
        # Verify download
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            status["downloaded"] = True
            status["raw_path"] = target_path
            file_size = os.path.getsize(target_path)
            print(f"  ✓ Download successful! ({file_size} bytes)")
        else:
            status["error"] = "Downloaded file is empty or missing"
            print(f"  ✗ Downloaded file appears empty")
            
    except urllib.error.HTTPError as e:
        status["error"] = f"HTTP Error {e.code}: {e.reason}"
        print(f"  ✗ HTTP Error {e.code}: {e.reason}")
        print(f"  💡 This data source may require registration/authentication.")
        print(f"     Try visiting the URL manually in a browser: {url}")
        
    except urllib.error.URLError as e:
        status["error"] = f"URL Error: {e.reason}"
        print(f"  ✗ URL Error: {e.reason}")
        print(f"  💡 Network may be unavailable or the URL may require auth.")
        
    except socket.timeout:
        status["error"] = "Connection timed out (30s)"
        print(f"  ✗ Connection timed out")
        print(f"  💡 The server may be slow or unreachable from this network.")
        
    except Exception as e:
        status["error"] = f"Unexpected error: {str(e)}"
        print(f"  ✗ Unexpected error: {e}")
        print(f"  💡 Try manual download from: {url}")
    
    return status


def preprocess_dataset(dataset_id: str, dataset_info: dict, 
                       raw_dir: str, processed_dir: str) -> dict:
    """
    Preprocess a downloaded dataset into .npz format.
    
    If raw data is not available, generate simulation fallback data using
    the config file for that dataset.
    
    Args:
        dataset_id: Dataset identifier
        dataset_info: Dataset metadata dictionary
        raw_dir: Path to datasets/raw/
        processed_dir: Path to datasets/processed/
    
    Returns:
        preprocess_status: Dict with preprocessing results
    """
    from data import load_config, SyntheticDataGenerator, save_processed
    
    dataset_name = dataset_id.lower()
    config_path = os.path.join(PROJECT_ROOT, dataset_info["config_file"])
    npz_path = os.path.join(processed_dir, f"{dataset_name}.npz")
    
    status = {
        "dataset_id": dataset_id,
        "preprocessed": False,
        "npz_path": None,
        "data_source": None,
        "error": None,
    }
    
    # Try to load config
    try:
        config = load_config(config_path)
    except Exception as e:
        status["error"] = f"Config load failed: {e}"
        print(f"  ✗ Config load failed for {dataset_id}: {e}")
        return status
    
    # Check if raw data exists and can be loaded
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw")
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
        print(f"  Found raw data for {dataset_id}, attempting preprocessing...")
        try:
            from data import load_npz
            # Try to load if it's npz format
            if raw_path.endswith('.npz') or os.path.exists(raw_path + '.npz'):
                data_dict = load_npz(raw_path if raw_path.endswith('.npz') else raw_path + '.npz')
                save_processed(data_dict, npz_path)
                status["preprocessed"] = True
                status["npz_path"] = npz_path
                status["data_source"] = "downloaded"
                print(f"  ✓ Preprocessed from raw data → {npz_path}")
                return status
        except Exception as e:
            print(f"  ✗ Raw data preprocessing failed: {e}")
            print(f"  → Falling back to simulation data...")
    
    # Fallback: generate simulation data
    print(f"  No usable raw data for {dataset_id}, generating simulation fallback...")
    config["data_source"] = "simulation"
    config["metadata_note"] = "Simulation fallback data (for framework validation only)"
    
    try:
        generator = SyntheticDataGenerator(config)
        data_dict = generator.generate()
        save_processed(data_dict, npz_path)
        status["preprocessed"] = True
        status["npz_path"] = npz_path
        status["data_source"] = "simulation"
        print(f"  ✓ Simulation data generated → {npz_path}")
        print(f"  ⚠️  Note: Simulation data is for framework validation only.")
    except Exception as e:
        status["error"] = f"Simulation generation failed: {e}"
        print(f"  ✗ Simulation generation failed: {e}")
    
    return status


def generate_status_report(download_statuses: list, 
                           preprocess_statuses: list) -> str:
    """
    Generate a human-readable download status report.
    
    Args:
        download_statuses: List of download status dicts
        preprocess_statuses: List of preprocess status dicts
    
    Returns:
        report_text: Formatted status report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  Dataset Download & Preprocessing Status Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    
    n_downloaded = sum(1 for s in download_statuses if s["downloaded"])
    n_processed = sum(1 for s in preprocess_statuses if s["preprocessed"])
    n_simulation = sum(1 for s in preprocess_statuses if s.get("data_source") == "simulation")
    
    lines.append(f"Summary: {n_downloaded}/{len(download_statuses)} datasets downloaded, "
                 f"{n_processed}/{len(preprocess_statuses)} preprocessed "
                 f"({n_simulation} using simulation fallback)")
    lines.append("")
    
    # Download status table
    lines.append("Download Status:")
    lines.append("-" * 60)
    for ds in download_statuses:
        status_str = "✓ Available" if ds["downloaded"] else "✗ Failed"
        lines.append(f"  [{ds['dataset_id']}] {ds['name']}")
        lines.append(f"    Download: {status_str}")
        if ds["error"]:
            lines.append(f"    Error: {ds['error']}")
        if ds["downloaded"]:
            lines.append(f"    Raw file: {ds['raw_path']}")
        else:
            lines.append(f"    Manual download URL: {ds['url']}")
            lines.append(f"    Source: {ds['source']} (may require registration)")
        lines.append("")
    
    # Preprocess status table
    lines.append("Preprocessing Status:")
    lines.append("-" * 60)
    for ps in preprocess_statuses:
        status_str = "✓ Done" if ps["preprocessed"] else "✗ Failed"
        source_str = ps.get("data_source", "unknown")
        lines.append(f"  [{ps['dataset_id']}] Preprocess: {status_str} (source: {source_str})")
        if ps["npz_path"]:
            lines.append(f"    NPZ file: {ps['npz_path']}")
        if ps["error"]:
            lines.append(f"    Error: {ps['error']}")
        lines.append("")
    
    lines.append("=" * 60)
    if n_simulation > 0:
        lines.append("")
        lines.append("⚠️  Note: Some datasets use simulation fallback data.")
        lines.append("    Simulation data is for framework validation only.")
        lines.append("    For publication results, replace with real data.")
    lines.append("")
    lines.append("To manually download datasets:")
    for ds in download_statuses:
        if not ds["downloaded"]:
            lines.append(f"    [{ds['dataset_id']}] Visit: {ds['url']}")
            lines.append(f"    Save to: datasets/raw/{ds['dataset_id'].lower()}_raw")
    lines.append("")
    lines.append("After manual download, re-run: python3 download_datasets.py --preprocess-only")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess permafrost datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download_datasets.py                  # Download all + preprocess
  python3 download_datasets.py --preprocess-only # Preprocess existing raw data only
  python3 download_datasets.py --status          # Show download status only
        """
    )
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Skip downloads, only preprocess existing raw data')
    parser.add_argument('--status', action='store_true',
                        help='Only print the download status report')
    parser.add_argument('--data-root', type=str, default=None,
                        help='数据存放根目录（AutoDL默认/root/autodl-tmp）')
    args = parser.parse_args()
    
    print_banner()
    
    # Resolve DATA_ROOT
    DATA_ROOT = resolve_data_root(PROJECT_ROOT, args.data_root)
    print(f"  DATA_ROOT: {DATA_ROOT}")
    if DATA_ROOT != PROJECT_ROOT:
        print(f"  → Data stored on AutoDL data disk: {DATA_ROOT}")
    print()
    
    # Use DATA_ROOT for all data paths (AutoDL data disk or project root)
    raw_dir = os.path.join(DATA_ROOT, 'datasets', 'raw')
    processed_dir = os.path.join(DATA_ROOT, 'datasets', 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # If --status only, just read and print existing status
    if args.status:
        status_file = os.path.join(DATA_ROOT, 'datasets', 'download_status.txt')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                print(f.read())
        else:
            print("No status report found. Run download_datasets.py first.")
        return
    
    download_statuses = []
    preprocess_statuses = []
    
    # Step 1: Download all datasets
    if not args.preprocess_only:
        print("\n[Step 1] Downloading datasets...")
        print("-" * 60)
        
        for dataset_id, dataset_info in DATASET_REGISTRY.items():
            ds = attempt_download(dataset_id, dataset_info, raw_dir)
            download_statuses.append(ds)
    else:
        print("\n[Step 1] Skipping downloads (--preprocess-only mode)")
        # Build download statuses from existing files
        for dataset_id, dataset_info in DATASET_REGISTRY.items():
            dataset_name = dataset_id.lower()
            raw_path = os.path.join(raw_dir, f"{dataset_name}_raw")
            ds = {
                "dataset_id": dataset_id,
                "name": dataset_info["name"],
                "url": dataset_info["url"],
                "source": dataset_info["source"],
                "downloaded": os.path.exists(raw_path) and os.path.getsize(raw_path) > 0,
                "raw_path": raw_path if os.path.exists(raw_path) else None,
                "error": None if os.path.exists(raw_path) else "Not downloaded",
                "timestamp": datetime.now().isoformat(),
            }
            download_statuses.append(ds)
    
    # Step 2: Preprocess all datasets
    print("\n[Step 2] Preprocessing datasets...")
    print("-" * 60)
    
    for dataset_id, dataset_info in DATASET_REGISTRY.items():
        ps = preprocess_dataset(dataset_id, dataset_info, raw_dir, processed_dir)
        preprocess_statuses.append(ps)
    
    # Step 3: Generate simulation fallback for any missing datasets
    print("\n[Step 3] Ensuring all datasets have processed data...")
    print("-" * 60)
    
    # Already handled in preprocess_dataset (simulation fallback)
    
    # Step 4: Generate status report
    print("\n[Step 4] Generating status report...")
    report = generate_status_report(download_statuses, preprocess_statuses)
    
    status_file = os.path.join(DATA_ROOT, 'datasets', 'download_status.txt')
    with open(status_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Status report saved: {status_file}")
    
    # Print final summary
    n_downloaded = sum(1 for s in download_statuses if s["downloaded"])
    n_processed = sum(1 for s in preprocess_statuses if s["preprocessed"])
    n_simulation = sum(1 for s in preprocess_statuses if s.get("data_source") == "simulation")
    
    print()
    print("=" * 60)
    print(f"  Downloaded: {n_downloaded}/{len(download_statuses)} datasets")
    print(f"  Preprocessed: {n_processed}/{len(preprocess_statuses)} datasets")
    print(f"  Simulation fallback: {n_simulation} datasets")
    print("=" * 60)
    
    if n_simulation > 0:
        print()
        print("  ⚠️  Some datasets use simulation fallback data.")
        print("  Simulation data is for framework validation ONLY.")
        print("  For publication, replace with real data from public sources.")
        print()
        print("  To manually download, visit the URLs in datasets/download_status.txt")
        print("  After manual download, run: python3 download_datasets.py --preprocess-only")
    
    print()
    print("  ✅ Dataset preparation complete. You can now run:")
    print("     python3 run_main.py --config configs/<config_file>.yaml")
    print()


if __name__ == '__main__':
    main()
