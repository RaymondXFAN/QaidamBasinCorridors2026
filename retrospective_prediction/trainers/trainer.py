#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic trainer for running all model experiments.

Provides:
  - train_model(): train a single model with data (auto-selects graph/flat format)
  - evaluate_model(): evaluate a fitted model
  - run_all_experiments(): run full comparison across all models

Key design:
  - Models with data_format='graph' receive (W, N, T_in, F) format data
  - Models with data_format='flat' receive (N*W, T_in, F) format data
  - adj_matrix (N, N) and geo_features (N, G) are passed to ALL models
"""

import os
import time
import json
import numpy as np
from typing import Dict, Optional, List

from models import (
    GeoFiLMSTGATWrapper,
    StandardSTGATWrapper,
    STGCNWrapper,
    DCRNNWrapper,
    PINNPermafrostWrapper,
    StefanOnlyWrapper,
)
from baselines import RandomForestBaseline, GBRTBaseline
from evaluation.metrics import compute_rmse, compute_mae, compute_r2, format_results_table, format_results_txt, format_results_csv


def _prepare_data_for_model(model, data_dict: dict, verbose: int = 1) -> dict:
    """
    Prepare data in the correct format for a model.
    
    Models with data_format='graph' receive:
      X_train/X_test: (W, N, T_in, F) — graph format
      y_train/y_test: (W, N, T_out) — graph format
      adj_matrix: (N, N)
      geo_features: (N, G)
    
    Models with data_format='flat' receive:
      X_train/X_test: (N*W, T_in, F) — flat format
      y_train/y_test: (N*W, T_out) — flat format
      adj_matrix: (N, N) — passed but not used for prediction
      geo_features: (N, G) — passed for evaluation
    
    Args:
        verbose: 0=quiet, 1=progress, 2=debug
    """
    fmt = getattr(model, 'data_format', 'flat')
    
    if fmt == 'graph':
        # Use graph-format data for GNN models
        model_data = {
            'X_train': data_dict['X_train_graph'],
            'y_train': data_dict['y_train_graph'],
            'X_test': data_dict['X_test_graph'],
            'y_test': data_dict['y_test_graph'],
            'adj_matrix': data_dict['adj_matrix'],
            'geo_features': data_dict['geo_features'],
        }
        if verbose >= 1:
            W, N, T_in, F = model_data['X_train'].shape
            print(f"[Trainer] Data format: graph -> (W={W}, N={N}, T_in={T_in}, F={F})")
        if verbose >= 2:
            for key in ['X_train', 'y_train', 'X_test', 'y_test', 'adj_matrix', 'geo_features']:
                if key in model_data:
                    arr = model_data[key]
                    print(f"[Trainer-DEBUG] {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
    else:
        # Use flat-format data for ML/PINN/Stefan models
        model_data = {
            'X_train': data_dict['X_train'],
            'y_train': data_dict['y_train'],
            'X_test': data_dict['X_test'],
            'y_test': data_dict['y_test'],
            'adj_matrix': data_dict['adj_matrix'],
            'geo_features': data_dict['geo_features'],
        }
        if verbose >= 1:
            print(f"[Trainer] Data format: flat -> X_train {model_data['X_train'].shape}")
        if verbose >= 2:
            for key in ['X_train', 'y_train', 'X_test', 'y_test', 'adj_matrix', 'geo_features']:
                if key in model_data:
                    arr = model_data[key]
                    print(f"[Trainer-DEBUG] {key}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
    
    return model_data


def train_model(model, config: dict, data_dict: dict) -> dict:
    """
    Generic training pipeline for a single model.
    
    Args:
        model:    BaseModel instance (with fit/predict/evaluate interface)
        config:   Configuration dict (includes 'verbose' level)
        data_dict: Dictionary with both graph and flat format data
    
    Returns:
        train_info: dict with training metadata
    """
    verbose = config.get('verbose', 1)
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"  Training: {model.get_name()}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Prepare data in correct format for this model
    model_data = _prepare_data_for_model(model, data_dict, verbose)
    
    # Get N_nodes for geo_features tiling (needed by flat-format models)
    N_nodes = data_dict.get('N_nodes', data_dict['geo_features'].shape[0])
    
    if verbose >= 2:
        print(f"[Trainer-DEBUG] Training config: epochs={config.get('epochs')}, lr={config.get('lr')}, "
              f"batch_size={config.get('batch_size')}, device={config.get('device')}")
        print(f"[Trainer-DEBUG] N_nodes={N_nodes}, model class={model.__class__.__name__}")
    
    model.fit(
        X_train=model_data['X_train'],
        y_train=model_data['y_train'],
        adj_matrix=model_data['adj_matrix'],
        geo_features=model_data['geo_features'],
        X_val=data_dict.get('X_val_graph') if getattr(model, 'data_format', 'flat') == 'graph' 
              else data_dict.get('X_val'),
        y_val=data_dict.get('y_val_graph') if getattr(model, 'data_format', 'flat') == 'graph'
              else data_dict.get('y_val'),
    )
    
    elapsed = time.time() - start_time
    
    train_info = {
        'model_name': model.get_name(),
        'train_time_sec': elapsed,
        'status': 'success',
    }
    
    if verbose >= 1:
        print(f"[Trainer] {model.get_name()} trained in {elapsed:.1f}s")
    return train_info


def evaluate_model(model, data_dict: dict, config: dict = None) -> dict:
    """
    Generic evaluation pipeline for a fitted model.
    
    Args:
        model:     Fitted BaseModel instance
        data_dict: Dictionary with both graph and flat format data
        config:    Configuration dict (for verbose level)
    
    Returns:
        metrics: dict with evaluation results
    """
    verbose = (config or {}).get('verbose', 1)
    
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model.get_name()}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Prepare data in correct format for this model
    model_data = _prepare_data_for_model(model, data_dict, verbose)
    
    metrics = model.evaluate(
        X_test=model_data['X_test'],
        y_test=model_data['y_test'],
        adj_matrix=model_data['adj_matrix'],
        geo_features=model_data['geo_features'],
    )
    
    elapsed = time.time() - start_time
    metrics['eval_time_sec'] = elapsed
    
    if verbose >= 1:
        print(f"[Trainer] {model.get_name()} evaluated in {elapsed:.1f}s")
    if verbose >= 2:
        for k, v in metrics.items():
            if isinstance(v, (float, int)):
                print(f"[Trainer-DEBUG] {k}: {v}")
    return metrics


def run_all_experiments(config: dict, data_dict: dict,
                       output_dir: str = 'outputs') -> dict:
    """
    Run all model comparison experiments.
    
    Args:
        config:     Global configuration dict (includes 'verbose' level)
        data_dict:  Data dictionary with both graph and flat format data
        output_dir: Directory to save results
    
    Returns:
        all_results: dict mapping model_name -> metrics dict
    """
    verbose = config.get('verbose', 1)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all model instances
    model_classes = [
        (GeoFiLMSTGATWrapper, "Geo-FiLM ST-GAT"),
        (StandardSTGATWrapper, "Standard ST-GAT"),
        (STGCNWrapper, "ST-GCN"),
        (DCRNNWrapper, "DCRNN"),
        (PINNPermafrostWrapper, "PINN-Permafrost"),
        (StefanOnlyWrapper, "Stefan-Only"),
        (RandomForestBaseline, "Random Forest"),
        (GBRTBaseline, "XGBoost"),
    ]
    
    all_results = {}
    all_metrics_list = []
    
    for ModelClass, description in model_classes:
        if verbose >= 1:
            print(f"\n{'#'*60}")
            print(f"# Model: {description}")
            print(f"{'#'*60}")
        
        try:
            model = ModelClass(config)
            if verbose >= 1:
                print(f"[Experiment] data_format={getattr(model, 'data_format', 'flat')}")
            if verbose >= 2:
                print(f"[Experiment-DEBUG] model class={ModelClass.__name__}, config keys={list(config.keys())}")
            
            # Train
            train_info = train_model(model, config, data_dict)
            
            # Evaluate
            metrics = evaluate_model(model, data_dict, config)
            metrics['train_time_sec'] = train_info['train_time_sec']
            
            all_results[model.get_name()] = metrics
            all_metrics_list.append(metrics)
            
        except Exception as e:
            if verbose >= 0:
                print(f"[ERROR] {ModelClass.__name__} failed: {e}")
            if verbose >= 2:
                import traceback
                traceback.print_exc()
            
            # Use canonical model name (matching get_name()) for consistency
            _canonical_names = {
                'GeoFiLMSTGATWrapper': 'Geo-FiLM ST-GAT',
                'StandardSTGATWrapper': 'Standard ST-GAT',
                'STGCNWrapper': 'ST-GCN',
                'DCRNNWrapper': 'DCRNN',
                'PINNPermafrostWrapper': 'PINN-Permafrost',
                'StefanOnlyWrapper': 'Stefan-Only',
                'RandomForestBaseline': 'Random Forest',
                'GBRTBaseline': 'XGBoost',
            }
            failed_name = _canonical_names.get(ModelClass.__name__, ModelClass.__name__)
            
            error_dict = {
                'model': failed_name,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': 0.0,
                'phys_violation_rate': 100.0,
                'film_improvement': '-',
                'train_time_sec': 0.0,
                'eval_time_sec': 0.0,
                'status': 'failed',
                'error': str(e),
            }
            all_results[failed_name] = error_dict
            all_metrics_list.append(error_dict)
    
    # Compute FiLM improvement
    geo_film_name = "Geo-FiLM ST-GAT"
    standard_name = "Standard ST-GAT"
    
    if geo_film_name in all_results and standard_name in all_results:
        geo_rmse = all_results[geo_film_name].get('rmse', float('inf'))
        std_rmse = all_results[standard_name].get('rmse', float('inf'))
        if std_rmse > 0:
            film_improvement = (std_rmse - geo_rmse) / std_rmse * 100
        else:
            film_improvement = 0.0
        all_results[geo_film_name]['film_improvement'] = film_improvement
        if verbose >= 1:
            print(f"\n[FiLM Improvement] Geo-FiLM vs Standard ST-GAT: "
                  f"{film_improvement:.1f}% RMSE reduction")
    
    # Format and print results table
    if verbose >= 1:
        print(f"\n{'='*60}")
        print("  Results Summary")
        print(f"{'='*60}")
        txt_table = format_results_txt(all_metrics_list)
        print(txt_table)
    
    # Save results: txt + csv format (replacing JSON + Markdown)
    txt_path = os.path.join(output_dir, 'experiment_results.txt')
    txt_table = format_results_txt(all_metrics_list)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Retrospective Prediction Experiment Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(txt_table)
        f.write("\n\n")
        if geo_film_name in all_results:
            fi = all_results[geo_film_name].get('film_improvement', 0)
            f.write(f"FiLM Improvement: Geo-FiLM ST-GAT achieves {fi:.1f}% "
                    f"RMSE reduction compared to Standard ST-GAT.\n")
    print(f"[Results] Plain text results saved to {txt_path}")
    
    # Save CSV results
    csv_path = os.path.join(output_dir, 'experiment_results.csv')
    csv_text = format_results_csv(all_metrics_list)
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_text)
    print(f"[Results] CSV results saved to {csv_path}")
    
    # Save diagnostics
    diag_path = os.path.join(output_dir, 'diagnostics.txt')
    with open(diag_path, 'w', encoding='utf-8') as f:
        f.write("Per-Model Diagnostic Information\n")
        f.write("=" * 70 + "\n\n")
        for name, m in all_results.items():
            f.write(f"Model: {name}\n")
            f.write("-" * 40 + "\n")
            status = m.get('status', 'ok')
            f.write(f"  Status: {status}\n")
            if status == 'failed':
                f.write(f"  Error: {m.get('error', 'unknown')}\n")
            else:
                for k, v in m.items():
                    if k in ('y_pred', 'y_true'):
                        continue  # skip large arrays
                    if isinstance(v, np.ndarray):
                        f.write(f"  {k}: shape={v.shape}, dtype={v.dtype}\n")
                    elif isinstance(v, (float, int)):
                        f.write(f"  {k}: {v}\n")
                    elif isinstance(v, str):
                        f.write(f"  {k}: {v}\n")
            f.write("\n")
        # Device info
        try:
            import torch
            f.write("\nDevice Information\n")
            f.write("-" * 40 + "\n")
            f.write(f"  PyTorch version: {torch.__version__}\n")
            f.write(f"  CUDA available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"  CUDA device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        except ImportError:
            f.write("  PyTorch not available\n")
        f.write(f"\n  Config device: {config.get('device', 'unknown')}\n")
        f.write(f"  Config epochs: {config.get('epochs', 'unknown')}\n")
        f.write(f"  Config lr: {config.get('lr', 'unknown')}\n")
        f.write(f"  Config batch_size: {config.get('batch_size', 'unknown')}\n")
    print(f"[Results] Diagnostics saved to {diag_path}")
    
    return all_results
