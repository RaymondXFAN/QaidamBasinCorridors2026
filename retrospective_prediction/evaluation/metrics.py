#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for permafrost settlement prediction.

Metrics:
  - RMSE (m): Root Mean Square Error
  - MAE (m): Mean Absolute Error
  - R²: Coefficient of Determination
  - Physical Violation Rate (%): |y_pred| > k_s * ALT
  - FiLM Improvement (%): Geo-FiLM vs Standard ST-GAT RMSE reduction
  - format_results_table(): Markdown table output
"""

import numpy as np
from typing import Dict, List


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Square Error."""
    y_pred_flat = np.asarray(y_pred).flatten()
    y_true_flat = np.asarray(y_true).flatten()
    return float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2)))


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_pred_flat = np.asarray(y_pred).flatten()
    y_true_flat = np.asarray(y_true).flatten()
    return float(np.mean(np.abs(y_pred_flat - y_true_flat)))


def compute_r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    y_pred_flat = np.asarray(y_pred).flatten()
    y_true_flat = np.asarray(y_true).flatten()
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_phys_violation_rate(y_pred: np.ndarray, geo_features: np.ndarray) -> float:
    """
    Physical constraint violation rate.
    
    Definition: percentage of predictions where |y_pred| > k_s * ALT
    
    Args:
        y_pred:        (N,) predicted settlement values
        geo_features:  (N, G) geological features
                        columns: [ALT, k_s, I_t, ...]
    Returns:
        violation_rate: percentage (0-100)
    """
    y_pred_flat = np.asarray(y_pred).flatten()
    k_s = geo_features[:, 1]  # soil-structure coefficient
    
    # Stefan ALT from geo_features
    I_t = geo_features[:, 2]  # thawing index (°C·day)
    k_thaw = 1.2  # W/(m·K)
    L = 3.34e7    # J/m³
    I_t_sec = I_t * 86400.0
    alt = np.sqrt(2.0 * k_thaw * I_t_sec / L + 1e-8)
    
    violation = np.abs(y_pred_flat) > k_s * alt
    violation_rate = violation.mean() * 100
    
    return float(violation_rate)


def compute_film_improvement(geo_film_metrics: Dict, standard_metrics: Dict) -> float:
    """
    Compute FiLM improvement percentage.
    
    FiLM Improvement = (RMSE_standard - RMSE_geo_film) / RMSE_standard × 100
    
    A positive value means Geo-FiLM is better (lower RMSE).
    
    Args:
        geo_film_metrics:  metrics dict from Geo-FiLM ST-GAT
        standard_metrics:  metrics dict from Standard ST-GAT
    Returns:
        improvement: percentage (positive = Geo-FiLM better)
    """
    geo_rmse = geo_film_metrics.get('rmse', float('inf'))
    std_rmse = standard_metrics.get('rmse', float('inf'))
    
    if std_rmse <= 0 or std_rmse == float('inf'):
        return 0.0
    
    improvement = (std_rmse - geo_rmse) / std_rmse * 100
    return float(improvement)


def format_results_txt(all_metrics) -> str:
    """
    Format experiment results as a plain text table (NOT markdown).
    
    Args:
        all_metrics: dict {model_name: metrics_dict} or list of metrics dicts
    
    Returns:
        text_table: string with formatted plain text table
    """
    # Normalize input: accept dict or list
    if isinstance(all_metrics, dict):
        metrics_list = list(all_metrics.values())
    else:
        metrics_list = list(all_metrics)
    
    # Filter out non-dict entries (e.g., metadata or error strings)
    metrics_list = [m for m in metrics_list if isinstance(m, dict)]
    
    # Column widths
    model_w = 22
    rmse_w = 10
    mae_w = 10
    r2_w = 8
    viol_w = 14
    film_w = 14
    status_w = 10
    
    # Table header
    header = f"{'Model':<{model_w}} {'RMSE(m)':<{rmse_w}} {'MAE(m)':<{mae_w}} {'R2':<{r2_w}} {'PhysViol(%)':<{viol_w}} {'FiLMImp(%)':<{film_w}} {'Status':<{status_w}}"
    separator = "-" * (model_w + rmse_w + mae_w + r2_w + viol_w + film_w + status_w + 6)
    
    # Table rows
    rows = []
    for m in metrics_list:
        model_name = m.get('model', 'Unknown')
        rmse = m.get('rmse', float('inf'))
        mae = m.get('mae', float('inf'))
        r2 = m.get('r2', 0.0)
        phys_viol = m.get('phys_violation_rate', 0.0)
        film_imp = m.get('film_improvement', '-')
        status = m.get('status', 'ok')
        
        if status == 'failed':
            row = f"{'FAILED':<{model_w}} {'FAILED':<{rmse_w}} {'FAILED':<{mae_w}} {'FAILED':<{r2_w}} {'FAILED':<{viol_w}} {'FAILED':<{film_w}} {'failed':<{status_w}}"
            rows.append(row)
            continue
        
        # Format numbers
        rmse_str = f"{rmse:.4f}" if rmse != float('inf') else "N/A"
        mae_str = f"{mae:.4f}" if mae != float('inf') else "N/A"
        r2_str = f"{r2:.4f}" if r2 != 0.0 else "N/A"
        viol_str = f"{phys_viol:.1f}"
        film_str = f"{film_imp:.1f}" if isinstance(film_imp, (int, float)) else "-"
        status_str = status
        
        row = f"{model_name:<{model_w}} {rmse_str:<{rmse_w}} {mae_str:<{mae_w}} {r2_str:<{r2_w}} {viol_str:<{viol_w}} {film_str:<{film_w}} {status_str:<{status_w}}"
        rows.append(row)
    
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    return table


def format_results_csv(all_metrics) -> str:
    """
    Format experiment results as CSV.
    
    Columns: model, rmse, mae, r2, phys_violation_rate, film_improvement,
             train_time_sec, eval_time_sec, status
    
    Args:
        all_metrics: dict {model_name: metrics_dict} or list of metrics dicts
    
    Returns:
        csv_text: string with CSV-formatted results
    """
    # Normalize input: accept dict or list
    if isinstance(all_metrics, dict):
        metrics_list = list(all_metrics.values())
    else:
        metrics_list = list(all_metrics)
    
    # Filter out non-dict entries
    metrics_list = [m for m in metrics_list if isinstance(m, dict)]
    
    # CSV header
    header = "model,rmse,mae,r2,phys_violation_rate,film_improvement,train_time_sec,eval_time_sec,status"
    
    rows = []
    for m in metrics_list:
        model_name = m.get('model', 'Unknown')
        rmse = m.get('rmse', float('inf'))
        mae = m.get('mae', float('inf'))
        r2 = m.get('r2', 0.0)
        phys_viol = m.get('phys_violation_rate', 0.0)
        film_imp = m.get('film_improvement', '-')
        train_time = m.get('train_time_sec', 0.0)
        eval_time = m.get('eval_time_sec', 0.0)
        status = m.get('status', 'ok')
        
        if status == 'failed':
            rmse_str = "inf"
            mae_str = "inf"
            r2_str = "0.0"
            viol_str = "100.0"
            film_str = "-"
        else:
            rmse_str = f"{rmse:.4f}" if rmse != float('inf') else "inf"
            mae_str = f"{mae:.4f}" if mae != float('inf') else "inf"
            r2_str = f"{r2:.4f}"
            viol_str = f"{phys_viol:.1f}"
            film_str = f"{film_imp:.1f}" if isinstance(film_imp, (int, float)) else "-"
        
        row = f"{model_name},{rmse_str},{mae_str},{r2_str},{viol_str},{film_str},{train_time:.1f},{eval_time:.1f},{status}"
        rows.append(row)
    
    csv_text = header + "\n" + "\n".join(rows)
    return csv_text


def format_results_table(all_metrics) -> str:
    """
    Format experiment results as a Markdown table.
    
    Args:
        all_metrics: dict {model_name: metrics_dict} or list of metrics dicts
    
    Returns:
        markdown_table: string with formatted table
    """
    # Normalize input: accept dict or list
    if isinstance(all_metrics, dict):
        metrics_list = list(all_metrics.values())
    else:
        metrics_list = list(all_metrics)
    
    # Filter out non-dict entries (e.g., metadata or error strings)
    metrics_list = [m for m in metrics_list if isinstance(m, dict)]
    
    # Table header
    header = "| Model | RMSE (m) | MAE (m) | R² | Phys.Viol. (%) | FiLM Imp. (%) |"
    separator = "|-------|----------|---------|-----|----------------|---------------|"
    
    # Table rows
    rows = []
    for m in metrics_list:
        model_name = m.get('model', 'Unknown')
        rmse = m.get('rmse', float('inf'))
        mae = m.get('mae', float('inf'))
        r2 = m.get('r2', 0.0)
        phys_viol = m.get('phys_violation_rate', 0.0)
        film_imp = m.get('film_improvement', '-')
        status = m.get('status', 'ok')
        
        if status == 'failed':
            error_msg = m.get('error', 'unknown error')
            row = f"| {model_name} | FAILED | FAILED | FAILED | FAILED | FAILED |"
            rows.append(row)
            continue
        
        # Format numbers
        rmse_str = f"{rmse:.4f}" if rmse != float('inf') else "N/A"
        mae_str = f"{mae:.4f}" if mae != float('inf') else "N/A"
        r2_str = f"{r2:.4f}" if r2 != 0.0 else "N/A"
        viol_str = f"{phys_viol:.1f}"
        film_str = f"{film_imp:.1f}" if isinstance(film_imp, (int, float)) else "-"
        
        row = f"| {model_name} | {rmse_str} | {mae_str} | {r2_str} | {viol_str} | {film_str} |"
        rows.append(row)
    
    table = header + "\n" + separator + "\n" + "\n".join(rows)
    return table
