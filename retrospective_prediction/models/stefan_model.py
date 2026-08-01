#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stefan-Only: Pure physics model for settlement prediction.

This model does NOT require training! It directly computes settlement
from the Stefan equation using input features.

Stefan equation:
  ALT = sqrt(2 * k_thaw * I_t * 86400 / L)

Settlement prediction:
  settlement = k_s * ALT * factor

where:
  k_thaw = thermal conductivity of thawed soil (~1.2 W/m·K for Qaidam)
  I_t    = thawing index (°C·day) — from geo_features[:,2]
  L      = volumetric latent heat (~3.34×10^7 J/m³)
  k_s    = soil-structure coefficient — from geo_features[:,1]
  factor = scaling factor (default 0.5, empirical)

This serves as the "pure physics" baseline — no learning, no parameters.
"""

import numpy as np
from typing import Dict, Optional
# flat-format model: no graph structure needed


class StefanOnlyWrapper:
    """
    Pure Stefan equation model — no training required.
    
    Conforms to unified BaseModel interface, but fit() is essentially a no-op.
    """
    data_format = 'flat'  # flat format: no graph structure needed

    def __init__(self, config: dict):
        self.config = self._sanitize_config(config)
        self.is_fitted = True  # always "fitted" — pure physics
        self.device = None     # no GPU needed (pure numpy)
        
        # Stefan equation parameters
        self.k_thaw = float(self.config.get('k_thaw', 1.2))  # W/(m·K)
        self.L_latent = float(self.config.get('L_latent', 3.34e7))  # J/m³
        self.factor = float(self.config.get('stefan_factor', 0.5))  # empirical scaling
    
    @staticmethod
    def _sanitize_config(cfg: dict) -> dict:
        """Type-safe config."""
        safe = {}
        for k, v in cfg.items():
            if isinstance(v, str):
                try: v = int(v)
                except ValueError:
                    try: v = float(v)
                    except ValueError: pass
            safe[k] = v
        return safe
    
    def get_name(self) -> str:
        return "Stefan-Only"
    
    def compute_stefan_alt(self, geo_features: np.ndarray) -> np.ndarray:
        """
        Compute Stefan equation ALT.
        
        ALT = sqrt(2 * k_thaw * I_t * 86400 / L)
        
        Args:
            geo_features: (N, G) — columns: [ALT, k_s, I_t, ...]
        Returns:
            alt: (N,) — Active Layer Thickness in meters
        """
        I_t = geo_features[:, 2]  # thawing index (°C·day)
        I_t_sec = I_t * 86400.0   # convert day → seconds
        
        alt = np.sqrt(2.0 * self.k_thaw * I_t_sec / self.L_latent + 1e-8)
        return alt
    
    def compute_settlement(self, geo_features: np.ndarray) -> np.ndarray:
        """
        Compute settlement from Stefan ALT.
        
        settlement = k_s * ALT * factor
        
        Args:
            geo_features: (N, G) — columns: [ALT, k_s, I_t, ...]
        Returns:
            settlement: (N,) — predicted settlement
        """
        k_s = geo_features[:, 1]  # soil-structure coefficient
        alt = self.compute_stefan_alt(geo_features)
        
        settlement = k_s * alt * self.factor
        return settlement
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            adj_matrix: np.ndarray, geo_features: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs):
        """
        "Train" the Stefan model — essentially a no-op for pure physics.
        
        We print diagnostic info but don't learn any parameters.
        Optionally, we could calibrate the factor from training data,
        but for a pure physics baseline, we keep it fixed.
        """
        # Flat format: no graph extraction needed, just print diagnostic info
        print(f"[Stefan-Only] Pure physics model — no training required.")
        print(f"[Stefan-Only] Data dimensions: N_train={X_train.shape[0]}, "
              f"T={X_train.shape[1]}, F={X_train.shape[2]}")
        
        # Optionally calibrate factor from training data
        if self.config.get('calibrate_factor', False):
            y_true = y_train
            if y_true.ndim == 2 and y_true.shape[1] > 1:
                y_true = y_true[:, -1]
            else:
                y_true = y_true.squeeze()
            
            # Flat format: reduce y_true per-node (mean across windows)
            W = X_train.shape[0] // geo_features.shape[0]
            if W > 1:
                y_true_per_node = y_true.reshape(-1, W).mean(axis=1)  # (N,)
            else:
                y_true_per_node = y_true
            
            k_s = geo_features[:, 1]
            alt = self.compute_stefan_alt(geo_features)
            
            # Factor = y_true / (k_s * ALT)  (mean calibration)
            denominator = k_s * alt
            # Avoid division by zero
            valid_mask = denominator > 1e-8
            if valid_mask.sum() > 0:
                calibrated_factor = np.mean(y_true_per_node[valid_mask] / denominator[valid_mask])
                self.factor = float(calibrated_factor)
                print(f"[Stefan-Only] Calibrated factor: {self.factor:.4f}")
        
        print(f"[Stefan-Only] Parameters: k_thaw={self.k_thaw}, "
              f"L={self.L_latent:.2e}, factor={self.factor}")
        print(f"[Stefan-Only] Model ready for prediction.")
    
    def predict(self, X_test: np.ndarray, adj_matrix: np.ndarray,
                geo_features: np.ndarray) -> np.ndarray:
        """
        Predict settlement using Stefan equation.
        
        Returns:
            y_pred: (N, 1) — predicted settlement
        """
        # Flat format: geo_features is (N, G), use directly for Stefan equation
        settlement = self.compute_settlement(geo_features)
        return settlement.reshape(-1, 1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 adj_matrix: np.ndarray, geo_features: np.ndarray) -> Dict:
        """Full evaluation returning metrics dict."""
        # Flat format: tile geo_features from (N, G) to (N_flat, G) for phys violation check
        W = X_test.shape[0] // geo_features.shape[0] if X_test.shape[0] > geo_features.shape[0] else 1
        geo_test = np.repeat(geo_features, W, axis=0)  # (N_flat, G)
        y_pred = self.predict(X_test, adj_matrix, geo_features)
        
        # Tile y_pred to match flat format samples
        if W > 1:
            y_pred_tiled = np.repeat(y_pred, W, axis=0)
        else:
            y_pred_tiled = y_pred
        
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_true = y_test[:, -1]
        else:
            y_true = y_test.squeeze()
        y_pred_flat = y_pred_tiled.squeeze()
        
        rmse = np.sqrt(np.mean((y_pred_flat - y_true) ** 2))
        mae = np.mean(np.abs(y_pred_flat - y_true))
        r2 = 1 - np.sum((y_true - y_pred_flat)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
        
        k_s = geo_test[:, 1]
        alt_vals = self.compute_stefan_alt(geo_test)
        violation = np.abs(y_pred_flat) > k_s * alt_vals
        phys_viol_rate = violation.mean() * 100
        
        metrics = {
            'model': self.get_name(),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'phys_violation_rate': phys_viol_rate,
            'y_pred': y_pred_flat,
            'y_true': y_true
        }
        print(f"[{self.get_name()}] RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f} "
              f"PhysViol={phys_viol_rate:.1f}%")
        return metrics
