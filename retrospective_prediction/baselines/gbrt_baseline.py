#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost/GBRT baseline for settlement prediction.

Uses xgboost.XGBRegressor (fallback to sklearn GradientBoostingRegressor
if xgboost is not available):
  - Input: flatten (N, T, F) → (N, T*F)
  - Output: settlement prediction (N, 1)
  - n_estimators=200, max_depth=6, learning_rate=0.1

Conforms to unified BaseModel interface.
"""

import numpy as np
from typing import Dict, Optional

try:
    import xgboost as xgb
    USE_XGBOOST = True
except ImportError:
    USE_XGBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor
    print("[GBRT] xgboost not available, falling back to sklearn GradientBoostingRegressor")

# flat-format model: no graph structure needed


class GBRTBaseline:
    """XGBoost/GBRT baseline conforming to unified BaseModel interface."""
    data_format = 'flat'  # flat format: no graph structure needed

    def __init__(self, config: dict):
        self.config = self._sanitize_config(config)
        self.model = None
        self.is_fitted = False
        self.use_xgboost = USE_XGBOOST
        
        # Hyperparameters
        self.n_estimators = int(self.config.get('n_estimators', 200))
        self.max_depth = int(self.config.get('max_depth', 6))
        self.learning_rate = float(self.config.get('learning_rate', 0.1))
        self.random_state = int(self.config.get('random_state', 42))
    
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
        return "XGBoost" if self.use_xgboost else "GBRT"
    
    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        """Flatten temporal dimension: (N, T, F) → (N, T*F)."""
        if X.ndim == 3:
            N, T, F = X.shape
            print(f"[{self.get_name()}] Flattening input: ({N}, {T}, {F}) → ({N}, {T*F})")
            return X.reshape(N, T * F)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")
    
    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Prepare target: take last timestep if multi-step."""
        if y.ndim == 2 and y.shape[1] > 1:
            return y[:, -1]
        else:
            return y.squeeze()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            adj_matrix: np.ndarray, geo_features: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs):
        """Train XGBoost/GBRT model."""
        X_flat = self._flatten_X(X_train)
        y_flat = self._prepare_y(y_train)
        
        print(f"[{self.get_name()}] Training with n_estimators={self.n_estimators}, "
              f"max_depth={self.max_depth}, learning_rate={self.learning_rate}")
        print(f"[{self.get_name()}] Input shape: {X_flat.shape}, Target shape: {y_flat.shape}")
        
        if self.use_xgboost:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        
        self.model.fit(X_flat, y_flat)
        self.is_fitted = True
        
        train_r2 = self.model.score(X_flat, y_flat)
        print(f"[{self.get_name()}] Training R²: {train_r2:.4f}")
    
    def predict(self, X_test: np.ndarray, adj_matrix: np.ndarray,
                geo_features: np.ndarray) -> np.ndarray:
        """Predict settlement for test nodes."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_flat = self._flatten_X(X_test)
        y_pred = self.model.predict(X_flat)
        
        return y_pred.reshape(-1, 1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 adj_matrix: np.ndarray, geo_features: np.ndarray) -> Dict:
        """Full evaluation returning metrics dict."""
        # Flat format: tile geo_features from (N, G) to (N_flat, G) for phys violation check
        W = X_test.shape[0] // geo_features.shape[0] if X_test.shape[0] > geo_features.shape[0] else 1
        geo_test = np.repeat(geo_features, W, axis=0)  # (N_flat, G)
        y_pred = self.predict(X_test, adj_matrix, geo_features)
        
        if y_test.ndim == 2 and y_test.shape[1] > 1:
            y_true = y_test[:, -1]
        else:
            y_true = y_test.squeeze()
        y_pred_flat = y_pred.squeeze()
        
        rmse = np.sqrt(np.mean((y_pred_flat - y_true) ** 2))
        mae = np.mean(np.abs(y_pred_flat - y_true))
        r2 = 1 - np.sum((y_true - y_pred_flat)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
        
        # Physical violation rate
        k_s = geo_test[:, 1]
        alt_vals = np.sqrt(2.0 * 1.2 * (geo_test[:, 2] * 86400.0) / 3.34e7 + 1e-8)
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
