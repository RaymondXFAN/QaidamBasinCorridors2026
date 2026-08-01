#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN-Permafrost: Physics-Informed Neural Network for permafrost settlement prediction.

Architecture:
  - MLP (4 layers, hidden_dim=64) from features → settlement prediction
  - Physical constraint losses:
    1. Stefan equation ALT: ALT = sqrt(2*k*I_t/L), loss = MSE(ALT_pred, ALT_stefan)
    2. Heat conduction equation residual: ∂T/∂t = k·∂²T/∂z²

  L_total = L_pred + λ_phys * L_stefan + λ_thermal * L_thermal

This is NOT a graph-based model — it's a pure MLP with physics constraints,
serving as the PINN comparison baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
# flat-format model: no graph structure needed


class PINNPermafrostNet(nn.Module):
    """
    PINN for permafrost settlement prediction.
    
    MLP backbone + Stefan equation constraint + thermal diffusion constraint.
    """
    def __init__(self, config: dict):
        super().__init__()
        cfg = {}
        for k, v in config.items():
            if isinstance(v, str):
                try: v = float(v)
                except ValueError: pass
            cfg[k] = v
        
        self.hidden_dim = int(cfg.get('hidden_dim', 64))
        self.num_layers = int(cfg.get('num_layers', 4))
        self.dropout = float(cfg.get('dropout', 0.1))
        self.output_dim = int(cfg.get('output_dim', 1))
        self.lambda_phys = float(cfg.get('lambda_phys', 0.3))
        self.lambda_thermal = float(cfg.get('lambda_thermal', 0.1))
        self.use_phys_loss = bool(cfg.get('use_phys_loss', True))
        
        self.input_dim = None
        self.geo_dim = None
        self._built = False
        self._config = cfg
    
    def _build_model(self, input_dim: int, geo_dim: int):
        """Build MLP after knowing data dimensions."""
        self.input_dim = input_dim
        self.geo_dim = geo_dim
        
        # MLP: input_dim → hidden_dim → ... → output_dim
        layers = []
        in_d = input_dim
        
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(in_d, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_d = self.hidden_dim
        
        # Final layer
        layers.append(nn.Linear(in_d, self.output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # ALT prediction head (for Stefan constraint)
        self.alt_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)  # predict ALT
        )
        
        # Temperature head (for thermal constraint)
        self.temp_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)  # predict temperature gradient
        )
        
        self._built = True
    
    def compute_stefan_alt_reference(self, geo_features):
        """
        Compute reference ALT from Stefan equation using geo_features.
        ALT = sqrt(2 * k_thaw * I_t * 86400 / L)
        
        geo_features columns: [ALT, k_s, I_t, ...]
        """
        I_t = geo_features[:, 2]
        k_thaw = 1.2
        L = 3.34e7
        I_t_sec = I_t * 86400.0
        alt_ref = torch.sqrt(2.0 * k_thaw * I_t_sec / L + 1e-8)
        return alt_ref
    
    def compute_stefan_loss(self, X_input, geo_features):
        """
        Stefan equation constraint loss:
        ALT_pred should match Stefan ALT from I_t
        
        loss = MSE(ALT_pred, ALT_stefan)
        """
        alt_pred = self.alt_head(X_input).squeeze(-1)  # (N,)
        alt_stefan = self.compute_stefan_alt_reference(geo_features)  # (N,)
        L_stefan = F.mse_loss(alt_pred, alt_stefan)
        return L_stefan
    
    def compute_thermal_loss(self, X_input, geo_features):
        """
        Heat conduction PDE residual loss:
        ∂T/∂t = k · ∂²T/∂z²
        
        Simplified: We use ground_temp (feature index 3) and seasonal_amplitude (feature index 11)
        to estimate the thermal gradient. The PDE residual is computed as:
        
        residual = |∂T/∂t - k_thermal * ∂²T/∂z²|
        
        Since we don't have spatial z-coordinates, we approximate:
        ∂²T/∂z² ≈ seasonal_amplitude / ALT² (thermal gradient variation with depth)
        ∂T/∂t ≈ deformation_velocity (proxy for temporal temperature change)
        
        loss = mean(residual²)
        """
        # Predict thermal gradient
        temp_grad_pred = self.temp_head(X_input).squeeze(-1)  # (N,)
        
        # Reference from Stefan ALT
        alt_stefan = self.compute_stefan_alt_reference(geo_features)  # (N,)
        
        # Approximate PDE residual
        # k_thermal: thermal conductivity (W/m·K)
        k_thermal = 1.2
        
        # ∂²T/∂z² approximation: use seasonal amplitude / ALT²
        if X_input.shape[-1] >= 12:
            seasonal_amp = X_input[:, 11]  # seasonal_amplitude
        else:
            seasonal_amp = torch.ones(X_input.shape[0], device=X_input.device)
        
        thermal_gradient_sq = seasonal_amp / (alt_stefan ** 2 + 1e-8)
        
        # PDE residual: |temp_grad_pred - k_thermal * thermal_gradient_sq|
        residual = temp_grad_pred - k_thermal * thermal_gradient_sq
        L_thermal = torch.mean(residual ** 2)
        
        return L_thermal
    
    def compute_settlement_phys_loss(self, y_pred, y_true, geo_features):
        """
        Physical bound: |y_pred| ≤ k_s * ALT
        Hinge loss: max(0, |y_pred| - k_s * ALT)
        """
        k_s = geo_features[:, 1]
        alt = self.compute_stefan_alt_reference(geo_features)
        delta_z = torch.abs(y_pred.squeeze(-1))
        violation = delta_z - k_s * alt
        return torch.mean(F.relu(violation))
    
    def forward(self, X, geo_features):
        """
        Args:
            X:            (N, T, F) — flatten temporal features per node
            geo_features: (N, G) — geological priors
        Returns:
            y_pred: (N, 1) — settlement prediction
        """
        if not self._built:
            raise RuntimeError("Model not built. Call fit() first.")
        
        # Flatten temporal dimension: (N, T, F) → (N, T*F) or use last timestep
        if X.dim() == 3:
            # Use mean across time for stability
            x_flat = X.mean(dim=1)  # (N, F)
        elif X.dim() == 2:
            x_flat = X
        else:
            x_flat = X.reshape(-1, self.input_dim)
        
        y_pred = self.mlp(x_flat)  # (N, 1)
        return y_pred


class PINNPermafrostWrapper:
    """Wrapper conforming to unified BaseModel interface."""
    data_format = 'flat'  # flat format: no graph structure needed

    def __init__(self, config: dict):
        self.config = self._sanitize_config(config)
        self.model = None
        self.device = None
        self.is_fitted = False
    
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
        return "PINN-Permafrost"
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            adj_matrix: np.ndarray, geo_features: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs):
        """Train the PINN model."""
        # Flatten temporal dimension for MLP: (N, T, F) → use mean across T → (N, F)
        input_dim = X_train.shape[2]  # F features per timestep
        
        # Flat format: tile geo_features from (N, G) to (N*W, G)
        W = X_train.shape[0] // geo_features.shape[0]
        geo_train = np.repeat(geo_features, W, axis=0)  # (N*W, G)
        geo_dim = geo_features.shape[1]
        
        self.config['input_dim'] = input_dim
        self.config['geo_dim'] = geo_dim
        
        print(f"[PINN-Permafrost] Data dimensions: input_dim={input_dim}, "
              f"geo_dim={geo_dim}, N_train={X_train.shape[0]}, T={X_train.shape[1]}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PINN-Permafrost] Using device: {self.device}")
        
        self.model = PINNPermafrostNet(self.config)
        self.model._build_model(input_dim, geo_dim)
        self.model = self.model.to(self.device)
        
        # Flatten input for MLP: mean across time
        X_flat = torch.FloatTensor(X_train.mean(axis=1)).to(self.device)  # (N, F)
        y_t = torch.FloatTensor(y_train).to(self.device)
        
        if y_t.dim() == 2 and y_t.shape[1] > 1:
            y_t = y_t[:, -1:]
        elif y_t.dim() == 1:
            y_t = y_t.unsqueeze(-1)
        
        geo_t = torch.FloatTensor(geo_train).to(self.device)
        
        epochs = int(self.config.get('epochs', 200))
        lr = float(self.config.get('lr', 0.001))
        lambda_phys = float(self.config.get('lambda_phys', 0.3))
        lambda_thermal = float(self.config.get('lambda_thermal', 0.1))
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            y_pred = self.model(X_flat, geo_t)  # (N, 1)
            
            # Prediction loss
            L_pred = F.mse_loss(y_pred, y_t)
            
            # Stefan constraint loss
            L_stefan = self.model.compute_stefan_loss(X_flat, geo_t)
            
            # Thermal PDE loss
            L_thermal = self.model.compute_thermal_loss(X_flat, geo_t)
            
            # Physical bound loss
            L_phys_bound = self.model.compute_settlement_phys_loss(y_pred, y_t, geo_t)
            
            # Combined loss
            L_total = L_pred + lambda_phys * L_stefan + lambda_thermal * L_thermal + lambda_phys * L_phys_bound
            
            L_total.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss = L_total.item()
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                X_v_flat = torch.FloatTensor(X_val.mean(axis=1)).to(self.device)
                y_v = torch.FloatTensor(y_val).to(self.device)
                if y_v.dim() == 2 and y_v.shape[1] > 1:
                    y_v = y_v[:, -1:]
                elif y_v.dim() == 1:
                    y_v = y_v.unsqueeze(-1)
                
                self.model.eval()
                with torch.no_grad():
                    y_pred_v = self.model(X_v_flat, geo_t)
                    L_pred_v = F.mse_loss(y_pred_v, y_v).item()
                    val_loss = L_pred_v
            
            current_metric = val_loss if val_loss is not None else total_loss
            if current_metric < best_loss:
                best_loss = current_metric
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if (epoch + 1) % 50 == 0:
                k_s = geo_t[:, 1]
                alt = self.model.compute_stefan_alt_reference(geo_t)
                delta_z = torch.abs(y_pred.squeeze(-1))
                viol = (delta_z > k_s * alt).float().mean().item() * 100
                msg = f"Epoch {epoch+1}/{epochs} | L_pred={L_pred.item():.4f} "
                msg += f"L_stefan={L_stefan.item():.4f} L_thermal={L_thermal.item():.4f}"
                msg += f" L_total={total_loss:.4f} | PhysViol={viol:.1f}%"
                if val_loss is not None:
                    msg += f" | Val={val_loss:.4f}"
                print(msg)
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        self.is_fitted = True
        print(f"[PINN-Permafrost] Training complete. Best loss: {best_loss:.4f}")
    
    def predict(self, X_test: np.ndarray, adj_matrix: np.ndarray,
                geo_features: np.ndarray) -> np.ndarray:
        """Predict settlement for test nodes."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Flat format: tile geo_features from (N, G) to (N_flat, G)
        W = X_test.shape[0] // geo_features.shape[0]
        geo_test = np.repeat(geo_features, W, axis=0)  # (N_flat, G)
        X_flat = torch.FloatTensor(X_test.mean(axis=1)).to(self.device)
        geo_t = torch.FloatTensor(geo_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_flat, geo_t)
        
        return y_pred.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 adj_matrix: np.ndarray, geo_features: np.ndarray) -> Dict:
        """Full evaluation returning metrics dict."""
        # Flat format: tile geo_features from (N, G) to (N_flat, G)
        W = X_test.shape[0] // geo_features.shape[0]
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
