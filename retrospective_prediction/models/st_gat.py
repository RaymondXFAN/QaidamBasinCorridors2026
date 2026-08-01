#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard ST-GAT: Spatio-Temporal Graph Attention Network (no FiLM conditioning).

This is the ablation counterpart to Geo-FiLM ST-GAT — identical architecture
but without FiLMGenerator and without FiLM modulation on GAT layers.

Architecture:
  1. Input projection: F → hidden_dim
  2. Spatial GAT layers (no FiLM conditioning) × num_layers
  3. Temporal GRU for temporal dynamics
  4. Output head: hidden_dim → 1 (settlement prediction)
  5. Physical constraint loss (same Stefan-based loss as Geo-FiLM)

Key difference from Geo-FiLM ST-GAT:
  - NO FiLMGenerator (gamma=None, beta=None in all GAT layers)
  - GAT layers use use_film=False → purely attention-based aggregation
  - Everything else identical (GRU, output head, physical loss, training loop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class StandardGATLayer(nn.Module):
    """
    Standard Graph Attention layer (NO FiLM conditioning).
    
    h'_i = Σ_j α_ij · W · h_j   (pure attention aggregation, no modulation)
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
        
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_alpha = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(out_dim, out_dim)
    
    def forward(self, h: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:          (batch, N, in_dim) — node features
            adj_matrix: (N, N) — adjacency matrix
        Returns:
            h_out: (batch, N, out_dim) — attention-aggregated features
        """
        batch_size, N, _ = h.shape
        
        src = self.W_src(h).view(batch_size, N, self.num_heads, self.head_dim)
        dst = self.W_dst(h).view(batch_size, N, self.num_heads, self.head_dim)
        
        concat = torch.cat([src.unsqueeze(2).expand(-1, -1, N, -1, -1),
                            dst.unsqueeze(1).expand(-1, N, -1, -1, -1)],
                           dim=-1)
        
        e = (concat * self.attn_alpha.unsqueeze(0)).sum(-1)
        e = self.leaky_relu(e)
        
        mask = adj_matrix.unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        alpha = F.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        
        attn_out = (alpha.unsqueeze(-1) * src.unsqueeze(2)).sum(2)
        attn_out = attn_out.view(batch_size, N, self.out_dim)
        
        h_out = self.out_proj(attn_out)
        return h_out


class StandardSTGATNet(nn.Module):
    """
    Full Standard ST-GAT network (no FiLM).
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
        self.num_heads = int(cfg.get('num_heads', 4))
        self.num_spatial_layers = int(cfg.get('num_spatial_layers', 2))
        self.dropout = float(cfg.get('dropout', 0.1))
        self.output_dim = int(cfg.get('output_dim', 1))
        self.lambda_phys = float(cfg.get('lambda_phys', 0.3))
        self.use_phys_loss = bool(cfg.get('use_phys_loss', True))
        
        self.input_dim = None
        self.geo_dim = None
        self._built = False
        self._config = cfg
    
    def _build_model(self, input_dim: int, geo_dim: int):
        """Build model layers after knowing actual data dimensions."""
        self.input_dim = input_dim
        self.geo_dim = geo_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Spatial GAT layers (NO FiLM)
        self.spatial_layers = nn.ModuleList()
        for i in range(self.num_spatial_layers):
            self.spatial_layers.append(StandardGATLayer(
                in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ))
        
        # Temporal GRU
        self.temporal_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
        self._built = True
    
    def compute_stefan_alt(self, geo_features: torch.Tensor) -> torch.Tensor:
        """Stefan equation: ALT = sqrt(2 * k_thaw * I_t * 86400 / L)"""
        I_t = geo_features[:, 2]
        k_thaw = 1.2
        L = 3.34e7
        I_t_sec = I_t * 86400.0
        alt = torch.sqrt(2.0 * k_thaw * I_t_sec / L + 1e-8)
        return alt
    
    def compute_physical_loss(self, y_pred, y_true, geo_features):
        """Physical constraint loss: Σ max(0, |Δz_pred| - k_s · ALT)"""
        if not self.use_phys_loss:
            return torch.tensor(0.0, device=y_pred.device)
        
        k_s = geo_features[:, 1].unsqueeze(0)
        alt = self.compute_stefan_alt(geo_features).unsqueeze(0)
        delta_z = torch.abs(y_pred.squeeze(-1))
        violation = delta_z - k_s * alt
        L_phys = torch.mean(F.relu(violation))
        return L_phys
    
    def forward(self, X, adj_matrix, geo_features):
        """
        Args:
            X:            (batch, N, T, F)
            adj_matrix:   (N, N)
            geo_features: (N, G) — used ONLY for physical loss, not for FiLM
        Returns:
            y_pred: (batch, N, 1)
        """
        if not self._built:
            raise RuntimeError("Model not built. Call fit() first.")
        
        batch_size, N, T, F_dim = X.shape
        
        temporal_outputs = []
        for t in range(T):
            x_t = X[:, :, t, :]
            h = self.input_proj(x_t)
            for gat_layer in self.spatial_layers:
                h = gat_layer(h, adj_matrix)  # NO gamma, beta
                h = F.relu(h)
            temporal_outputs.append(h)
        
        temporal_seq = torch.stack(temporal_outputs, dim=2)
        temporal_seq = temporal_seq.reshape(batch_size * N, T, self.hidden_dim)
        
        gru_out, _ = self.temporal_gru(temporal_seq)
        gru_last = gru_out[:, -1, :].reshape(batch_size, N, self.hidden_dim)
        
        y_pred = self.output_head(gru_last)
        return y_pred


class StandardSTGATWrapper:
    """
    Wrapper conforming to unified BaseModel interface.
    Identical to GeoFiLMSTGATWrapper but with use_film=False (no FiLM conditioning).
    
    ⚡ Uses GRAPH-FORMAT data: (W, N, T_in, F) where W=time windows, N=real graph nodes.
    This is CRITICAL — GNN models must process real graph topology (N×N adj_matrix),
    NOT flattened N*W samples which would create OOM-sized attention matrices.
    
    data_format = 'graph' → trainer will pass graph-format data to this model.
    """
    data_format = 'graph'  # tell trainer which data format to use
    
    def __init__(self, config: dict):
        self.config = self._sanitize_config(config)
        # Explicitly set use_film=False for Standard ST-GAT
        self.config['use_film'] = False
        self.model = None
        self.device = None
        self.is_fitted = False
    
    @staticmethod
    def _sanitize_config(cfg: dict) -> dict:
        """Type-safe config: convert all YAML numeric strings to float/int."""
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
        return "Standard ST-GAT"
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            adj_matrix: np.ndarray, geo_features: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs):
        """
        Train the model with GRAPH-FORMAT data.
        
        Args:
            X_train:      (W_train, N, T_in, F) — graph format, each window has all N nodes
            y_train:      (W_train, N, T_out) — graph format target
            adj_matrix:   (N, N) — real graph adjacency (NOT expanded to N*W)
            geo_features: (N, G) — geological priors per real node
            X_val:        (W_val, N, T_in, F) optional validation data (graph format)
            y_val:        (W_val, N, T_out) optional validation target (graph format)
        """
        # ===== CRITICAL: dynamic input_dim from data =====
        W_train, N, T_in, F_dim = X_train.shape  # (W, N=40, T_in=24, F=12)
        input_dim = F_dim
        geo_dim = geo_features.shape[1]
        
        self.config['input_dim'] = input_dim
        self.config['geo_dim'] = geo_dim
        
        print(f"[Standard ST-GAT] Graph-format data: W={W_train}, N={N}, "
              f"T_in={T_in}, F={F_dim}")
        print(f"[Standard ST-GAT] adj_matrix={adj_matrix.shape}, "
              f"geo_features={geo_features.shape}")
        print(f"[Standard ST-GAT] ⚡ GAT attention: {N}×{N} nodes "
              f"(NOT {W_train*N}×{W_train*N} — OOM avoided!)")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Standard ST-GAT] Using device: {self.device}")
        
        self.model = StandardSTGATNet(self.config)
        self.model._build_model(input_dim, geo_dim)
        self.model = self.model.to(self.device)
        
        # Convert numpy to torch tensors — all in graph format
        X_t = torch.FloatTensor(X_train).to(self.device)    # (W, N, T_in, F)
        y_t = torch.FloatTensor(y_train).to(self.device)    # (W, N, T_out)
        adj_t = torch.FloatTensor(adj_matrix).to(self.device)   # (N, N)
        geo_t = torch.FloatTensor(geo_features).to(self.device) # (N, G)
        
        # Ensure y shape: (W, N, T_out) → (W, N, 1)
        if y_t.dim() == 3 and y_t.shape[2] > 1:
            y_t = y_t[:, :, -1:]  # take last step for single-step prediction
        elif y_t.dim() == 3 and y_t.shape[2] == 1:
            pass  # already correct
        
        # Training parameters
        epochs = int(self.config.get('epochs', 200))
        lr = float(self.config.get('lr', 0.001))
        batch_size = int(self.config.get('batch_size', 16))  # mini-batch over W windows
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop — mini-batch over W (time windows), NOT over N*W
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffle windows for mini-batch training
            perm = torch.randperm(W_train)
            
            for start in range(0, W_train, batch_size):
                idx = perm[start:start + batch_size]
                X_batch = X_t[idx]  # (batch_W, N, T_in, F) — each window has ALL N nodes!
                y_batch = y_t[idx]  # (batch_W, N, 1)
                
                optimizer.zero_grad()
                
                # Forward pass: model expects (batch, N, T, F)
                y_pred = self.model(X_batch, adj_t, geo_t)  # (batch_W, N, 1)
                
                # Prediction loss
                L_pred = F.mse_loss(y_pred, y_batch)
                
                # Physical constraint loss
                L_phys = self.model.compute_physical_loss(y_pred, y_batch, geo_t)
                
                # Combined loss
                lambda_phys = float(self.config.get('lambda_phys', 0.3))
                L_total = L_pred + lambda_phys * L_phys
                
                L_total.backward()
                optimizer.step()
                
                epoch_loss += L_total.item() * len(idx)
            
            scheduler.step()
            avg_loss = epoch_loss / W_train
            
            # Validation check
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._compute_val_loss(X_val, y_val, adj_t, geo_t)
            
            # Save best model
            current_metric = val_loss if val_loss is not None else avg_loss
            if current_metric < best_loss:
                best_loss = current_metric
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                # Compute violation rate on last batch for monitoring
                k_s = geo_t[:, 1].unsqueeze(0)  # (1, N)
                alt = self.model.compute_stefan_alt(geo_t).unsqueeze(0)  # (1, N)
                delta_z = torch.abs(y_pred.squeeze(-1))  # (batch_W, N)
                viol = (delta_z > k_s * alt).float().mean().item() * 100
                msg = f"Epoch {epoch+1}/{epochs} | avg_loss={avg_loss:.4f}"
                msg += f" | PhysViol={viol:.1f}%"
                if val_loss is not None:
                    msg += f" | Val={val_loss:.4f}"
                print(msg)
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        self.is_fitted = True
        print(f"[Standard ST-GAT] Training complete. Best loss: {best_loss:.4f}")
    
    def _compute_val_loss(self, X_val, y_val, adj_t, geo_t):
        """Compute validation loss. X_val/y_val in graph format."""
        X_v = torch.FloatTensor(X_val).to(self.device)  # (W_val, N, T_in, F)
        y_v = torch.FloatTensor(y_val).to(self.device)  # (W_val, N, T_out)
        if y_v.dim() == 3 and y_v.shape[2] > 1:
            y_v = y_v[:, :, -1:]
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_v, adj_t, geo_t)  # (W_val, N, 1)
            L_pred = F.mse_loss(y_pred, y_v)
            L_phys = self.model.compute_physical_loss(y_pred, y_v, geo_t)
            val_loss = L_pred.item() + float(self.config.get('lambda_phys', 0.3)) * L_phys.item()
        return val_loss
    
    def predict(self, X_test: np.ndarray, adj_matrix: np.ndarray,
                geo_features: np.ndarray) -> np.ndarray:
        """Predict settlement for test windows. X_test in graph format (W, N, T_in, F)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_t = torch.FloatTensor(X_test).to(self.device)  # (W_test, N, T_in, F)
        adj_t = torch.FloatTensor(adj_matrix).to(self.device)   # (N, N)
        geo_t = torch.FloatTensor(geo_features).to(self.device) # (N, G)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_t, adj_t, geo_t)  # (W_test, N, 1)
        
        return y_pred.cpu().numpy()  # (W_test, N, 1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 adj_matrix: np.ndarray, geo_features: np.ndarray) -> Dict:
        """Full evaluation returning metrics dict. X_test/y_test in graph format."""
        y_pred = self.predict(X_test, adj_matrix, geo_features)  # (W_test, N, 1)
        
        # Flatten predictions and targets for metrics
        y_pred_flat = y_pred.reshape(-1)  # (W_test * N)
        # Target: (W_test, N, T_out) → take last step → flatten
        if y_test.ndim == 3 and y_test.shape[2] > 1:
            y_true = y_test[:, :, -1].reshape(-1)
        else:
            y_true = y_test.reshape(-1)
        
        rmse = np.sqrt(np.mean((y_pred_flat - y_true) ** 2))
        mae = np.mean(np.abs(y_pred_flat - y_true))
        r2 = 1 - np.sum((y_true - y_pred_flat)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
        
        # Physical violation rate: per-node check across all windows
        k_s = geo_features[:, 1]   # (N,)
        alt_vals = np.sqrt(2.0 * 1.2 * (geo_features[:, 2] * 86400.0) / 3.34e7 + 1e-8)  # (N,)
        y_pred_2d = y_pred.squeeze(-1)  # (W_test, N)
        violation_matrix = np.abs(y_pred_2d) > k_s * alt_vals  # broadcasting: (W,N) > (N,)
        phys_viol_rate = violation_matrix.mean() * 100
        
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
