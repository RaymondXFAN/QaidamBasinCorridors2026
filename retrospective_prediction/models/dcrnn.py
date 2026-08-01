#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCRNN: Diffusion Convolutional Recurrent Neural Network.

Architecture:
  - Diffusion convolution (based on random walk: P_fwd=D^{-1}A, P_bwd=D^{-1}A^T)
  - GRU with diffusion convolution replacing matrix multiplication
  - Output head → settlement prediction

Key design:
  - Manual diffusion matrix computation (K_step=2 steps)
  - No torch_geometric dependency
  - Diffusion convolution: Σ_{k=0}^{K-1} (P_fwd^k · W_fwd_k + P_bwd^k · W_bwd_k) · x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
# extract_subgraph removed — DCRNNWrapper now uses graph format (W, N, T_in, F)


def compute_diffusion_matrices(adj_matrix: np.ndarray, K: int = 2):
    """
    Compute forward and backward random walk transition matrices up to K steps.
    
    P_fwd = D^{-1} A (forward random walk)
    P_bwd = D^{-1} A^T (backward random walk)
    
    Returns precomputed powers: P_fwd^0, P_fwd^1, ..., P_fwd^{K-1}
                             and P_bwd^0, P_bwd^1, ..., P_bwd^{K-1}
    """
    N = adj_matrix.shape[0]
    A = adj_matrix.copy().astype(np.float64)
    
    # Add self-loops for numerical stability
    A = A + np.eye(N) * 0.01
    
    # Degree for forward: D_fwd where D_fwd[i] = sum of A[i,:]
    D_fwd = np.sum(A, axis=1)
    D_fwd_inv = np.power(D_fwd, -1)
    D_fwd_inv[np.isinf(D_fwd_inv)] = 0.0
    P_fwd = D_fwd_inv.reshape(-1, 1) * A  # (N, N)
    
    # Degree for backward: D_bwd where D_bwd[i] = sum of A^T[i,:] = sum of A[:,i]
    A_T = A.T
    D_bwd = np.sum(A_T, axis=1)
    D_bwd_inv = np.power(D_bwd, -1)
    D_bwd_inv[np.isinf(D_bwd_inv)] = 0.0
    P_bwd = D_bwd_inv.reshape(-1, 1) * A_T  # (N, N)
    
    # Compute powers
    P_fwd_powers = [np.eye(N)]  # P_fwd^0 = I
    P_bwd_powers = [np.eye(N)]  # P_bwd^0 = I
    
    for k in range(1, K):
        P_fwd_powers.append(P_fwd_powers[-1] @ P_fwd)
        P_bwd_powers.append(P_bwd_powers[-1] @ P_bwd)
    
    # Convert to float32 tensors
    P_fwd_list = [torch.FloatTensor(p.astype(np.float32)) for p in P_fwd_powers]
    P_bwd_list = [torch.FloatTensor(p.astype(np.float32)) for p in P_bwd_powers]
    
    return P_fwd_list, P_bwd_list


class DiffusionConvLayer(nn.Module):
    """
    Diffusion Convolution layer.
    
    y = Σ_{k=0}^{K-1} (P_fwd^k · x · W_fwd_k + P_bwd^k · x · W_bwd_k)
    
    Each step produces a (N, out_dim) output, all summed together.
    """
    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.K = K
        self.out_dim = out_dim
        
        # Forward and backward filter weights for each step
        self.W_fwd = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_dim, out_dim))
            for _ in range(K)
        ])
        self.W_bwd = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_dim, out_dim))
            for _ in range(K)
        ])
        
        # Initialize
        for w in self.W_fwd:
            nn.init.xavier_uniform_(w)
        for w in self.W_bwd:
            nn.init.xavier_uniform_(w)
        
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x: torch.Tensor, P_fwd_list: list, P_bwd_list: list) -> torch.Tensor:
        """
        Args:
            x:           (N, in_dim) or (batch, N, in_dim) — node features
            P_fwd_list:  list of (N, N) tensors — P_fwd^0, P_fwd^1, ..., P_fwd^{K-1}
            P_bwd_list:  list of (N, N) tensors — P_bwd^0, P_bwd^1, ..., P_bwd^{K-1}
        Returns:
            out: (N, out_dim) or (batch, N, out_dim) — convolved features
        """
        if x.dim() == 2:
            return self._forward_single(x, P_fwd_list, P_bwd_list)
        elif x.dim() == 3:
            return self._forward_batched(x, P_fwd_list, P_bwd_list)
        else:
            raise ValueError(f"DiffusionConvLayer expects 2D or 3D input, got {x.dim()}D")
    
    def _forward_single(self, x, P_fwd_list, P_bwd_list):
        """Single sample: x is (N, in_dim)."""
        N = x.shape[0]
        out = torch.zeros(N, self.out_dim, device=x.device)
        for k in range(self.K):
            fwd_term = torch.matmul(P_fwd_list[k], x)  # (N, in_dim)
            fwd_term = torch.matmul(fwd_term, self.W_fwd[k])  # (N, out_dim)
            bwd_term = torch.matmul(P_bwd_list[k], x)  # (N, in_dim)
            bwd_term = torch.matmul(bwd_term, self.W_bwd[k])  # (N, out_dim)
            out = out + fwd_term + bwd_term
        out = out + self.bias
        return out
    
    def _forward_batched(self, x, P_fwd_list, P_bwd_list):
        """Batched: x is (batch, N, in_dim). Uses torch.bmm for parallel diffusion convolution."""
        batch_size, N, in_dim = x.shape
        out = torch.zeros(batch_size, N, self.out_dim, device=x.device)
        for k in range(self.K):
            P_fwd_exp = P_fwd_list[k].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N, N)
            P_bwd_exp = P_bwd_list[k].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N, N)
            fwd_term = torch.bmm(P_fwd_exp, x)  # (batch, N, in_dim)
            fwd_term = torch.matmul(fwd_term, self.W_fwd[k])  # (batch, N, out_dim)
            bwd_term = torch.bmm(P_bwd_exp, x)  # (batch, N, in_dim)
            bwd_term = torch.matmul(bwd_term, self.W_bwd[k])  # (batch, N, out_dim)
            out = out + fwd_term + bwd_term
        out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    
    Replaces matrix multiplication in standard GRU with diffusion convolution.
    Standard GRU:
      z = σ(W_z · [h_prev, x])
      r = σ(W_r · [h_prev, x])
      h_new = tanh(W_h · [r·h_prev, x])
    
    DCGRU:
      z = σ(DC([h_prev, x]))
      r = σ(DC([r·h_prev, x]))
      h_new = tanh(DC([r·h_prev, x]))
    """
    def __init__(self, input_dim: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Gates use diffusion convolution on concatenated [h, x]
        concat_dim = input_dim + hidden_dim
        
        self.gate_conv_z = DiffusionConvLayer(concat_dim, hidden_dim, K)
        self.gate_conv_r = DiffusionConvLayer(concat_dim, hidden_dim, K)
        self.update_conv = DiffusionConvLayer(concat_dim, hidden_dim, K)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor,
                P_fwd_list: list, P_bwd_list: list) -> torch.Tensor:
        """
        Args:
            x:       (N, input_dim) or (batch, N, input_dim)
            h_prev:  (N, hidden_dim) or (batch, N, hidden_dim)
            P_fwd_list, P_bwd_list: diffusion matrices
        Returns:
            h_new: (N, hidden_dim) or (batch, N, hidden_dim)
        """
        concat = torch.cat([h_prev, x], dim=-1)  # (batch, N, input_dim+hidden_dim) or (N, ...)
        
        z = torch.sigmoid(self.gate_conv_z(concat, P_fwd_list, P_bwd_list))
        r = torch.sigmoid(self.gate_conv_r(concat, P_fwd_list, P_bwd_list))
        
        concat_r = torch.cat([r * h_prev, x], dim=-1)
        h_candidate = torch.tanh(self.update_conv(concat_r, P_fwd_list, P_bwd_list))
        
        h_new = z * h_prev + (1 - z) * h_candidate
        return h_new


class DCRNNNet(nn.Module):
    """
    Full DCRNN network.
    
    Architecture:
      1. Input projection: F → hidden_dim
      2. DCGRU cells process temporal sequence (diffusion conv in GRU)
      3. Output head → settlement prediction
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
        self.K = int(cfg.get('diffusion_K', 2))
        self.dropout = float(cfg.get('dropout', 0.1))
        self.output_dim = int(cfg.get('output_dim', 1))
        self.lambda_phys = float(cfg.get('lambda_phys', 0.3))
        self.use_phys_loss = bool(cfg.get('use_phys_loss', True))
        
        self.input_dim = None
        self.geo_dim = None
        self._built = False
        self._config = cfg
        self.P_fwd_list = None
        self.P_bwd_list = None
    
    def _build_model(self, input_dim: int, geo_dim: int):
        """Build model layers after knowing actual data dimensions."""
        self.input_dim = input_dim
        self.geo_dim = geo_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # DCGRU cell
        self.dcgru = DCGRUCell(self.hidden_dim, self.hidden_dim, self.K)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
        self._built = True
    
    def compute_stefan_alt(self, geo_features):
        """Stefan equation: ALT = sqrt(2 * k_thaw * I_t * 86400 / L)"""
        I_t = geo_features[:, 2]
        k_thaw = 1.2
        L = 3.34e7
        I_t_sec = I_t * 86400.0
        alt = torch.sqrt(2.0 * k_thaw * I_t_sec / L + 1e-8)
        return alt
    
    def compute_physical_loss(self, y_pred, y_true, geo_features):
        """Physical constraint loss."""
        if not self.use_phys_loss:
            return torch.tensor(0.0, device=y_pred.device)
        k_s = geo_features[:, 1].unsqueeze(0)
        alt = self.compute_stefan_alt(geo_features).unsqueeze(0)
        delta_z = torch.abs(y_pred.squeeze(-1))
        violation = delta_z - k_s * alt
        return torch.mean(F.relu(violation))
    
    def forward(self, X, adj_matrix, geo_features):
        """
        Args:
            X:            (batch, N, T, F)
            adj_matrix:   (N, N)
            geo_features: (N, G)
        Returns:
            y_pred: (batch, N, 1)
        """
        if not self._built:
            raise RuntimeError("Model not built. Call fit() first.")
        
        batch_size, N, T, F_dim = X.shape
        
        # Initialize per-batch hidden state: (batch, N, hidden_dim)
        h = torch.zeros(batch_size, N, self.hidden_dim, device=X.device)
        
        for t in range(T):
            x_t = X[:, :, t, :]  # (batch, N, F)
            # nn.Linear on last dim: (batch, N, F) → (batch, N, hidden_dim)
            x_proj = self.input_proj(x_t)  # (batch, N, hidden_dim)
            
            # DCGRU step with batched inputs: (batch, N, dim)
            h = self.dcgru(x_proj, h, self.P_fwd_list, self.P_bwd_list)
        
        # Use final hidden state for prediction — nn.Linear on last dim
        y_pred = self.output_head(h)  # (batch, N, 1)
        return y_pred



class DCRNNWrapper:
    """
    Wrapper conforming to unified BaseModel interface.
    
    Uses GRAPH-FORMAT data: (W, N, T_in, F) where W=time windows, N=real graph nodes.
    This is CRITICAL - GNN models must process real graph topology (NxN adj_matrix),
    NOT flattened N*W samples which would create OOM-sized diffusion matrices.
    
    data_format = 'graph' -> trainer will pass graph-format data to this model.
    """
    data_format = 'graph'  # tell trainer which data format to use
    
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
        return "DCRNN"
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            adj_matrix: np.ndarray, geo_features: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs):
        """
        Train the model with GRAPH-FORMAT data.
        
        Args:
            X_train:      (W_train, N, T_in, F) - graph format, each window has all N nodes
            y_train:      (W_train, N, T_out) - graph format target
            adj_matrix:   (N, N) - real graph adjacency (NOT expanded to N*W)
            geo_features: (N, G) - geological priors per real node
            X_val:        (W_val, N, T_in, F) optional validation data (graph format)
            y_val:        (W_val, N, T_out) optional validation target (graph format)
        """
        W_train, N, T_in, F_dim = X_train.shape  # (W, N=40, T_in=24, F=12)
        input_dim = F_dim
        geo_dim = geo_features.shape[1]
        
        self.config['input_dim'] = input_dim
        self.config['geo_dim'] = geo_dim
        
        print(f"[DCRNN] Graph-format data: W={W_train}, N={N}, "
              f"T_in={T_in}, F={F_dim}")
        print(f"[DCRNN] adj_matrix={adj_matrix.shape}, "
              f"geo_features={geo_features.shape}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DCRNN] Using device: {self.device}")
        
        # Precompute diffusion matrices from ORIGINAL adjacency matrix (N, N)
        K = int(self.config.get('diffusion_K', 2))
        P_fwd_np, P_bwd_np = compute_diffusion_matrices(adj_matrix, K)
        P_fwd_list = [p.to(self.device) for p in P_fwd_np]
        P_bwd_list = [p.to(self.device) for p in P_bwd_np]
        
        self.model = DCRNNNet(self.config)
        self.model._build_model(input_dim, geo_dim)
        self.model.P_fwd_list = P_fwd_list
        self.model.P_bwd_list = P_bwd_list
        self.model = self.model.to(self.device)
        
        # Convert numpy to torch tensors - all in graph format
        X_t = torch.FloatTensor(X_train).to(self.device)    # (W, N, T_in, F)
        y_t = torch.FloatTensor(y_train).to(self.device)    # (W, N, T_out)
        adj_t = torch.FloatTensor(adj_matrix).to(self.device)   # (N, N)
        geo_t = torch.FloatTensor(geo_features).to(self.device) # (N, G)
        
        # Ensure y shape: (W, N, T_out) -> (W, N, 1)
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
        
        # Training loop - mini-batch over W (time windows), NOT over N*W
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Shuffle windows for mini-batch training
            perm = torch.randperm(W_train)
            
            for start in range(0, W_train, batch_size):
                idx = perm[start:start + batch_size]
                X_batch = X_t[idx]  # (batch_W, N, T_in, F) - each window has ALL N nodes!
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
        print(f"[DCRNN] Training complete. Best loss: {best_loss:.4f}")
    
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
        # Target: (W_test, N, T_out) -> take last step -> flatten
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
        print(f"[{self.get_name()}] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
              f"PhysViol={phys_viol_rate:.1f}%")
        return metrics
