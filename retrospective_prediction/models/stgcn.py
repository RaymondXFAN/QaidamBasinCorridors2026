#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GCN: Spatio-Temporal Graph Convolutional Network.

Architecture:
  - GCN layer (Chebyshev spectral convolution, K=3) for spatial convolution
  - 1D Convolution for temporal dimension
  - Output head → settlement prediction

Key design:
  - Hand-written ChebConv (no torch_geometric dependency for AutoDL compatibility)
  - Chebyshev polynomials: T_k(L)·x where L = I - D^{-1/2}AD^{-1/2} (normalized Laplacian)
  - K=3 order (T_0, T_1, T_2)
  - Temporal: Conv1d across time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
# extract_subgraph removed — STGCNWrapper now uses graph format (W, N, T_in, F)


class ChebConvLayer(nn.Module):
    """
    Chebyshev Spectral Graph Convolution (K-order).
    
    Computes: y = Σ_{k=0}^{K-1} θ_k · T_k(L̃) · x
    
    where L̃ = (2/λ_max) * L - I is the scaled Laplacian,
    L = I - D^{-1/2}AD^{-1/2} is the normalized Laplacian,
    T_k are Chebyshev polynomials: T_0(x)=1, T_1(x)=x, T_k(x)=2xT_{k-1}-T_{k-2}
    
    K=3 order: T_0(L̃)·x = x, T_1(L̃)·x = L̃·x, T_2(L̃)·x = 2L̃·(L̃·x) - x
    """
    def __init__(self, in_dim: int, out_dim: int, K: int = 3):
        super().__init__()
        self.K = K
        self.out_dim = out_dim
        # K filter parameters θ_0, θ_1, ..., θ_{K-1}
        self.theta = nn.Parameter(torch.FloatTensor(K, in_dim, out_dim))
        nn.init.xavier_uniform_(self.theta)
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x: torch.Tensor, L_scaled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        (N, in_dim) or (batch, N, in_dim) — node features
            L_scaled: (N, N) — scaled normalized Laplacian L̃
        Returns:
            out: (N, out_dim) or (batch, N, out_dim) — convolved features
        """
        if x.dim() == 2:
            return self._forward_single(x, L_scaled)
        elif x.dim() == 3:
            return self._forward_batched(x, L_scaled)
        else:
            raise ValueError(f"ChebConvLayer expects 2D or 3D input, got {x.dim()}D")
    
    def _forward_single(self, x: torch.Tensor, L_scaled: torch.Tensor) -> torch.Tensor:
        """Single sample: x is (N, in_dim)."""
        N = x.shape[0]
        Tx_0 = x  # (N, in_dim)
        Tx_1 = torch.matmul(L_scaled, x)  # (N, in_dim)
        cheb_polys = [Tx_0, Tx_1]
        for k in range(2, self.K):
            Tx_k = 2 * torch.matmul(L_scaled, cheb_polys[-1]) - cheb_polys[-2]
            cheb_polys.append(Tx_k)
        out = torch.zeros(N, self.out_dim, device=x.device)
        for k in range(self.K):
            out = out + torch.matmul(cheb_polys[k], self.theta[k])
        out = out + self.bias
        return out
    
    def _forward_batched(self, x: torch.Tensor, L_scaled: torch.Tensor) -> torch.Tensor:
        """Batched: x is (batch, N, in_dim). Uses torch.bmm for parallel graph convolution."""
        batch_size, N, in_dim = x.shape
        L_exp = L_scaled.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N, N)
        
        Tx_0 = x  # (batch, N, in_dim)
        Tx_1 = torch.bmm(L_exp, x)  # (batch, N, in_dim)
        cheb_polys = [Tx_0, Tx_1]
        for k in range(2, self.K):
            Tx_k = 2 * torch.bmm(L_exp, cheb_polys[-1]) - cheb_polys[-2]
            cheb_polys.append(Tx_k)
        
        out = torch.zeros(batch_size, N, self.out_dim, device=x.device)
        for k in range(self.K):
            # cheb_polys[k]: (batch, N, in_dim), theta[k]: (in_dim, out_dim)
            # torch.matmul on (batch, N, in_dim) × (in_dim, out_dim) → (batch, N, out_dim)
            out = out + torch.matmul(cheb_polys[k], self.theta[k])
        out = out + self.bias
        return out


def compute_scaled_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Compute scaled normalized Laplacian: L̃ = (2/λ_max) * L - I
    
    where L = I - D^{-1/2}AD^{-1/2} is the normalized graph Laplacian,
    λ_max is the largest eigenvalue of L (clamped to 2.0 for stability).
    """
    N = adj_matrix.shape[0]
    A = adj_matrix.copy().astype(np.float64)
    
    # Add self-loops for numerical stability
    A = A + np.eye(N) * 0.01
    
    # Degree matrix
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    
    # Normalized Laplacian: L = I - D^{-1/2}AD^{-1/2}
    L_norm = np.eye(N) - D_inv_sqrt.reshape(-1, 1) * A * D_inv_sqrt.reshape(1, -1)
    
    # Eigenvalues for scaling
    try:
        eigvals = np.linalg.eigvalsh(L_norm)
        lambda_max = max(eigvals.max(), 2.0)
    except np.linalg.LinAlgError:
        lambda_max = 2.0
    
    # Scale: L̃ = (2/λ_max) * L - I
    L_scaled = (2.0 / lambda_max) * L_norm - np.eye(N)
    return L_scaled.astype(np.float32)


class STGCNNet(nn.Module):
    """
    Full ST-GCN network.
    
    Architecture:
      1. Input projection: F → hidden_dim
      2. Spatial ChebConv layers × num_spatial_layers
      3. Temporal Conv1d for temporal dynamics
      4. Output head → settlement prediction
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
        self.num_spatial_layers = int(cfg.get('num_spatial_layers', 2))
        self.K = int(cfg.get('cheb_K', 3))
        self.dropout = float(cfg.get('dropout', 0.1))
        self.output_dim = int(cfg.get('output_dim', 1))
        self.lambda_phys = float(cfg.get('lambda_phys', 0.3))
        self.use_phys_loss = bool(cfg.get('use_phys_loss', True))
        
        self.input_dim = None
        self.geo_dim = None
        self._built = False
        self._config = cfg
        self.L_scaled = None  # precomputed scaled Laplacian
    
    def _build_model(self, input_dim: int, geo_dim: int):
        """Build model layers after knowing actual data dimensions."""
        self.input_dim = input_dim
        self.geo_dim = geo_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Spatial ChebConv layers
        self.spatial_layers = nn.ModuleList()
        for i in range(self.num_spatial_layers):
            self.spatial_layers.append(ChebConvLayer(
                in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                K=self.K
            ))
        
        # Temporal Conv1d
        # Input: (batch*N, T, hidden_dim) → Conv1d along T dimension
        self.temporal_conv = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1  # preserve temporal length
        )
        
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
        
        # Process each time step through spatial GCN layers
        # IMPORTANT: keep (batch, N, dim) format for ChebConv — don't flatten batch*N!
        temporal_outputs = []
        for t in range(T):
            x_t = X[:, :, t, :]  # (batch, N, F)
            # nn.Linear operates on last dim, so (batch, N, F) → (batch, N, hidden_dim) ✓
            h = self.input_proj(x_t)  # (batch, N, hidden_dim)
            
            for gcn_layer in self.spatial_layers:
                h = gcn_layer(h, self.L_scaled)  # batched ChebConv → (batch, N, hidden_dim)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            temporal_outputs.append(h)  # (batch, N, hidden_dim)
        
        # Stack temporal outputs: (batch, N, T, hidden_dim)
        temporal_seq = torch.stack(temporal_outputs, dim=2)
        # Reshape for Conv1d: (batch*N, hidden_dim, T) — Conv1d is per-node temporal
        temporal_seq = temporal_seq.reshape(batch_size * N, self.hidden_dim, T)
        
        # Temporal Conv1d
        temporal_out = self.temporal_conv(temporal_seq)  # (batch*N, hidden_dim, T)
        # Take last time step
        temporal_last = temporal_out[:, :, -1]  # (batch*N, hidden_dim)
        temporal_last = temporal_last.reshape(batch_size, N, self.hidden_dim)
        
        # Output prediction — nn.Linear on last dim: (batch, N, hidden_dim) → (batch, N, 1)
        y_pred = self.output_head(temporal_last)  # (batch, N, 1)
        return y_pred


class STGCNWrapper:
    """
    Wrapper conforming to unified BaseModel interface.
    
    ⚡ Uses GRAPH-FORMAT data: (W, N, T_in, F) where W=time windows, N=real graph nodes.
    This is CRITICAL — GNN models must process real graph topology (N×N adj_matrix),
    NOT flattened N*W samples which would create OOM-sized ChebConv matrices.
    
    data_format = 'graph' → trainer will pass graph-format data to this model.
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
        return "ST-GCN"
    
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
        W_train, N, T_in, F_dim = X_train.shape  # (W, N=40, T_in=24, F=12)
        input_dim = F_dim
        geo_dim = geo_features.shape[1]
        
        self.config['input_dim'] = input_dim
        self.config['geo_dim'] = geo_dim
        
        print(f"[ST-GCN] Graph-format data: W={W_train}, N={N}, "
              f"T_in={T_in}, F={F_dim}")
        print(f"[ST-GCN] adj_matrix={adj_matrix.shape}, "
              f"geo_features={geo_features.shape}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ST-GCN] Using device: {self.device}")
        
        # Precompute scaled Laplacian from ORIGINAL adjacency matrix (N, N)
        L_scaled_np = compute_scaled_laplacian(adj_matrix)
        L_scaled = torch.FloatTensor(L_scaled_np).to(self.device)
        
        self.model = STGCNNet(self.config)
        self.model._build_model(input_dim, geo_dim)
        self.model.L_scaled = L_scaled  # store precomputed Laplacian
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
        print(f"[ST-GCN] Training complete. Best loss: {best_loss:.4f}")
    
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
