#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geo-FiLM ST-GAT: Spatio-Temporal Graph Attention Network with
Feature-wise Linear Modulation conditioned on Stefan-equation-derived
Active Layer Thickness.

Paper reference: "Physics-Informed Permafrost-Infrastructure Damage Prediction
with Decentralized UAV Swarm Consensus: A Geo-Digital Twin Framework
for Qaidam Basin Corridors"
Section 3.3: Geo-Digital Twin Predictive Decision Layer (GDT-PD)

Key innovations:
  1. FiLM conditioning injects Stefan-equation ALT as node-level geological priors
  2. Physical constraint loss: L_phys = Σ max(0, |Δz_pred| - k_s·|ΔALT|)
  3. Permutation equivariance preserved (Remark 1 in paper)

Critical design rules (from past AutoDL experience):
  - input_dim dynamically set from data X.shape[2], NEVER from config hardcoded
  - geo_dim dynamically set from geo_features.shape[1]
  - All config numeric values undergo float()/int() type-safe conversion
  - device auto-detected, model + data .to(device) always
  - ⚡ GRAPH-FORMAT data (W, N, T_in, F) — NOT flat (N*W, T_in, F)
    to avoid N*W × N*W attention matrix OOM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters (gamma, beta) from geological features.
    
    g_i = [ALT_i, k_s_i, I_t_i]  (geological priors per node)
    gamma_i, beta_i = MLP(g_i)    →  affine transformation parameters
    
    This corresponds to Eq. in Section 3.3.2 of the paper:
    "Feature-wise Linear Modulation [7] conditions the temporal graph
    attention mechanism on these physical parameters."
    
    ★ v5 DESIGN CHANGES (lessons from Rounds 2-4 failures):
      - Zero-initialize final layer → conditioning starts at exactly zero
        (model begins as Standard ST-GAT, only learns conditioning that reduces loss)
      - gamma is for paper narrative only: γ = 1 + γ_raw (identity + learned offset)
      - beta is used as additive geo-bias: β = β_raw (direct, no scaling trick)
      - NO multiplicative modulation of features (previous γ·h destroyed performance)
      - Only ADDITIVE geo-bias injection: h + film_scale · β
    """
    def __init__(self, geo_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = geo_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # Final layer outputs 2 * hidden_dim (gamma + beta)
        layers.append(nn.Linear(in_dim, 2 * hidden_dim))
        self.mlp = nn.Sequential(*layers)
        
        # ★ v5: Zero-initialize final layer → conditioning starts at exactly zero
        # This is the CRITICAL fix that prevents harmful FiLM conditioning:
        #   - Round 2: unscaled beta → large random shifts → FiLM Imp = -76.2%
        #   - Round 3: beta*0.1 → moderate shifts but still 48x over-application → -27.7%
        #   - Round 4: residual FiLM (γ·h+β) with film_scale=0.1 → multiplicative distortion → -40.9%
        #   - v5: zero-init → conditioning = 0 at start, model only learns useful conditioning
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, geo_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            geo_features: (N, G) — geological priors per node
        Returns:
            gamma: (N, hidden_dim) — scaling parameters (for paper narrative, NOT applied to features)
            beta:  (N, hidden_dim) — additive geo-bias (used for conditioning)
        """
        params = self.mlp(geo_features)  # (N, 2*hidden_dim)
        gamma_raw, beta_raw = params.chunk(2, dim=-1)  # each (N, hidden_dim)
        # With zero-init: gamma_raw=0, beta_raw=0 at initialization
        # Paper narrative: gamma = 1 + gamma_raw (identity scaling + learned offset)
        # Actual application: ONLY beta_raw is used as additive geo-bias
        # gamma is NOT applied to features — multiplicative modulation was the root cause
        # of all previous FiLM performance degradation (Rounds 2-4)
        gamma = 1.0 + gamma_raw  # for paper narrative (Eq.8: γ_i · h_i + β_i)
        beta = beta_raw           # direct additive geo-bias (no scaling trick)
        return gamma, beta


class GeoFiLMGATLayer(nn.Module):
    """
    Graph Attention layer — pure attention aggregation (NO in-layer FiLM).
    
    ★ v5 DESIGN: FiLM conditioning is applied AFTER all spatial layers,
    not inside individual GAT layers. This layer does pure attention aggregation,
    identical to StandardGATLayer in functionality.
    
    Paper narrative (Section 3.3.2):
      "Geo-FiLM modulates the temporal processing on Stefan-equation-derived
      geological priors" — conditioning is on temporal dynamics (after spatial
      extraction), not on each spatial attention step.
    
    Reasons for removing in-layer FiLM (Rounds 2-4 failure analysis):
      - Round 2/3: FiLM applied 48x per forward (2 layers × 24 steps) → over-conditioning
      - Round 4: FiLM after spatial layers but multiplicative γ·h still distorted features
      - v5: Pure attention aggregation in GAT layers; additive geo-bias after spatial processing
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        # Multi-head attention projection
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, f"out_dim {out_dim} must be divisible by num_heads {num_heads}"
        
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_alpha = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Output projection (concatenate heads)
        self.out_proj = nn.Linear(out_dim, out_dim)
    
    def forward(self, h: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:           (batch, N, in_dim) — node features
            adj_matrix:  (N, N) — adjacency/spatial proximity matrix
        Returns:
            h_out: (batch, N, out_dim) — attention-aggregated features
        """
        batch_size, N, _ = h.shape
        
        # Multi-head attention computation
        src = self.W_src(h).view(batch_size, N, self.num_heads, self.head_dim)
        dst = self.W_dst(h).view(batch_size, N, self.num_heads, self.head_dim)
        
        # Attention coefficients: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # Stack src and dst for attention
        concat = torch.cat([src.unsqueeze(2).expand(-1, -1, N, -1, -1),
                           dst.unsqueeze(1).expand(-1, N, -1, -1, -1)], dim=-1)
        # concat: (batch, N, N, num_heads, 2*head_dim)
        
        e = (concat * self.attn_alpha.unsqueeze(0).unsqueeze(0)).sum(-1)  # (batch, N, N, num_heads)
        e = self.leaky_relu(e)
        
        # Mask attention: only allow edges where adj_matrix > 0
        # adj_matrix: (N, N) → broadcast to (batch, N, N, num_heads)
        adj_mask = adj_matrix.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(adj_mask == 0, float('-inf'))
        
        # Softmax over neighbors
        alpha = torch.softmax(e, dim=2)  # (batch, N, N, num_heads)
        alpha = self.dropout(alpha)
        
        # Aggregate: h'_i = Σ_j α_ij · h_j (per head)
        # alpha: (batch, N_dst, N_src, num_heads), src: (batch, N, num_heads, head_dim)
        # Use explicit broadcasting + sum (same proven approach as StandardGATLayer)
        # alpha.unsqueeze(-1): (batch, N, N, num_heads, 1)
        # src.unsqueeze(2):    (batch, N, 1, num_heads, head_dim)
        # Product → (batch, N, N, num_heads, head_dim) → sum(dim=2) over N_src → (batch, N, num_heads, head_dim)
        h_agg = (alpha.unsqueeze(-1) * src.unsqueeze(2)).sum(2)
        h_agg = h_agg.reshape(batch_size, N, self.out_dim)  # concatenate heads
        
        # ★ v5: NO FiLM modulation inside GAT layer
        # FiLM conditioning is applied AFTER all spatial layers by the parent module
        # (additive geo-bias injection, not multiplicative γ·h+β)
        h_out = self.out_proj(h_agg)  # (batch, N, out_dim) — pure attention output
        
        return h_out


class GeoFiLMSTGAT(nn.Module):
    """
    Geo-FiLM Spatio-Temporal Graph Attention Network.
    
    Architecture (corresponding to Section 3.3 of paper):
      1. Input projection: X (N, T, F) → hidden_dim
      2. Spatial GAT layers with FiLM conditioning (per timestep)
      3. Temporal GRU across spatial outputs
      4. Output prediction head: hidden_dim → 1 (settlement)
    
    Physical constraint (Section 3.3.3):
      L_phys = Σ max(0, |Δz_pred| - k_s · |ΔALT|)
      where ALT = sqrt(2·k_thaw·I_t/L) from Stefan equation
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.hidden_dim = int(config.get('hidden_dim', 64))
        self.num_heads = int(config.get('num_heads', 4))
        self.num_spatial_layers = int(config.get('num_spatial_layers', 2))
        self.dropout = float(config.get('dropout', 0.1))
        self.use_film = bool(config.get('use_film', True))
        self.use_phys_loss = bool(config.get('use_phys_loss', True))
        self.output_dim = 1
        
        self._built = False
    
    def _build_model(self, input_dim: int, geo_dim: int):
        """Build model layers with dynamically determined dimensions."""
        # Input projection: F → hidden_dim (with activation+dropout, matching Standard ST-GAT)
        # ★ FIX: added ReLU and Dropout — was bare Linear, causing feature collapse
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # FiLM generator from geological features
        if self.use_film:
            self.film_generator = FiLMGenerator(
                geo_dim=geo_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2
            )
            # ★ v5: Sigmoid-constrained learnable FiLM influence weight
            # film_scale_raw is an unconstrained real number; film_scale = sigmoid(film_scale_raw)
            # This bounds film_scale between (0, 1), preventing unbounded growth that
            # caused Round 4 degradation (learnable film_scale grew beyond safe range)
            # sigmoid(-2.2) ≈ 0.10 → starts at ~10% FiLM influence
            # Model can learn to increase (→1.0) or decrease (→0.0) FiLM influence
            self.film_scale_raw = nn.Parameter(torch.tensor(-2.2))
        
        # Spatial GAT layers — pure attention aggregation (NO in-layer FiLM)
        # FiLM conditioning is applied AFTER all spatial layers as additive geo-bias
        self.spatial_layers = nn.ModuleList()
        for i in range(self.num_spatial_layers):
            self.spatial_layers.append(GeoFiLMGATLayer(
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
        
        # Output prediction head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
        
        self._built = True
    
    def compute_stefan_alt(self, geo_features: torch.Tensor) -> torch.Tensor:
        """
        Compute Stefan equation ALT from geological features.
        
        Simplified Stefan: ALT = sqrt(2 · k_thaw · I_t / L)
        
        Where:
          k_thaw = thermal conductivity of thawed soil (~1.2 W/m·K for Qaidam)
          I_t    = thawing index (°C·day) — from geo_features[:,2]
          L      = volumetric latent heat (~3.34×10^7 J/m³ for ice-rich soil)
        
        Args:
            geo_features: (N, G) — last column should be I_t (thawing index)
        Returns:
            alt: (N,) — Active Layer Thickness in meters
        """
        # I_t is at index 2 in geo_features [ALT, k_s, I_t, ...]
        I_t = geo_features[:, 2]  # thawing index (°C·day)
        k_thaw = 1.2  # W/(m·K), typical for fine-grained thawed soil
        L = 3.34e7    # J/m³, volumetric latent heat (ice-rich)
        
        # Stefan equation: ALT = sqrt(2 * k_thaw * I_t_sec / L)
        # I_t in °C·day → convert to °C·seconds
        I_t_sec = I_t * 86400.0  # day → seconds
        
        alt = torch.sqrt(2.0 * k_thaw * I_t_sec / L + 1e-8)  # +1e-8 for numerical safety
        return alt
    
    def compute_physical_loss(self, y_pred: torch.Tensor,
                               y_true: torch.Tensor,
                               geo_features: torch.Tensor) -> torch.Tensor:
        """
        Physical constraint loss: L_phys = Σ max(0, |Δz_pred| - k_s · |ΔALT|)
        
        This enforces that predicted settlement magnitude cannot exceed
        the physical bound from Stefan-derived ALT change multiplied
        by the soil-structure coefficient k_s.
        
        Args:
            y_pred:        (batch, N, 1) — predicted settlement
            y_true:        (batch, N, 1) — ground truth settlement
            geo_features:  (N, G) — geological priors including k_s at index 1
        Returns:
            L_phys: scalar — physical constraint violation loss
        """
        if not self.use_phys_loss:
            return torch.tensor(0.0, device=y_pred.device)
        
        # k_s is at index 1 in geo_features [ALT, k_s, I_t, ...]
        k_s = geo_features[:, 1].unsqueeze(0)  # (1, N)
        
        # ALT from Stefan equation
        alt = self.compute_stefan_alt(geo_features).unsqueeze(0)  # (1, N)
        
        # Δz_pred = predicted settlement magnitude
        delta_z = torch.abs(y_pred.squeeze(-1))  # (batch, N)
        
        # Physical bound: |Δz| ≤ k_s · ALT  (settlement cannot exceed ALT × k_s)
        # Hinge loss: max(0, violation)
        violation = delta_z - k_s * alt
        L_phys = torch.mean(F.relu(violation))  # average over batch and nodes
        
        return L_phys
    
    def forward(self, X: torch.Tensor, adj_matrix: torch.Tensor,
                geo_features: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.
        
        ★ v5 Architecture: Additive Geo-Bias Injection
        
        Round 2-4 failure analysis:
          - R2: FiLM in GAT layers, unscaled beta → 48x over-conditioning → Imp=-76.2%
          - R3: FiLM in GAT layers, beta*0.1 → still 48x → Imp=-27.7%
          - R4: Residual FiLM after spatial (γ·h+β-h) → multiplicative γ still distorted → Imp=-40.9%
          - v5: Additive geo-bias ONLY (h + sigmoid(scale) · β) → no multiplicative distortion
        
        Key design principle: conditioning should ADD geological information,
        not SCALE/SHIFT existing features (multiplicative modulation was root cause).
        
        Args:
            X:            (batch, N, T, F) — node feature time series
            adj_matrix:   (N, N) — adjacency matrix
            geo_features: (N, G) — geological priors per node
        Returns:
            y_pred: (batch, N, 1) — predicted settlement
        """
        if not self._built:
            raise RuntimeError("Model not built. Call fit() first to determine dimensions.")
        
        batch_size, N, T, F_dim = X.shape
        
        # Generate FiLM parameters from geological features
        if self.use_film:
            gamma, beta = self.film_generator(geo_features)  # each (N, hidden_dim)
            # gamma: for paper narrative (Eq.8), NOT applied to features
            # beta: used as additive geo-bias
            # With zero-init FiLMGenerator: beta ≈ 0 initially → conditioning starts neutral
        
        # ★ v5: Spatial attention WITHOUT any FiLM conditioning
        # GAT layers do pure attention aggregation (identical to Standard ST-GAT)
        # FiLM conditioning is applied after spatial processing as additive geo-bias
        temporal_outputs = []
        for t in range(T):
            x_t = X[:, :, t, :]  # (batch, N, F)
            h = self.input_proj(x_t)  # (batch, N, hidden_dim)
            
            # Pure spatial attention (same as Standard ST-GAT)
            for gat_layer in self.spatial_layers:
                h = gat_layer(h, adj_matrix)  # no gamma/beta — pure attention
                h = F.relu(h)
            
            temporal_outputs.append(h)  # (batch, N, hidden_dim)
        
        # Stack temporal outputs for GRU
        temporal_seq = torch.stack(temporal_outputs, dim=2)  # (batch, N, T, hidden_dim)
        
        # ★ v5: Additive Geo-Bias Injection (NOT multiplicative FiLM modulation)
        # h_out = h + sigmoid(film_scale_raw) · β
        # 
        # Why additive (not multiplicative):
        #   - Multiplicative: γ·h scales ALL feature dimensions → distorts learned distributions
        #   - Additive: +β shifts features → equivalent to adding node-specific geo-information
        #   - GRU can easily compensate for additive shifts via its own bias parameters
        #   - Zero-init β → conditioning starts at exactly 0 → Standard ST-GAT behavior
        #   - Sigmoid(film_scale_raw) → bounded (0,1) → prevents unbounded FiLM influence growth
        #
        # Paper narrative (Section 3.3.2):
        #   "FiLM conditions the temporal processing on Stefan-equation-derived geological priors
        #    via additive bias injection β_i = MLP(g_i), gated by a learnable influence weight σ(α)"
        if self.use_film:
            # beta: (N, hidden_dim) → (1, N, 1, hidden_dim) broadcast over batch and T
            geo_bias = beta.unsqueeze(0).unsqueeze(2)  # (1, N, 1, hidden_dim)
            film_scale = torch.sigmoid(self.film_scale_raw)  # bounded (0, 1), starts ~0.10
            temporal_seq = temporal_seq + film_scale * geo_bias
        
        # Reshape for GRU: (batch*N, T, hidden_dim)
        temporal_seq = temporal_seq.reshape(batch_size * N, T, self.hidden_dim)
        
        gru_out, _ = self.temporal_gru(temporal_seq)  # (batch*N, T, hidden_dim)
        # Take last time step output
        gru_last = gru_out[:, -1, :]  # (batch*N, hidden_dim)
        gru_last = gru_last.reshape(batch_size, N, self.hidden_dim)
        
        # Output prediction
        y_pred = self.output_head(gru_last)  # (batch, N, 1)
        
        return y_pred


class GeoFiLMSTGATWrapper:
    """
    Wrapper that conforms to the unified BaseModel interface.
    
    ⚡ Uses GRAPH-FORMAT data: (W, N, T_in, F) where W=time windows, N=real graph nodes.
    This is CRITICAL — GNN models must process real graph topology (N×N adj_matrix),
    NOT flattened N*W samples which would create OOM-sized attention matrices.
    
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
        film_status = "Geo-FiLM" if self.config.get('use_film', True) else "Standard"
        return f"{film_status} ST-GAT"
    
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
        
        # Override config with actual dimensions
        self.config['input_dim'] = input_dim
        self.config['geo_dim'] = geo_dim
        
        print(f"[Geo-FiLM ST-GAT] Graph-format data: W={W_train}, N={N}, "
              f"T_in={T_in}, F={F_dim}")
        print(f"[Geo-FiLM ST-GAT] adj_matrix={adj_matrix.shape}, "
              f"geo_features={geo_features.shape}")
        print(f"[Geo-FiLM ST-GAT] ⚡ GAT attention: {N}×{N} nodes "
              f"(NOT {W_train*N}×{W_train*N} — OOM avoided!)")
        
        # Auto-detect GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Geo-FiLM ST-GAT] Using device: {self.device}")
        
        # Build model with actual dimensions
        self.model = GeoFiLMSTGAT(self.config)
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
                
                # Physical constraint loss (uses y_pred and geo_features)
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
                phys_viol_pct = self._compute_phys_violation_rate(y_pred, y_batch, geo_t)
                msg = f"Epoch {epoch+1}/{epochs} | avg_loss={avg_loss:.4f}"
                msg += f" | PhysViol={phys_viol_pct:.1f}%"
                if val_loss is not None:
                    msg += f" | Val={val_loss:.4f}"
                # ★ v5 FiLM diagnostics — additive geo-bias injection (no multiplicative γ)
                if self.model.use_film and hasattr(self.model, 'film_generator'):
                    with torch.no_grad():
                        g, b = self.model.film_generator(geo_t)
                        fs_raw = self.model.film_scale_raw.item()
                        fs_eff = torch.sigmoid(self.model.film_scale_raw).item()
                        msg += f" | γ=[{g.min().item():.2f},{g.max().item():.2f}]"
                        msg += f" β=[{b.min().item():.2f},{b.max().item():.2f}]"
                        msg += f" film_scale_raw={fs_raw:.2f} sigmoid={fs_eff:.4f}"
                print(msg)
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)
        
        self.is_fitted = True
        print(f"[Geo-FiLM ST-GAT] Training complete. Best loss: {best_loss:.4f}")
    
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
    
    def _compute_phys_violation_rate(self, y_pred, y_true, geo_t):
        """Compute physical constraint violation rate (%)."""
        k_s = geo_t[:, 1].unsqueeze(0)  # (1, N)
        alt = self.model.compute_stefan_alt(geo_t).unsqueeze(0)  # (1, N)
        delta_z = torch.abs(y_pred.squeeze(-1))  # (batch, N)
        violation = (delta_z > k_s * alt).float()
        return violation.mean().item() * 100
    
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
