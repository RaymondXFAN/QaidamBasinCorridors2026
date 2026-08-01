#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for all GNN models.

Provides subgraph extraction for handling mismatched N_samples vs N_total.
"""

import numpy as np


def extract_subgraph(X, adj_matrix, geo_features):
    """
    Extract subgraph when N_samples < N_total.
    
    If X.shape[0] < adj_matrix.shape[0], extract the relevant rows/columns
    from adj_matrix and geo_features to match the sample count.
    
    Args:
        X:            (N_samples, T, F)
        adj_matrix:   (N_total, N_total) or (N_samples, N_samples)
        geo_features: (N_total, G) or (N_samples, G)
    Returns:
        adj_sub, geo_sub matching N_samples
    """
    N_samples = X.shape[0]
    N_total = adj_matrix.shape[0]
    
    if N_samples == N_total:
        return adj_matrix, geo_features
    
    if N_samples < N_total:
        # Extract first N_samples rows/columns from adj_matrix
        adj_sub = adj_matrix[:N_samples, :N_samples]
        geo_sub = geo_features[:N_samples, :]
        print(f"[Subgraph] Extracted {N_samples} nodes from {N_total} total")
        return adj_sub, geo_sub
    
    # N_samples > N_total: pad adj_matrix and geo_features
    adj_pad = np.zeros((N_samples, N_samples), dtype=adj_matrix.dtype)
    adj_pad[:N_total, :N_total] = adj_matrix
    adj_pad[N_total:, N_total:] = np.eye(N_samples - N_total)  # self-loops for new nodes
    geo_pad = np.zeros((N_samples, geo_features.shape[1]), dtype=geo_features.dtype)
    geo_pad[:N_total, :] = geo_features
    print(f"[Subgraph] Padded from {N_total} to {N_samples} nodes")
    return adj_pad, geo_pad
