#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical constraint checking module.

Stefan equation validation:
  - Compute Stefan ALT from geological features
  - Check if predictions violate physical constraints
  - Output violation statistics
"""

import numpy as np
from typing import Dict, Tuple


def compute_stefan_alt(geo_features: np.ndarray,
                       k_thaw: float = 1.2,
                       L_latent: float = 3.34e7) -> np.ndarray:
    """
    Compute Stefan equation ALT.
    
    ALT = sqrt(2 * k_thaw * I_t * 86400 / L)
    
    Args:
        geo_features: (N, G) — columns: [ALT, k_s, I_t, ...]
        k_thaw:       thermal conductivity (W/m·K)
        L_latent:     volumetric latent heat (J/m³)
    Returns:
        alt: (N,) — Active Layer Thickness in meters
    """
    I_t = geo_features[:, 2]  # thawing index (°C·day)
    I_t_sec = I_t * 86400.0  # convert day → seconds
    
    alt = np.sqrt(2.0 * k_thaw * I_t_sec / L_latent + 1e-8)
    return alt


def check_physical_constraints(y_pred: np.ndarray,
                               geo_features: np.ndarray,
                               k_thaw: float = 1.2,
                               L_latent: float = 3.34e7) -> Dict:
    """
    Check if predictions violate Stefan-derived physical constraints.
    
    Physical constraint: |settlement| ≤ k_s * ALT
    
    Args:
        y_pred:        (N,) predicted settlement values
        geo_features:  (N, G) — columns: [ALT, k_s, I_t, ...]
        k_thaw:        thermal conductivity
        L_latent:      volumetric latent heat
    Returns:
        constraint_report: dict with violation statistics
    """
    y_pred_flat = np.asarray(y_pred).flatten()
    
    k_s = geo_features[:, 1]  # soil-structure coefficient
    alt = compute_stefan_alt(geo_features, k_thaw, L_latent)
    
    # Physical bound: |settlement| ≤ k_s * ALT
    physical_bound = k_s * alt
    violation = np.abs(y_pred_flat) > physical_bound
    
    # Statistics
    violation_count = violation.sum()
    total_count = len(y_pred_flat)
    violation_rate = violation_count / total_count * 100 if total_count > 0 else 0
    
    # Violation magnitude
    violation_magnitude = np.abs(y_pred_flat[violation]) - physical_bound[violation]
    mean_violation_mag = violation_magnitude.mean() if violation_count > 0 else 0
    max_violation_mag = violation_magnitude.max() if violation_count > 0 else 0
    
    report = {
        'violation_count': int(violation_count),
        'total_count': int(total_count),
        'violation_rate_pct': float(violation_rate),
        'mean_violation_magnitude': float(mean_violation_mag),
        'max_violation_magnitude': float(max_violation_mag),
        'physical_bound_mean': float(physical_bound.mean()),
        'physical_bound_std': float(physical_bound.std()),
        'alt_mean': float(alt.mean()),
        'alt_std': float(alt.std()),
    }
    
    print(f"[Physical Constraint Check] Violation rate: {violation_rate:.1f}% "
          f"({violation_count}/{total_count})")
    if violation_count > 0:
        print(f"  Mean violation magnitude: {mean_violation_mag:.4f}")
        print(f"  Max violation magnitude: {max_violation_mag:.4f}")
    
    return report


class PhysicalConstraintChecker:
    """
    Physical constraint checker class for systematic evaluation.
    """
    def __init__(self, k_thaw: float = 1.2, L_latent: float = 3.34e7):
        self.k_thaw = float(k_thaw)
        self.L_latent = float(L_latent)
    
    def check(self, y_pred: np.ndarray, geo_features: np.ndarray) -> Dict:
        """Check physical constraints."""
        return check_physical_constraints(
            y_pred, geo_features, self.k_thaw, self.L_latent
        )
    
    def compute_alt(self, geo_features: np.ndarray) -> np.ndarray:
        """Compute Stefan ALT."""
        return compute_stefan_alt(geo_features, self.k_thaw, self.L_latent)
