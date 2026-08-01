#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainers module: generic training and evaluation utilities.
"""

from .trainer import train_model, evaluate_model, run_all_experiments

__all__ = [
    'train_model',
    'evaluate_model',
    'run_all_experiments',
]
