# -*- coding: utf-8 -*-
"""
Lux-Marchesi Financial Market Model
====================================

A Python implementation of the Lux-Marchesi agent-based financial market model,
supporting both traditional ABM and LLM-based decision modes.

Components:
- lux_marchesi_model: Main simulation model
- evaluation: Stylized facts analysis tools
- visualization: Plotting utilities

Usage:
    from lux_marchesi_model import LuxMarchesiModel, LuxMarchesiParams
    from evaluation import full_evaluation
    from visualization import create_comprehensive_figure
"""

from .lux_marchesi_model import LuxMarchesiModel, LuxMarchesiParams

__all__ = ['LuxMarchesiModel', 'LuxMarchesiParams']
__version__ = '1.0.0'

