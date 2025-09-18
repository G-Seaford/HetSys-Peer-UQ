#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging, matplotlib as mpl

# Please Note:
# This file contains selected resources to be used in a custom pipeline for reproducibility of the Summer Project Report.
# It is not a standalone script and is intended to be imported as a module and/or copied into other scripts.
# Parameters have been provided for you to reuse.

# DoS evaluation
BROADENING_EV: float = 0.10  # Gaussian broadening (σ) for DoS kernel in eV; choose to balance smoothness vs bias.
N_GRID: int = 2000
ENERGY_RANGE: tuple[float, float] = (-10.0, +10.0) # Absolute energy grid for DoS evaluation; Dirac search is relative to EF.

# Offsets / normalisation
OFFSET_STEP_DEFAULT: float = 1.00
OFFSET_STEP_PER_SYSTEM: dict[str, float] = {"Au-000": 2.05, }  # larger spacing for the clean graphene system

MARGIN: float = 0.05
NORMALISATION: str = "none"   # "none", "panel_max", "area", or "curve_max"

# Figure sizes / saving
FIGSIZE_1x4: tuple[float, float] = (16.0, 3.8)
FIGSIZE_2x2: tuple[float, float] = (10.6, 6.4)
FIGSIZE_DELTA_SERIES: tuple[float, float] = (10.6, 6.0)
FIGSIZE_DELTA_HEAT: tuple[float, float]   = (8.2, 3.8)
DPI_SAVE: int = 800

HARTREE_TO_EV: float = 27.211386245988

# Axis ticks
X_MAJOR_EVERY_EV: float = 5.0
X_MINOR_EVERY_EV: float = 1.0

# Colours (Okabe–Ito + black)
COLOURS: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "blue":   "#0072B2",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

BIAS_COLOUR: dict[str, str] = {
    "-0.10V": COLOURS["blue"],
    "+0.00V": COLOURS["black"],
    "+0.10V": COLOURS["orange"],
}

# Typography & line widths
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "axes.unicode_minus": True,
})

# Logging setup
logger = logging.getLogger(__name__)