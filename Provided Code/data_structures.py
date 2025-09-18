#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Data structures
@dataclass
class KPoint:
    """A single k-point block with weight and eigenvalues in eV."""
    weight: float
    eig_eV: np.ndarray  # shape (nspin, neig)

@dataclass
class ValBandsData:
    """Parsed contents of a .val_bands file (plus fallback EF if needed)."""
    nk: int
    nspin: int
    neig: int
    ef_eV: Optional[float]
    kpoints: list[KPoint]
