#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
from data_structures import ValBandsData
from parameters import logger

import numpy as np

class DoSBuilder:
    """Builds DoS on a fixed energy grid from valence eigenvalues."""

    @staticmethod
    def flatten_states(data: ValBandsData, align_zero: bool = False, spin: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Concatenate all eigenvalues into (E, W) arrays suitable for DoS evaluation.

        Parameters
        ----------
        align_zero : bool
            If True and EF is known, subtract EF so returned energies are relative
            to EF. Otherwise, energies are absolute (eV).
        spin : int or None
            Select a single spin channel (0-based). If None, include all spins.

        Returns
        -------
        E : (N,) float64
            Eigenvalues (eV), flattened across k-points (and spins if not selected).
        W : (N,) float64
            Corresponding k-point weights, repeated per eigenvalue.

        Notes
        -----
        Weights are carried through to the DoS as multiplicative factors.
        """
        logger.debug("Flatten states: align_zero=%s, spin=%s, nk=%d", align_zero, spin, len(data.kpoints))
        if align_zero and data.ef_eV is None: logger.debug("align_zero=True but EF is missing; energies remain absolute")
        
        ef: float = data.ef_eV or 0.0
        E_list: list[np.ndarray] = []
        W_list: list[np.ndarray] = []
        for kp in data.kpoints:
            w: float = kp.weight
            eig: np.ndarray = kp.eig_eV
            if spin is not None:
                if not (0 <= spin < eig.shape[0]): logger.warning("Requested spin %d out of range [0,%d); skipping this k-point", spin, eig.shape[0]); continue
                else: eig = kp.eig_eV[spin][None, :]
            
            if align_zero and data.ef_eV is not None: eig = eig - ef
            
            E_list.append(eig.reshape(-1))
            W_list.append(np.full(eig.size, w, dtype=float))
            
        if not E_list: logger.debug("No eigenvalues present; returning empty arrays"); return np.array([]), np.array([])
        
        E = np.concatenate(E_list); W = np.concatenate(W_list)
        logger.debug("Flattened arrays: E.size=%d, W.size=%d", E.size, W.size)
        return E, W

    @staticmethod
    def gaussian_dos(E: np.ndarray, W: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian-broadened DoS on `grid` with stddev `sigma` (eV)."""
        logger.debug("gaussian_dos: N=%d, sigma=%.3f, gridN=%d", E.size, sigma, grid.size)
        
        if E.size == 0: logger.debug("gaussian_dos: empty E; returning zeros"); return np.zeros_like(grid)
        if sigma <= 0: logger.warning("gaussian_dos: non-positive sigma=%.3g; returning zeros", sigma); return np.zeros_like(grid)
        
        diff = grid[:, None] - E[None, :]
        K = np.exp(-0.5 * (diff / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        
        logger.debug("gaussian_dos: computed DOS array of length %d", grid.size)
        return (K * W[None, :]).sum(axis=1)