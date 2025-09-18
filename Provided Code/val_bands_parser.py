#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional
from data_structures import ValBandsData, KPoint
from parameters import HARTREE_TO_EV, logger

import re, numpy as np

class ValBandsParser:
    """Parser for ONETEP val_bands + EF fallback from logs."""

    _rx_int: dict[str, re.Pattern] = {
        "nk": re.compile(r"Number of k-points\s+(\d+)", re.IGNORECASE),
        "nspin": re.compile(r"Number of spin components\s+(\d+)", re.IGNORECASE),
        "neig": re.compile(r"Number of eigenvalues\s+(\d+)", re.IGNORECASE),
    }
    _rx_ef: re.Pattern = re.compile(r"Fermi\s+energy.*?([+\-]?\d+(?:\.\d*)?)", re.IGNORECASE)

    def parse(self, path: Path) -> ValBandsData:
        """
        Parse an ONETEP .val_bands file.

        Returns
        -------
        ValBandsData
            nk (number of k-points), nspin (number of spin components), 
            neig (number of eigenvalues) as reported (or inferred); ef_eV if 
            present in the file (converted Ha→eV); and a list of KPoint blocks 
            each with a scalar weight and an (nspin × neig) array of eigenvalues in eV.

        Behaviour
        ---------
        - Multi-spin blocks are detected via "Spin component" markers.
        - If no "K-point" headers are found, the file is treated as a single
          k-point with weight 1.0 and a flat list of eigenvalues (shape inferred).
        - Does not search logs for EF; see `fallback_ef_from_folder` for that.

        Raises
        ------
        FileNotFoundError
            If `path` does not exist.
        """
        
        logger.debug("Parsing val_bands: %s", path)
        
        if not path.exists(): raise FileNotFoundError(f"Missing file: {path}")
        text: str = path.read_text()
        lines: list[str] = text.splitlines()

        def _first_int(rx: re.Pattern) -> Optional[int]:
            m = rx.search(text)
            return int(m.group(1)) if m else None

        def _first_float(rx: re.Pattern) -> Optional[float]:
            m = rx.search(text)
            return float(m.group(1)) if m else None

        reported_nk: Optional[int] = _first_int(self._rx_int["nk"])
        reported_nspin: Optional[int] = _first_int(self._rx_int["nspin"])
        reported_neig: Optional[int] = _first_int(self._rx_int["neig"])

        nk: int = reported_nk or 0
        nspin: int = reported_nspin or 1
        neig: int = reported_neig or 0

        ef_ha: Optional[float] = _first_float(self._rx_ef)
        ef_eV: Optional[float] = ef_ha * HARTREE_TO_EV if ef_ha is not None else None
        
        logger.debug("Header values (reported): nk=%s, nspin=%s, neig=%s, EF(Ha)=%s",
                     reported_nk or "?", reported_nspin or "?", reported_neig or "?", 
                     f"{ef_ha:.6f}" if ef_ha is not None else "—",
                    )
        
        if ef_eV is not None: logger.debug("EF converted to eV: %.6f", ef_eV)

        # Identify "K-point" blocks (multi-spin aware)
        headers: list[int] = [i for i, l in enumerate(lines) if l.strip().startswith("K-point")]
        kpoints: list[KPoint] = []

        if not headers: # Single flat list of eigenvalues with k-point weighting 1.0
            logger.debug("No 'K-point' headers in %s; assuming single k-point with weight 1.0", path)
            
            vals: list[float] = []
            for l in lines: vals.extend([float(t) for t in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", l)])
            eig = np.array(vals, float).reshape(1, -1) * HARTREE_TO_EV
            kpoints.append(KPoint(weight=1.0, eig_eV=eig))
            nk = 1
            neig = eig.shape[1]
            nspin = eig.shape[0]
            logger.debug("Assumed shapes: nk=1, nspin=%d, neig=%d", nspin, neig)
            return ValBandsData(nk=nk, nspin=nspin, neig=neig, ef_eV=ef_eV, kpoints=kpoints)

        headers.append(len(lines))
        per_spin_lengths: set[int] = set() # For consistency checks
        for b in range(len(headers) - 1):
            i0, i1 = headers[b], headers[b+1]
            header = lines[i0]
            nums = re.findall(r"[+-]?\d+(?:\.\d+)?", header)
            weight: float = float(nums[-1]) if nums else 1.0
            if weight <= 0: logger.warning("Non-positive k-point weight %.3g at block starting line %d in %s", weight, i0, path)
            
            block = lines[i0+1:i1]
            spin_idx: list[int] = [j for j, l in enumerate(block) if re.search(r"Spin component", l, re.IGNORECASE)]
            
            if not spin_idx:
                vals: list[float] = []
                for l in block: vals.extend([float(t) for t in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", l)])
                eig = np.array(vals, float).reshape(1, -1)
                per_spin_lengths.add(eig.shape[1])
                
            else:
                spin_idx.append(len(block))
                parts: list[np.ndarray] = []
                for s in range(len(spin_idx)-1):
                    a, bnd = spin_idx[s] + 1, spin_idx[s+1]
                    vals: list[float] = []
                    for l in block[a:bnd]: vals.extend([float(t) for t in re.findall(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", l)])
                    parts.append(np.array(vals, float))
                    
                try: eig = np.stack(parts, axis=0); per_spin_lengths.update(p.size for p in parts)
                except ValueError:
                    logger.warning("Inconsistent eigenvalue counts across spin components in block starting line %d (%s)", i0, path)
                    # Attempt ragged fallback is not trivial; return early with what we have
                    eig = np.vstack([p.reshape(1, -1) for p in parts])
                    per_spin_lengths.update(p.size for p in parts) 
            
            kpoints.append(KPoint(weight=weight, eig_eV=eig * HARTREE_TO_EV))
            
        if len(kpoints) == 0: 
            logger.warning("No k-points parsed from %s; downstream metrics will be NaN", path)
            return ValBandsData(nk=0, nspin=nspin, neig=neig, ef_eV=ef_eV, kpoints=[])
        
        # Infer missing counts from parsed data
        if nk == 0: nk = len(kpoints); logger.debug("nk inferred from parsed data: %d", nk)
        if neig == 0: neig = kpoints[0].eig_eV.shape[1]; logger.debug("neig inferred from parsed data: %d", neig)
        if nspin == 0: nspin = kpoints[0].eig_eV.shape[0]; logger.debug("nspin inferred from parsed data: %d", nspin)
        
        # Compare reported vs inferred
        if reported_nk and reported_nk != nk: logger.warning("Reported nk=%d differs from parsed nk=%d in %s", reported_nk, nk, path)
        if reported_neig and reported_neig != neig: logger.warning("Reported neig=%d differs from parsed neig=%d in %s", reported_neig, neig, path)
        if reported_nspin and reported_nspin != nspin: logger.warning("Reported nspin=%d differs from parsed nspin=%d in %s", reported_nspin, nspin, path)

        # Sanity check: per-spin eigenvalue counts
        if len(per_spin_lengths) > 1: logger.warning("Inconsistent eigenvalue counts across blocks/spins in %s: %s", path, sorted(per_spin_lengths))

        logger.debug("Parsed k-points: %d (nspin=%d, neig=%d)", len(kpoints), nspin, neig)

        return ValBandsData(nk=nk, nspin=nspin, neig=neig, ef_eV=ef_eV, kpoints=kpoints)

    def fallback_ef_from_folder(self, folder: Path) -> Optional[float]:
        """
        Scan typical ONETEP logs for 'Fermi energy (in atomic units)  <value>'.
        Return EF in eV (Ha→eV) or None if not found.
        """
        logger.debug("Searching EF fallback in %s", folder)
        
        candidates: list[Path] = []
        for pat in ("*.onetep", "*.out"): candidates.extend(folder.glob(pat))
        
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        logger.debug("Scanning %d candidate log files for EF", len(candidates))

        for p in candidates:
            try: text = p.read_text(errors="ignore")
            except Exception: continue
            
            hits = self._rx_ef.findall(text)
            if hits:
                try: ef_ha = float(hits[-1]); logger.debug("EF fallback from %s: %.6f eV", p, ef_ha * HARTREE_TO_EV); return ef_ha * HARTREE_TO_EV
                except ValueError: pass
                
        logger.warning("No EF found in logs under %s", folder)
        return None