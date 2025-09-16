#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Optional, Callable

import argparse, csv, json, logging, sys, re, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

"""
Reduced DoS and Dirac energy analysis for ONETEP outputs.

Inputs
------
For each system/bias, this script expects two files under e.g.
  <SYSTEM>/<BIAS>/:
    - "<prefix>_vacuum.val_bands"  (vacuum/jellium)
    - "<prefix>.val_bands"         (solvated)
where <prefix> is "graphene0" for Au-000 and "Au-Graphene0" otherwise.

If a .val_bands file lacks a Fermi energy, the code scans recent logs
in that folder for "Fermi energy (in atomic units)" and converts Ha→eV.

Processing
----------
1) Parse valence eigenvalues (eV) and k-point weights (dimensionless).
2) Build a Gaussian-broadened DoS on a fixed energy grid over ±10 eV
   centred at 0 eV (absolute energies unless noted), with σ = 0.10 eV.
3) Define the Dirac energy E_D as the energy of the global DoS minimum
   within a ±2 eV window relative to EF for that bias/environment.
4) Quantify uncertainty in E_D via leave-one-out jackknife over
   k-points (1σ); report 2σ = 2x Std. Err. (SE). A local quadratic fit around
   the minimum provides an alternative 1σ estimate (diagnostic only).
5) Produce stacked DoS plots (vac/solv), a 2x2 comparison panel, and
   ΔE_D = E_D(solv) - E_D(vac) summaries (series + heatmap).
6) Write per-case metrics to 'UQ_results.csv' and ΔE_D to 'Dirac_deltas.csv'
   in cfg.figdir.

Outputs
-------
PNG figures and CSV tables in cfg.figdir (default: 'Figures/DoS)).

Notes
-----
- Energies are in eV throughout after conversion from Ha.
- When EF cannot be identified, E_D and related metrics are left blank (NaN).
- Normalisation/offset settings affect only vertical stacking for plotting,
  not any energy metrics.
"""

# DoS evaluation
BROADENING_EV: float = 0.10  #Gaussian broadening (σ) for DoS kernel in eV; choose to balance smoothness vs bias.
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

# Command line interface
@dataclass
class CLIConfig:
    """
    Wrap CLI parsing, system discovery, bias discovery, prefix resolution,
    logging config, pretty system names, and output directory management.

    Inputs (CLI)
    ------------
    - data_root: root folder that contains system folders (default: current directory)
    - sys-include / sys-exclude: regex filters on system labels (relative to root)
    - only-au: keep systems whose top folder starts with 'Au-'
    - sites/heights: optional allow-lists for Site-* and height folders
    - bias-regex: regex that matches bias directory names
    - prefix-mode: how to get ONETEP prefix ('auto'|'rule'|'map')
    - prefix-au000 / prefix-default: rule-mode fallbacks
    - prefix-map-json: JSON with list of {"regex": "...", "prefix": "..."} items
    - logging flags: -v/-q/--log-level/--log-file (same semantics as before)

    Resolved at parse time
    ----------------------
    - systems_map: {system_label -> Path}
    - biases: discovered (or default) list of biases, sorted by numeric value
    - _prefix_map_patterns: compiled from JSON when prefix-mode='map'

    Utilities (now methods)
    -----------------------
    - compute_log_level()           → int
    - configure_logging()           → None
    - ensure_outdir(path=None)      → Path (creates FIGDIR or `path`)
    - system_display_name(label)    → str  (e.g. "Au-001(H)")
    """
    # User inputs
    data_root: Path = Path(".")
    sys_include: str = r".*"
    sys_exclude: Optional[str] = None
    only_au: bool = False
    sites: Optional[list[str]] = None
    heights: Optional[list[str]] = None
    bias_regex: str = r"[+\-]\d+\.\d+V"

    # Prefix resolution
    prefix_mode: str = "auto"  # 'auto'|'rule'|'map'
    prefix_au000: str = "graphene0"
    prefix_default: str = "Au-Graphene0"
    prefix_map_json: Optional[Path] = None

    # Logging flags
    log_level: Optional[str] = None
    verbose: int = 0
    quiet: int = 0
    log_file: Optional[Path] = None

    # Resolved fields
    systems_map: dict[str, Path] | None = None
    biases: list[str] | None = None
    _prefix_map_patterns: list[tuple[re.Pattern, str]] | None = None
    
    # Plotting labels & output dir
    vac_label: str = r"Vacuum/Jellium"
    solv_label: str = r"H$_2$O, 0.1 M HAuCl$_4$"
    figdir: Path = Path("Figures/DoS")

    # Console CSV printing
    show_csv: bool = False
    max_csv_rows: int = 0

    # Argument parsing
    @staticmethod
    def _build_argparser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Reduced DoS & Dirac analysis for ONETEP outputs")

        # Verbosity controls
        p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v=INFO, -vv=DEBUG)")
        p.add_argument("-q", "--quiet", action="count", default=0, help="Decrease verbosity (-q=WARNING, -qq=ERROR)")
        p.add_argument("--log-level", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Explicit log level (overrides -v/-q)")
        p.add_argument("--log-file", type=Path, help="Write logs to this file in addition to stderr")

        # Root discovery
        p.add_argument("--data-root", dest="data_root", type=Path, default=None, help="Root folder containing system folders (preferred)")

        # System filters
        p.add_argument("--sys-include", default=r".*", help="Regex to keep system paths (relative to data_root)")
        p.add_argument("--sys-exclude", default=None, help="Regex to drop system paths")
        p.add_argument("--only-au", action="store_true", help="Keep systems whose top folder starts with 'Au-'")
        p.add_argument("--sites", nargs="*", default=None, help="Allowed site names, e.g. Site-000 Site-005")
        p.add_argument("--heights", nargs="*", default=None, help="Allowed adsorption heights, e.g. 2.8 3.0")

        # Bias directory matching
        p.add_argument("--bias-regex", default=r"[+\-]\d+\.\d+V", help="Regex that matches bias directory names")

        # Prefix resolution
        p.add_argument("--prefix-mode", choices=["auto", "rule", "map"], default="auto", help="How to determine the val_bands prefix per (system,bias)")
        p.add_argument("--prefix-au000", default="graphene0", help="Fallback prefix for Au-000 if --prefix-mode=rule")
        p.add_argument("--prefix-default", default="Au-Graphene0", help="Fallback prefix for other systems if --prefix-mode=rule")
        p.add_argument("--prefix-map-json", type=Path, default=None, help="JSON file with objects {regex, prefix} used if --prefix-mode=map")
        
        # Plotting labels & output dir
        p.add_argument("--vac-label", default=r"Vacuum/Jellium", help="Legend/axis label for the vacuum environment")
        p.add_argument("--solv-label", default=r"H$_2$O, 0.1 M HAuCl$_4$", help="Legend/axis label for the solvated environment")
        p.add_argument("--figdir", type=Path, default=Path("Figures/DoS"), help="Directory to write figures and CSVs")

        # CSV outputs
        p.add_argument("--show-csv", action="store_true", help="Print the CSV tables to console after writing")
        p.add_argument("--max-csv-rows", type=int, default=0, help="Limit rows printed for each CSV (0 = no limit)")

        return p

    @classmethod
    def parse_from_argv(cls, argv: Optional[list[str]] = None) -> "CLIConfig":
        p = cls._build_argparser()
        ns = p.parse_args(argv)
        cfg = cls(
            data_root=ns.data_root,
            sys_include=ns.sys_include,
            sys_exclude=ns.sys_exclude,
            only_au=ns.only_au,
            sites=ns.sites,
            heights=ns.heights,
            bias_regex=ns.bias_regex,
            prefix_mode=ns.prefix_mode,
            prefix_au000=ns.prefix_au000,
            prefix_default=ns.prefix_default,
            prefix_map_json=ns.prefix_map_json,
            log_level=ns.log_level,
            verbose=ns.verbose,
            quiet=ns.quiet,
            log_file=ns.log_file,
            vac_label=ns.vac_label,
            solv_label=ns.solv_label,
            figdir=ns.figdir,
            show_csv=ns.show_csv,
            max_csv_rows=ns.max_csv_rows,
        )
        
        cfg._prepare_prefix_map()
        cfg._discover_systems_and_biases()
        return cfg

    # Logging & filesystem utilities
    def compute_log_level(self) -> int:
        """Compute log level from flags (or explicit --log-level)."""
        if self.log_level: return getattr(logging, self.log_level)
        level = logging.WARNING - 10 * self.verbose + 10 * self.quiet
        level = max(logging.DEBUG, min(logging.CRITICAL, level))
        return level

    def configure_logging(self) -> None:
        """Configure logging (root + clean summary logger)."""
        level = self.compute_log_level()
        handlers = [logging.StreamHandler(sys.stderr)]
        if self.log_file: handlers.append(logging.FileHandler(self.log_file))
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s", datefmt="%H:%M:%S", handlers=handlers, force=True,)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        # Clean summary on stdout
        summary = logging.getLogger("ReducedDoS.summary")
        summary.setLevel(logging.INFO)
        summary.propagate = False
        summary.handlers = []

        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(message)s"))
        summary.addHandler(h)

    def ensure_outdir(self, path: Optional[Path] = None) -> Path:
        """
        Ensure the output directory exists.
        If `path` is None, uses cfg.figdir..
        """
        out = path or self.figdir
        try: out.mkdir(parents=True, exist_ok=True); logging.getLogger(__name__).debug("Ensured output directory exists: %s", out)
        except OSError as e: logging.getLogger(__name__).warning("Failed to create output directory %s: %s", out, e); raise
        return out

    # Display name helper
    @staticmethod
    def system_display_name(sys_label: str) -> str:
        """
        Canonical pretty name from a label like:
            "Au-001/Site-005/2.8"  →  "Au-001(B)"
            "Au-001/Site-002"      →  "Au-001(T)"
            "Au-001/Site-000"      →  "Au-001(H)"
            "Au-000"               →  "Au-000"
        Rules
        -----
        - Base/system root = first path segment (e.g. "Au-000", "Au-001", ...).
        - If base ends with '-000' (e.g. "Au-000", "Si-000"), return base (no suffix).
        - Otherwise, if a 'Site-xxx' segment exists (case-insensitive):
              000 → H, 002 → T, 005 → B
          return "{base}({letter})". Unknown site codes → base only.
        - Heights like '2.8' are ignored.
        """
        
        log = logging.getLogger(__name__)
        log.debug("Resolving display name for system label: %s", sys_label)

        parts = sys_label.split("/")
        if not parts:return sys_label

        base = parts[0]
        if base.endswith("-000"): return base

        # Find a 'Site-xxx' segment (case-insensitive), ignore heights
        m = next((re.match(r"(?i)^site-(\d{3})$", p) for p in parts[1:] if re.match(r"(?i)^site-\d{3}$", p)),None,)
        if not m: return base

        code = m.group(1)
        letter = {"000": "H", "002": "T", "005": "B"}.get(code)
        if not letter:log.debug("Unknown site code '%s' in '%s'; using base only.", code, sys_label); return base
        return f"{base}({letter})"

    # Prefix helpers 
    def _prepare_prefix_map(self) -> None:
        self._prefix_map_patterns = []
        if self.prefix_mode != "map" or not self.prefix_map_json: return
        try:
            obj = json.loads(self.prefix_map_json.read_text())
            if isinstance(obj, dict) and "rules" in obj: obj = obj["rules"]
            if isinstance(obj, list):
                for item in obj:
                    rx = re.compile(item["regex"])
                    self._prefix_map_patterns.append((rx, str(item["prefix"])))
                    
        except Exception as e: logging.getLogger(__name__).warning("Failed to load prefix-map JSON %s: %s", self.prefix_map_json, e)

    def resolve_prefix_for(self, sys_label: str) -> str:
        base = sys_label.split('/')[0]  # "Au-000", "Au-001", ...
        if self.prefix_mode == "map" and self._prefix_map_patterns:
            for rx, pref in self._prefix_map_patterns:
                if rx.search(sys_label):
                    return pref
        if self.prefix_mode == "rule":
            return self.prefix_au000 if base == "Au-000" else self.prefix_default
        # default 'auto' heuristic
        return "graphene0" if base == "Au-000" else "Au-Graphene0"

    # Discovery of systems & biases
    def _discover_systems_and_biases(self) -> None:
        """
        Populate self.systems_map and self.biases by scanning data_root with the filters.
        Accepts both layouts:
            Au-000/<BIAS>/
            Au-001/Site-XXX[/HEIGHT]/<BIAS>/
        """
        root = self.data_root
        keep_rx = re.compile(self.sys_include)
        drop_rx = re.compile(self.sys_exclude) if self.sys_exclude else None
        bias_rx = re.compile(self.bias_regex)

        systems: dict[str, Path] = {}

        # Pattern A: top-level Au-000 (no sites/heights)
        for p in sorted(root.glob("Au-000")):
            if not p.is_dir(): continue
            if self.only_au and not p.name.startswith("Au-"): continue
            label = "Au-000"
            if not keep_rx.search(label): continue
            if drop_rx and drop_rx.search(label): continue
            systems[label] = p

        # Pattern B: Au-001/Site-XXX[/HEIGHT]
        for base in sorted(root.glob("Au-*")):
            if not base.is_dir(): continue
            if self.only_au and not base.name.startswith("Au-"): continue
            if base.name == "Au-000": continue
            for site_dir in sorted(base.glob("Site-*")):
                if not site_dir.is_dir(): continue
                site_name = site_dir.name
                if self.sites and site_name not in self.sites: continue
                # optional height level
                height_dirs = [d for d in site_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d+(\.\d+)?", d.name)]
                if height_dirs:
                    for h in sorted(height_dirs, key=lambda x: float(x.name)):
                        if self.heights and h.name not in self.heights: continue
                        label = f"{base.name}/{site_name}/{h.name}"
                        if not keep_rx.search(label): continue
                        if drop_rx and drop_rx.search(label): continue
                        systems[label] = h
                else:
                    label = f"{base.name}/{site_name}"
                    if not keep_rx.search(label): continue
                    if drop_rx and drop_rx.search(label): continue
                    systems[label] = site_dir

        # Bias discovery = union over all systems
        bias_set: set[str] = set()
        for _, sys_path in systems.items():
            try:sub = [d.name for d in sys_path.iterdir() if d.is_dir() and bias_rx.fullmatch(d.name)]
            except Exception: sub = []
            bias_set.update(sub)

        # Sort biases by numeric value (e.g., "-0.10V" < "+0.00V" < "+0.10V")
        def _bias_key(b: str) -> float:
            try: return float(b.replace("V", ""))
            except Exception: return 0.0

        self.biases = sorted(bias_set, key=_bias_key) if bias_set else ["-0.10V", "+0.00V", "+0.10V"]
        self.systems_map = systems


# Parsing
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


# DoS builder
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


# Dirac energy estimator + UQ
class DiracAnalyser:
    """Extract Dirac point energy and uncertainties."""

    @staticmethod
    def ed_by_minimum(grid_abs: np.ndarray, dos: np.ndarray, ef_eV: Optional[float],
                      window_rel: tuple[float, float] = (-2.0, 2.0)) -> tuple[float, float, int]:
        """
        Estimate Dirac energy E_D as the DoS minimum near EF.

        Definition
        ----------
            Compute E_rel = grid_abs - EF and find the index of the minimum DoS 
            within `window_rel` (default ±2 eV). Return E_D relative to EF and 
            in absolute units.

        Returns
        -------
        (E_D_rel, E_D_abs, idx)
            E_D_rel in eV (may be NaN if EF unknown or window empty),
            E_D_abs in eV, and the grid index of the minimum (-1 if undefined).

        Rationale
        ---------
            The ±2 eV window avoids spurious global minima far from EF caused by 
            band edges or finite sampling.
        """
        logger.debug("ed_by_minimum: EF=%s, window=%s", f"{ef_eV:.4f}" if ef_eV is not None else "None", window_rel)
        
        if ef_eV is None: logger.warning("Cannot estimate E_D: EF is missing"); return np.nan, np.nan, -1
        
        E_rel = grid_abs - ef_eV
        m = (E_rel >= window_rel[0]) & (E_rel <= window_rel[1])
        if not np.any(m): logger.debug("E_D window %s around EF=%.4f eV has no grid coverage", window_rel, ef_eV); return np.nan, np.nan, -1
        
        j_local = int(np.argmin(dos[m]))
        idx = np.nonzero(m)[0][j_local]
        logger.debug("E_D found at index %d: E_rel=%.4f eV, E_abs=%.4f eV", idx, E_rel[idx], grid_abs[idx])
        return float(E_rel[idx]), float(grid_abs[idx]), idx

    @staticmethod
    def ed_se_quadratic(grid_abs: np.ndarray, dos: np.ndarray, ef_eV: float,
                         idx_min: int, halfwidth: float = 0.25) -> float:
        """
        1σ standard error for E_D from a local quadratic model.

        Method
        ------
            Fit y = a x^2 + b x + c to DoS with x = (E - EF) over 
            |x - x_min| ≤ halfwidth, where x_min corresponds to `idx_min`.
            The vertex μ = -b/(2a) estimates E_D_rel. Propagate uncertainty via 
            the delta method using the OLS covariance of (a, b, c).

        Returns
        -------
        float
            1σ standard error for μ in eV, or NaN if the fit is ill-conditioned
            or insufficient data (< 6 points).
        """
        
        logger.debug("Quadratic SE: idx_min=%d, halfwidth=%.3f eV", idx_min, halfwidth)
    
        x = grid_abs - ef_eV
        x0 = float(x[idx_min])
        m = np.abs(x - x0) <= halfwidth
        X = np.column_stack([x[m] ** 2, x[m], np.ones(np.count_nonzero(m))])
        y = dos[m]
        if X.shape[0] < 6: logger.debug("Quadratic SE: insufficient points (%d) within ±%.2f eV", X.shape[0], halfwidth); return np.nan
        
        XtX = X.T @ X
        try: XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError: logger.debug("Quadratic SE: ill-conditioned fit near index %d", idx_min); return np.nan
        
        beta = XtX_inv @ (X.T @ y)
        a, b, _ = beta
        yhat = X @ beta
        dof = max(1, X.shape[0] - X.shape[1])
        s2 = float(np.sum((y - yhat) ** 2) / dof)
        Cov = s2 * XtX_inv
        if a == 0: logger.debug("Quadratic SE: zero curvature (a==0); cannot form vertex SE"); return np.nan
        
        # μ = -b/(2a); propagate var(μ) via delta method
        g = np.array([b / (2 * a * a), -1 / (2 * a), 0.0], float)
        var_mu = float(g @ Cov @ g)
        
        if var_mu < 0: logger.debug("Quadratic SE: negative variance from delta method; returning NaN"); return np.nan
        
        logger.debug("Quadratic SE: 1σ=%.4f eV", np.sqrt(var_mu))
        return np.sqrt(var_mu)

    @staticmethod
    def ed_se_jackknife(data: ValBandsData, grid: np.ndarray, sigma: float,
                        window_rel: tuple[float, float] = (-2.0, 2.0)) -> float:
        """
        1σ jackknife standard error for E_D over leave-one-out k-points.

        Procedure
        ---------
            For each k-point i, drop it, rebuild the DoS (same σ), re-estimate E_D
            via `ed_by_minimum`, and compute the jackknife variance:
            var_jk = ((n - 1) / n) * Σ (E_D^(i) - mean_loo)^2
            Return sqrt(var_jk). Reported uncertainties elsewhere use 2σ = 2 × SE.

        Notes
        -----
            Requires at least 2 k-points and a known EF. Returns NaN otherwise.
        """
        logger.debug("Jackknife SE: n_kpoints=%d, sigma=%.3f, window=%s", len(data.kpoints), sigma, window_rel)
        
        kps: list[KPoint] = data.kpoints
        ef: Optional[float] = data.ef_eV
        n: int = len(kps)
        if ef is None or n < 2: logger.warning("Jackknife SE unavailable: EF missing or too few k-points (n=%d)", n); return np.nan

        # Full estimate (not used in variance directly; kept for clarity)
        E_full, W_full = DoSBuilder.flatten_states(data, align_zero=False, spin=None)
        dos_full = DoSBuilder.gaussian_dos(E_full, W_full, grid, sigma)
        ed_full_rel, _, _ = DiracAnalyser.ed_by_minimum(grid, dos_full, ef, window_rel=window_rel)
        _ = ed_full_rel  # kept to indicate the reference estimate

        # Leave-one-out estimates
        lo: list[float] = []
        for i in range(n):
            sub = ValBandsData(nk=n-1, nspin=data.nspin, neig=data.neig, ef_eV=ef, kpoints=kps[:i] + kps[i+1:])
            E_i, W_i = DoSBuilder.flatten_states(sub, align_zero=False, spin=None)
            dos_i = DoSBuilder.gaussian_dos(E_i, W_i, grid, sigma)
            ed_i_rel, _, _ = DiracAnalyser.ed_by_minimum(grid, dos_i, ef, window_rel=window_rel)
            if np.isfinite(ed_i_rel): lo.append(ed_i_rel)
            
        logger.debug("Jackknife: computed %d/%d estimates", len(lo), n)

        if len(lo) < 2: logger.warning("Jackknife SE undefined: <2 valid leave-one-out estimates"); return np.nan
        lo_arr = np.array(lo, float)
        mean_lo = float(np.mean(lo_arr))
        var_jk = ((n - 1) / n) * float(np.sum((lo_arr - mean_lo) ** 2))
        se = np.sqrt(var_jk)
        logger.debug("Jackknife SE: 1σ=%.4f eV", se)
        return se


# Plotting
class DoSPlotter:
    """Plotting helpers for stacked and comparison panels + ΔE_D summaries."""

    @staticmethod
    def _panel_common_format(ax: plt.Axes) -> None:
        """Common axes cosmetics; EF lines are added per curve (not at 0 eV)."""
        logger.debug("Applying common panel format")
        
        ax.tick_params(which="both", direction="in", top=True, right=False, left=False, labelleft=False)
        ax.xaxis.set_major_locator(MultipleLocator(X_MAJOR_EVERY_EV))
        ax.xaxis.set_minor_locator(MultipleLocator(X_MINOR_EVERY_EV))
        ax.set_xlim(ENERGY_RANGE)
        
    @staticmethod
    def _bias_positions(biases: list[str]) -> tuple[np.ndarray, list[str]]:
        xs = np.arange(len(biases))
        return xs, biases

    @staticmethod
    def _panel_offsets(doses: list[np.ndarray], grid: np.ndarray, sys_label: str) -> tuple[dict[str, float], tuple[float, float]]:
        """
        Choose vertical offsets to avoid overlap in stacked plots.

        Strategy
        --------
        - If NORMALISATION == 'none', derive an offset step from the panel's
          maximum amplitude (x1.10) and scale by a per-system factor so that
          Au-000 lines have extra spacing (empirically clearer).
        - Otherwise, use fixed steps independent of curve amplitudes.

        Returns
        -------
        (bias_offsets, (ymin, ymax))
            A mapping bias→offset and suggested y-limits for the panel.
        """
        
        logger.debug("Computing panel offsets for %s (mode=%s)", sys_label, NORMALISATION)
        sys_base = sys_label.split('/')[0]  # "Au-000" or "Au-001"

        if NORMALISATION == "none":
            panel_max = max((float(np.max(d)) if d.size else 0.0) for d in doses) if doses else 0.0
            if panel_max == 0.0: logger.debug("Panel max is zero for %s; offsets are heuristic", sys_label)
            
            base_step = 1.10 * panel_max if panel_max > 0 else 1.0
            factor = OFFSET_STEP_PER_SYSTEM.get(sys_base, OFFSET_STEP_DEFAULT) / max(OFFSET_STEP_DEFAULT, 1e-12)
            step = base_step * factor
            bias_offsets = {"-0.10V": 0.0, "+0.00V": step, "+0.10V": 2.0 * step}
            y_min = -MARGIN * max(panel_max, 1.0)
            y_max = 2.0 * step + (1.0 + MARGIN) * max(panel_max, 1.0)
            
            logger.debug("Offsets (none): step=%.3f, y=(%.3f, %.3f)", step, y_min, y_max)
            return bias_offsets, (y_min, y_max)

        step = OFFSET_STEP_PER_SYSTEM.get(sys_base, OFFSET_STEP_DEFAULT)
        bias_offsets = {"-0.10V": 0.0, "+0.00V": step, "+0.10V": 2.0 * step}
        y_min = -MARGIN
        y_max = 2.0 * step + 1.0 + MARGIN
        
        logger.debug("Offsets (fixed): step=%.3f, y=(%.3f, %.3f)", step, y_min, y_max)
        return bias_offsets, (y_min, y_max)

    @staticmethod
    def _normalise_panel(doses: list[np.ndarray], grid: np.ndarray, mode: str) -> list[np.ndarray]:
        """'none' (raw), 'panel_max' (shared max), 'area' (∫DOS dE = 1), 'curve_max' (per-curve)."""
        logger.debug("Normalising panel: mode=%s, ncurves=%d", mode, len(doses) if doses else 0)
        
        if not doses: return doses
        if mode == "none": return doses
        
        if mode == "panel_max":
            mx = max(float(np.max(d)) for d in doses) if max(map(np.size, doses)) else 0.0
            if mx <= 0: logger.warning("panel_max normalisation: maximum ≤ 0; leaving curves unscaled")
            return [d / mx if mx > 0 else d for d in doses]
        
        if mode == "area":
            outs: list[np.ndarray] = []
            for d in doses:
                a = float(np.trapz(d, grid))
                if a <= 0: logger.warning("area normalisation: area ≤ 0; leaving curve unscaled")
                outs.append(d / a if a > 0 else d)
            return outs
        
        outs: list[np.ndarray] = []
        for d in doses:
            mx = float(np.max(d)) if d.size else 0.0
            outs.append(d / mx if mx > 0 else d)
            
        logger.debug("Normalisation complete for mode=%s", mode)
        return outs

    @staticmethod
    def plot_vertical_stack(
        env_key: str,
        env_label: str,
        parsed_by_sys: dict[str, dict[str, dict[str, ValBandsData]]],
        systems: dict[str, Path],
        biases: list[str],
        name_fn: Callable[[str], str],
        ensure_outdir: Callable[[], Path],
    ) -> Path:
        """1x4 layout; each subplot shows three biases with EF vertical lines."""
        logger.debug("Plot vertical stack: env=%s", env_key)

        fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_1x4, sharex=True, sharey=True, constrained_layout=False)
        grid = np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], N_GRID)

        legend_handles: list[plt.Line2D] = []
        legend_labels: list[str] = []

        sys_labels = sorted(systems.keys(), key=name_fn)[:4]
        for i, sys_label in enumerate(sys_labels):
            logger.debug("Stack subplot for system: %s", sys_label)
            ax = axes[i]
            ax.set_title(name_fn(sys_label))
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Density of States (arb.)")

            letters = ["(a)", "(b)", "(c)", "(d)"]
            ax.text(0.02, 0.95, letters[i], transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

            DoSPlotter._panel_common_format(ax)

            raw_doses: list[np.ndarray] = []
            ok_biases: list[str] = []
            ef_map: dict[str, Optional[float]] = {}

            for bias in biases:
                ds = parsed_by_sys.get(sys_label, {}).get(bias, {}).get(env_key)
                if not ds or not ds.kpoints: logger.warning("Missing dataset: system=%s bias=%s env=%s", sys_label, bias, env_key); continue

                E, W = DoSBuilder.flatten_states(ds, align_zero=False, spin=None)
                DOS = DoSBuilder.gaussian_dos(E, W, grid, BROADENING_EV)
                raw_doses.append(DOS)
                ok_biases.append(bias)
                ef_map[bias] = ds.ef_eV

            logger.debug("Plotted %d biases for %s (%s)", len(ok_biases), sys_label, env_key)
            doses = DoSPlotter._normalise_panel(raw_doses, grid, NORMALISATION)
            bias_offsets, (y_min, y_max) = DoSPlotter._panel_offsets(doses, grid, sys_label)

            for DOS, bias in zip(doses, ok_biases):
                (line,) = ax.plot(grid, DOS + bias_offsets[bias], color=BIAS_COLOUR.get(bias, COLOURS["black"]), label=bias,)
                ef_here = ef_map.get(bias)
                if ef_here is not None and np.isfinite(ef_here): ax.axvline(float(ef_here), color=BIAS_COLOUR.get(bias, COLOURS["black"]), lw=1.0, ls="--", alpha=0.8,)
                if i == 0:legend_handles.append(line); legend_labels.append(bias)

            for bias in biases:
                if bias not in ok_biases: ax.text(0.5, 0.5 - 0.15 * biases.index(bias), f"No data {bias}", transform=ax.transAxes, 
                                                  ha="center", va="center", fontsize=9, color="red",
                                                )
            ax.set_ylim(y_min, y_max)

        fig.legend(legend_handles, legend_labels, ncol=3, frameon=True, facecolor="white", edgecolor="0.3", 
                   loc="upper center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.4,
                   )

        fig.subplots_adjust(top=0.86, left=0.07, right=0.98, bottom=0.18, wspace=0.15)
        outdir = ensure_outdir()
        out = outdir / f"Stacked_DoS_{env_key}.png"
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved vertical stack plot: %s", out)
        return out

    @staticmethod
    def plot_2x2_comparison(
        parsed_by_sys: dict[str, dict[str, dict[str, ValBandsData]]],
        biases: list[str],
        name_fn: Callable[[str], str],
        ensure_outdir: Callable[[], Path],
        vac_label: str,
        solv_label: str,
    ) -> Path:
        """
        2x2 grid:
            Columns = (vac_label, solv_label)
            Rows    = (two representative systems, e.g. Au-000 and Au-001(B))
        """
        logger.debug("Plot 2x2 comparison")
        available:list = list(parsed_by_sys.keys())
        prefer = ["Au-000", "Au-001/Site-005/2.8"]
        row_systems = [s for s in prefer if s in available]
        for s in available:
            if len(row_systems) >= 2: break
            if s not in row_systems: row_systems.append(s)
            
        row_systems = row_systems[:2]
        
        col_envs: list[tuple[str, str]] = [("vac", vac_label), ("solv", solv_label)]

        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2x2, sharex=True, sharey=True, constrained_layout=False)
        grid = np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], N_GRID)

        for j, (_, env_label) in enumerate(col_envs): axes[0, j].set_title(env_label)

        for i, sys_key in enumerate(row_systems): axes[i, 0].text(-0.20, 0.5, name_fn(sys_key), transform=axes[i, 0].transAxes,
                                                                  ha="right", va="center", rotation=90, fontsize=11,
                                                                 )

        legend_handles: list[plt.Line2D] = []
        legend_labels: list[str] = []

        for i, sys_key in enumerate(row_systems):
            for j, (env_key, _) in enumerate(col_envs):
                logger.debug("2x2 subplot: system=%s env=%s", sys_key, env_key)
                ax = axes[i, j]
                letters = [["(a)", "(b)"], ["(c)", "(d)"]]
                ax.text(0.02, 0.95, letters[i][j], transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

                DoSPlotter._panel_common_format(ax)

                raw_doses: list[np.ndarray] = []
                ok_biases: list[str] = []
                ef_map: dict[str, Optional[float]] = {}

                for bias in biases:
                    ds = parsed_by_sys.get(sys_key, {}).get(bias, {}).get(env_key)
                    if not ds or not ds.kpoints: logger.warning("Missing dataset: system=%s bias=%s env=%s", sys_key, bias, env_key); continue

                    E, W = DoSBuilder.flatten_states(ds, align_zero=False, spin=None)
                    DoS = DoSBuilder.gaussian_dos(E, W, grid, BROADENING_EV)
                    raw_doses.append(DoS)
                    ok_biases.append(bias)
                    ef_map[bias] = ds.ef_eV

                doses = DoSPlotter._normalise_panel(raw_doses, grid, NORMALISATION)
                bias_offsets, (y_min, y_max) = DoSPlotter._panel_offsets(doses, grid, sys_key)

                for DoS, bias in zip(doses, ok_biases):
                    (line,) = ax.plot(grid, DoS + bias_offsets[bias], color=BIAS_COLOUR.get(bias, COLOURS["black"]), label=bias,)
                    ef_here = ef_map.get(bias)
                    if ef_here is not None and np.isfinite(ef_here): ax.axvline(float(ef_here), color=BIAS_COLOUR.get(bias, COLOURS["black"]),
                                                                                lw=1.0, ls="--", alpha=0.8
                                                                                )

                    if (i, j) == (0, 0): legend_handles.append(line); legend_labels.append(bias)

                for bias in biases:
                    if bias not in ok_biases: ax.text(0.5, 0.5 - 0.15 * biases.index(bias), f"No data {bias}", transform=ax.transAxes,
                                                      ha="center", va="center", fontsize=9, color="red",
                                                      )
                ax.set_ylim(y_min, y_max)
                if i == 1: ax.set_xlabel("Energy (eV)")
                if j == 0: ax.set_ylabel("Density of States (arb.)")

        fig.legend(legend_handles, legend_labels, ncol=3, frameon=True, facecolor="white",  edgecolor="0.3", 
                   loc="upper center", bbox_to_anchor=(0.5, 1.02), borderaxespad=0.4,
                   )

        fig.subplots_adjust(top=0.90, left=0.05, right=0.98, bottom=0.12, wspace=0.15, hspace=0.15)
        outdir = ensure_outdir()
        out = outdir / "Stacked_DoS_2x2_Vac_vs_Solv_Au000_vs_Au001B.png"
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2x2 comparison plot: %s", out)
        return out

    @staticmethod
    def plot_delta_dirac_series(
        delta_records: list[dict[str, float]],
        biases: list[str],
        name_fn: Callable[[str], str],
        ensure_outdir: Callable[[], Path],
    ) -> Path:
        logger.debug("Plot ΔE_D series: nrecords=%d", len(delta_records))
        
        sys_labels = sorted({r["system"] for r in delta_records}, key=name_fn)
        sys_labels.sort(key=lambda s: (0 if s.startswith("Au-000") else 1, name_fn(s)))
        if not sys_labels:
            logger.warning("No ΔE_D records; skipping series plot")
            out = ensure_outdir() / "Delta_E_D_vs_Bias_series.png"
            plt.figure(); plt.savefig(out); plt.close(); return out

        sys_order = sys_labels[:4]
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DELTA_SERIES, constrained_layout=False, sharey=True)
        xs, xticklabels = DoSPlotter._bias_positions(biases)
        table = {(r["system"], r["bias"]): r for r in delta_records}

        for idx, sys_key in enumerate(sys_order):
            ax = axes[idx // 2, idx % 2]
            letters = [["(a)", "(b)"], ["(c)", "(d)"]]
            ax.text(0.02, 0.95, letters[idx // 2][idx % 2], transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

            ys: list[float] = []
            yerr: list[float] = []
            for b in biases:
                rec = table.get((sys_key, b))
                if rec is None: ys.append(np.nan); yerr.append(0.0)
                else: ys.append(float(rec["Delta_E_D_eV"])); yerr.append(float(rec["Delta_unc_2sigma_eV"]))

            for x, y, e, b in zip(xs, ys, yerr, biases):
                ax.errorbar(x, y, yerr=e, fmt="o-", lw=1.4, ms=4, color=BIAS_COLOUR.get(b, COLOURS["black"]), capsize=3)

            ax.axhline(0.0, color="0.4", lw=1.0, ls="--")
            ax.set_xticks(xs, xticklabels)
            ax.tick_params(which="both", direction="in", top=True)
            ax.set_title(name_fn(sys_key))
            if idx // 2 == 1: ax.set_xlabel("Applied bias (V)")
            if idx % 2 == 0: ax.set_ylabel(r"$\Delta E_D$ (eV)")

        fig.subplots_adjust(top=0.92, left=0.08, right=0.98, bottom=0.12, wspace=0.20, hspace=0.25)
        outdir = ensure_outdir()
        out = outdir / "Delta_E_D_vs_Bias_series.png"
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved ΔE_D series plot: %s", out)
        return out

    @staticmethod
    def plot_delta_dirac_heatmap(
        delta_records: list[dict[str, float]],
        biases: list[str],
        name_fn: Callable[[str], str],
        ensure_outdir: Callable[[], Path],
    ) -> Path:
        logger.debug("Plot ΔE_D heatmap: nrecords=%d", len(delta_records))
        sys_labels = sorted({r["system"] for r in delta_records}, key=name_fn)
        sys_order = sorted(sys_labels, key=lambda s: (0 if s.startswith("Au-000") else 1, name_fn(s)))
        
        if not sys_order:
            logger.warning("No ΔE_D records; skipping heatmap")
            out = ensure_outdir() / "Delta_E_D_heatmap.png"
            plt.figure(); plt.savefig(out); plt.close(); return out
            
        n_sys, n_bias = len(sys_order), len(biases)
        Z = np.full((n_sys, n_bias), np.nan, float)
        U = np.full((n_sys, n_bias), np.nan, float)

        index = {(r["system"], r["bias"]): r for r in delta_records}
        for i, s in enumerate(sys_order):
            for j, b in enumerate(biases):
                rec = index.get((s, b))
                if rec: Z[i, j] = float(rec["Delta_E_D_eV"]); U[i, j] = float(rec["Delta_unc_2sigma_eV"])

        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_DELTA_HEAT, constrained_layout=False)
        vmax = np.nanmax(np.abs(Z))
        if not np.isfinite(vmax) or vmax == 0: logger.warning("Heatmap ΔE_D has zero/NaN range; using fallback vmax=0.1"); vmax = 0.1

        im = ax.imshow(Z, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=+vmax)
        for i in range(n_sys):
            for j in range(n_bias):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i - 0.10, f"{Z[i,j]:+.2f}", ha="center", va="center", fontsize=10, color="k")
                    ax.text(j, i + 0.25, r"$\pm$" + f"{U[i,j]:.2f}", ha="center", va="center", fontsize=9, color="k")

        ax.set_xticks(np.arange(n_bias), biases)
        ax.set_yticks(np.arange(n_sys), [name_fn(s) for s in sys_order])
        ax.tick_params(which="both", direction="in", top=False, right=False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\Delta E_D$ (eV)")

        fig.subplots_adjust(top=0.95, left=0.20, right=0.98, bottom=0.12)
        outdir = ensure_outdir()
        out = outdir / "Delta_E_D_heatmap.png"
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved ΔE_D heatmap: %s", out)
        return out


# Pipeline
class ReducedDoSPipeline:
    """End-to-end pipeline: parse → build DoS → Dirac/UQ → plots + CSV."""

    def __init__(self, cfg: CLIConfig) -> None:
        self.cfg = cfg
        self.parser = ValBandsParser()
    
    @staticmethod
    def _pm(x: str, u: str) -> str:
        """Return 'x±u' with blanks handled; expects already-formatted strings."""
        if not x: return "—"
        if not u: return f"{x}±—"
        return f"{x}±{u}"
    
    @staticmethod
    def _fmt(x: Optional[float]) -> str:
            if x is None: return ""
            if isinstance(x, float) and not np.isfinite(x): return ""
            return f"{float(x):.4f}"
            
    def _print_csv_table(self, title: str, header: list[str], rows: list[list[str]], max_rows: int) -> None:
        """Pretty-print a small CSV-like table to the summary logger."""
        summary = logging.getLogger("ReducedDoS.summary")
        summary.info("\n%s", title)

        body = rows
        if max_rows and len(body) > max_rows: body = body[:max_rows]; clipped = True
        else: clipped = False

        # Compute simple column widths
        cols = list(zip(*([header] + body))) if body else [header]
        widths = [max(len(str(x)) for x in col) for col in cols]
        fmt = " | ".join("{:%d}" % w for w in widths)

        summary.info(fmt.format(*header))
        summary.info("-+-".join("-" * w for w in widths))
        for r in body: summary.info(fmt.format(*r))
        if clipped: summary.info("… (showing %d of %d rows)", len(body), len(rows))

    def run(self) -> None:
        """
        Execute the full pipeline:
            parse → DoS (σ=0.10 eV) → Dirac + UQ (±2 eV window) → plots → CSV.

        Side effects
        ------------
        - Writes figures into cfg.figdir.
        - Writes 'UQ_results.csv' and 'Dirac_deltas.csv' into cfg.figdir.
        - Prints a compact textual summary (use logging in batch runs).

        Failure modes
        -------------
        Missing inputs or unknown EF yield blank (NaN) metrics and on-figure
        'No data' annotations; the pipeline continues for other cases.
        """
        logger.debug("Config: σ=%.3f eV, N_GRID=%d, ENERGY_RANGE=%s, NORMALISATION=%s, FIGDIR=%s",
                     BROADENING_EV, N_GRID, ENERGY_RANGE, NORMALISATION, self.cfg.figdir)
        
        parsed_all: dict[str, dict[str, dict[str, ValBandsData]]] = {}
        
        # CSV Headers
        metrics_header = ["system","env","bias","E_D_rel_eV","E_D_abs_eV","ED_uq_2sigma_eV",
                          "ED_se_jack_1sigma_eV","ED_se_quad_1sigma_eV","FWHM_near_EF_eV"]
        delta_header = ["system","bias", "Delta_E_D_eV","Delta_unc_2sigma_eV","ED_vac_rel_eV",
                        "ED_vac_unc_2sigma_eV","ED_solv_rel_eV","ED_solv_unc_2sigma_eV"]
        
        metrics_table: list[list[str]] = [metrics_header]
        delta_table: list[list[str]] = [delta_header]

        # for ΔE_D table: (sys,bias) → {env: (ED_rel, 2σ_UQ)}
        dirac_store: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}

        # Parse & compute per-env metrics
        for sys_label, root in self.cfg.systems_map.items():
            parsed_all[sys_label] = {}
            for bias in self.cfg.biases:
                parsed_all[sys_label][bias] = {}
                base = root / bias
                prefix = self.cfg.resolve_prefix_for(sys_label)

                vac_path  = base / f"{prefix}_vacuum.val_bands"
                solv_path = base / f"{prefix}.val_bands"

                logger.debug("Processing system=%s bias=%s (prefix=%s)", sys_label, bias, prefix)
                logger.debug("Paths: vac=%s | solv=%s", vac_path, solv_path)

                def _try(path: Path) -> Optional[ValBandsData]:
                    try: return self.parser.parse(path)
                    except FileNotFoundError: logger.debug("Missing val_bands file: %s", path); return None

                ds_vac = _try(vac_path)
                ds_solv = _try(solv_path)

                # EF fallback from logs if missing
                for env_key, ds in (("vac", ds_vac), ("solv", ds_solv)):
                    if ds is not None and ds.ef_eV is None:
                        ef_fb = self.parser.fallback_ef_from_folder(base)
                        if ef_fb is not None: ds.ef_eV = ef_fb; logger.debug("EF recovered from logs for %s/%s: %.4f eV", sys_label, bias, ef_fb)
                        else: logger.warning("EF not found for %s/%s (%s); metrics will be blank.", sys_label, bias, env_key)

                if ds_vac: parsed_all[sys_label][bias]["vac"] = ds_vac
                if ds_solv: parsed_all[sys_label][bias]["solv"] = ds_solv

                # Metrics
                grid = np.linspace(ENERGY_RANGE[0], ENERGY_RANGE[1], N_GRID)
                dirac_store[(sys_label, bias)] = {}

                for env_key, ds in (("vac", ds_vac), ("solv", ds_solv)):
                    if not ds or not ds.kpoints: metrics_table.append([sys_label, env_key, bias, "", "", "", "", "", ""]); continue

                    E, W = DoSBuilder.flatten_states(ds, align_zero=False, spin=None)
                    dos = DoSBuilder.gaussian_dos(E, W, grid, BROADENING_EV)

                    # Dirac minimum & FWHM near EF
                    E_D_rel, E_D_abs, idx_min = DiracAnalyser.ed_by_minimum(grid, dos, ds.ef_eV, window_rel=(-2.0, 2.0))
                    fwhm = self._find_fwhm_near_EF(grid, dos, ds.ef_eV, window_rel=(-1.0, 1.0))

                    # Uncertainties (1σ): jackknife + quadratic; main 2σ uses jackknife
                    ed_se_jack = DiracAnalyser.ed_se_jackknife(ds, grid, BROADENING_EV, window_rel=(-2.0, 2.0))
                    ed_se_quad = (DiracAnalyser.ed_se_quadratic(grid, dos, float(ds.ef_eV), idx_min, halfwidth=0.25) 
                                  if (ds.ef_eV is not None and idx_min >= 0 and np.isfinite(E_D_rel))
                                  else np.nan
                                 )
                    ed_uq_2sigma = 2.0 * ed_se_jack if np.isfinite(ed_se_jack) else np.nan

                    metrics_table.append([ sys_label, env_key, bias, self._fmt(E_D_rel), self._fmt(E_D_abs), self._fmt(ed_uq_2sigma), 
                                          self._fmt(ed_se_jack), self._fmt(ed_se_quad), self._fmt(fwhm),]
                                        )

                    if np.isfinite(E_D_rel): dirac_store[(sys_label, bias)][env_key] = ( float(E_D_rel), float(ed_uq_2sigma) if np.isfinite(ed_uq_2sigma) else np.nan,)
        
        # Build ΔE_D = E_D(solv) - E_D(vac) with 2σ uncertainty in quadrature
        for (sys_label, bias), env_map in dirac_store.items():
            if "vac" in env_map and "solv" in env_map:
                ed_vac, u2_vac = env_map["vac"]
                ed_solv, u2_solv = env_map["solv"]
                delta = ed_solv - ed_vac
                
                if np.isfinite(u2_vac) and np.isfinite(u2_solv): u2_delta = float(np.sqrt(u2_vac ** 2 + u2_solv ** 2))
                else: u2_delta = np.nan
                
                delta_table.append([sys_label, bias, self._fmt(delta), self._fmt(u2_delta), self._fmt(ed_vac), 
                                    self._fmt(u2_vac), self._fmt(ed_solv), self._fmt(u2_solv),])

        # Save plots
        DoSPlotter.plot_vertical_stack("vac", self.cfg.vac_label, parsed_all, systems=self.cfg.systems_map, biases=self.cfg.biases, 
                                       name_fn=CLIConfig.system_display_name, ensure_outdir=lambda: self.cfg.ensure_outdir(),
                                      )

        DoSPlotter.plot_vertical_stack("solv", self.cfg.solv_label, parsed_all, systems=self.cfg.systems_map, biases=self.cfg.biases, 
                                       name_fn=CLIConfig.system_display_name, ensure_outdir=lambda: self.cfg.ensure_outdir(),
                                      )

        DoSPlotter.plot_2x2_comparison(parsed_by_sys=parsed_all, biases=self.cfg.biases, name_fn=CLIConfig.system_display_name, 
                                       ensure_outdir=lambda: self.cfg.ensure_outdir(), vac_label=self.cfg.vac_label, solv_label=self.cfg.solv_label,
                                      )

        # Write CSVs
        self.cfg.ensure_outdir()
        csv_metrics = self.cfg.figdir / "UQ_results.csv"
        csv_deltas  = self.cfg.figdir / "Dirac_deltas.csv"
        try:
            with open(csv_metrics, "w", newline="") as f: csv.writer(f).writerows(metrics_table)
            with open(csv_deltas, "w", newline="") as f: csv.writer(f).writerows(delta_table)
        except OSError as e: logger.warning("Failed writing CSVs into %s: %s", self.cfg.figdir, e); raise

        # Parse delta rows back to records for ΔE_D figures
        delta_records: list[dict[str, float]] = []
        for row in delta_table[1:]:
            sys_label, bias, dED, u2, ev, uv2, es, us2 = (row + [""] * 8)[:8]
            
            def _f(x: str) -> float:
                try: return float(x)
                except Exception: return float("nan")
                
            delta_records.append({
                "system": sys_label,
                "bias": bias,
                "Delta_E_D_eV": _f(dED),
                "Delta_unc_2sigma_eV": _f(u2),
                "ED_vac_rel_eV": _f(ev),
                "ED_vac_unc_2sigma_eV": _f(uv2),
                "ED_solv_rel_eV": _f(es),
                "ED_solv_unc_2sigma_eV": _f(us2),
            })
            
        logger.debug("Parsed %d ΔE_D records for plotting", len(delta_records))
            
        # Console summaries
        summary = logging.getLogger("ReducedDoS.summary")
        summary.info("Dirac energies relative to EF (eV):")
        midx = {(r[0], r[1], r[2]): (r[3], r[5]) for r in metrics_table[1:]}

        env_order = ("vac", "solv")
        for sys_label in self.cfg.systems_map.keys():
            sys_disp = CLIConfig.system_display_name(sys_label)
            summary.info(f"\n[{sys_disp}]")
            for env in env_order:
                for bias in self.cfg.biases:
                    ED_rel, u2 = midx.get((sys_label, env, bias), ("", ""))
                    if ED_rel or u2: summary.info(f"  {env:4s} | {bias:6s} : {self._pm(ED_rel, u2):15s} eV")

        summary.info("\nΔE_D = E_D(solv) − E_D(vac) (eV):")
        for row in delta_table[1:]:
            sys_label, bias, dED, u2, ev, uv2, es, us2 = row
            sys_disp = CLIConfig.system_display_name(sys_label)
            vac_s  = self._pm(ev,  uv2)
            solv_s = self._pm(es, us2)
            summary.info(f" {sys_disp:9s} | {bias:6s} : {self._pm(dED,u2):15s} eV  [vac {vac_s:15s}; solv {solv_s}]")

        DoSPlotter.plot_delta_dirac_series(delta_records,biases=self.cfg.biases,name_fn=CLIConfig.system_display_name,
                                           ensure_outdir=lambda: self.cfg.ensure_outdir(),
                                           )
        
        DoSPlotter.plot_delta_dirac_heatmap(delta_records,biases=self.cfg.biases,name_fn=CLIConfig.system_display_name,
                                            ensure_outdir=lambda: self.cfg.ensure_outdir(),
                                            )

        logger.info("Saved reduced DoS figures and CSVs:")
        logger.info(" - %s", csv_metrics.resolve())
        logger.info(" - %s", csv_deltas.resolve())

    @staticmethod
    def _interp_x(x0: float, y0: float, x1: float, y1: float, yt: float) -> Optional[float]:
        """Linear interpolation x(y) between (x0,y0) and (x1,y1)."""
        
        if y1 == y0: logger.debug("Interpolation degenerate (y1==y0); returning None"); return None
        t = (yt - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    def _find_fwhm_near_EF(self, grid_abs: np.ndarray, dos: np.ndarray, ef_eV: Optional[float],
                           window_rel: tuple[float, float] = (-1.0, 1.0)) -> float:
        """
        If a peak exists within ±1 eV of EF, compute its FWHM using linear interpolation.
        #This measures peak width near EF as a proxy for spectral sharpness; unrelated to the Dirac dip width.
        
        Returns FWHM in eV or NaN if not well-defined.
        """
        logger.debug("FWHM: window=%s", window_rel)
        
        if ef_eV is None: logger.debug("FWHM: EF missing; NaN"); return np.nan
        
        E_rel = grid_abs - ef_eV
        m = (E_rel >= window_rel[0]) & (E_rel <= window_rel[1])
        if not np.any(m): logger.debug("FWHM: no samples within window around EF=%.4f eV", ef_eV); return np.nan
        
        xw = E_rel[m]
        yw = dos[m]
        jpk = int(np.argmax(yw))
        ypk = float(yw[jpk])
        
        if ypk <= 0: logger.debug("FWHM: non-positive peak height near EF; NaN"); return np.nan
        
        yhalf = 0.5 * ypk

        # Left half-max
        xl: Optional[float] = None
        for k in range(jpk - 1, -1, -1):
            if (yw[k] - yhalf) * (yw[k + 1] - yhalf) <= 0:
                xl = self._interp_x(float(xw[k]), float(yw[k]), float(xw[k + 1]), float(yw[k + 1]), yhalf)
                break
            
        # Right half-max
        xr: Optional[float] = None
        for k in range(jpk, len(xw) - 1):
            if (yw[k] - yhalf) * (yw[k + 1] - yhalf) <= 0:
                xr = self._interp_x(float(xw[k]), float(yw[k]), float(xw[k + 1]), float(yw[k + 1]), yhalf)
                break

        if xl is None or xr is None: logger.debug("FWHM: could not bracket half-maximum; NaN"); return np.nan
        
        width = float(xr - xl)
        logger.debug("FWHM: %.4f eV", width)
        return width

# Main routine: Call ReducedDoSPipeline
def main() -> None:
    cfg = CLIConfig.parse_from_argv()
    cfg.configure_logging()

    log = logging.getLogger(__name__)
    log.debug("Logger initialised at level %s", logging.getLevelName(log.getEffectiveLevel()))
    log.info("Writing outputs to %s", cfg.figdir)

    ReducedDoSPipeline(cfg).run()

if __name__ == "__main__":
    main()