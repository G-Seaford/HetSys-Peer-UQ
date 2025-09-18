#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import argparse, json, logging, sys, re

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

    Utilities
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
        p.add_argument("--data-root", dest="data_root", type=Path, default=Path("."), help="Root folder containing system folders (preferred)")

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