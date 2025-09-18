#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Optional, Callable

from data_structures import ValBandsData
from dos_builder import DoSBuilder
from parameters import (X_MAJOR_EVERY_EV, X_MINOR_EVERY_EV, ENERGY_RANGE, N_GRID, 
                       BROADENING_EV, OFFSET_STEP_DEFAULT, OFFSET_STEP_PER_SYSTEM, 
                       MARGIN, NORMALISATION, FIGSIZE_1x4, FIGSIZE_2x2, DPI_SAVE, 
                       COLOURS, BIAS_COLOUR, logger)

import  numpy as np, matplotlib.pyplot as plt

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
        out = outdir / "Stacked_DoS_2x2.png"
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2x2 comparison plot: %s", out)
        return out