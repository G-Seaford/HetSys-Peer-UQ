# HetSys-Peer-UQ: Dirac Energy Analysis and Uncertainty Quantification

This repository contains the code and workflow developed for the **PX915 Peer-to-Peer UQ exercise** and forms part of the reproducibility and uncertainty quantification (UQ) sections of my **HetSys Summer Project**.  

The project analyses **ONETEP outputs** to:
- Parse valence band eigenvalues,
- Build Gaussian-broadened densities of states (DoS),
- Extract **Dirac point energies (E_D)**,
- Quantify uncertainties using **jackknife resampling** and quadratic fits,
- Generate **ΔE_D heatmaps**, **bias series plots**, and **2×2 comparison plots**.

The full methodology and results are discussed in the final report included in this repository:
> *Summer_Project_Report.pdf*

---

## Repository Structure

HetSys-Peer-UQ/
├── Data/ # Main data directory
│   ├── Au-000/
│   │   ├── -0.10V/
│   │   ├── +0.00V/
│   │   └── +0.10V/
│   │       ├── $modules_list
│   │       ├── graphene.dat
│   │       ├── graphene.out
│   │       ├── graphene0.onetep
│   │       ├── graphene0.val_bands
│   │       ├── graphene0.xyz
│   │       └── graphene0_vacuum.val_bands
│   └── Au-001/
│       ├── Site-000/
│       ├── Site-002/
│       └── Site-005/
│           ├── -0.10V/
│           ├── +0.00V/
│           └── +0.10V/
│               ├── $modules_list
│               ├── Au-Graphene.dat
│               ├── Au-Graphene.out
│               ├── Au-Graphene0.onetep
│               ├── Au-Graphene0.val_bands
│               ├── Au-Graphene0.xyz
│               └── Au-Graphene0_vacuum.val_bands
├── Figures/ # Selected figures and files from the report for comparison
│   ├── Dirac_deltas.csv
│   ├── Stacked_DoS_2x2_Vac_vs_Solv_Au000_vs_Au001B.png
│   ├── Stacked_DoS_solv.png
│   ├── Stacked_DoS_vac.png
│   └── UQ_results.csv
│ 
└── Summer_Project_Report.pdf

---
Files present:

`$modules_list` : Auto-generated file containing modules loaded for ONETEP calculation
`*.dat`: ONETEP input file
`*.out`: ONETEP output file (contains energies only)
`*0.onetep`: ONETEP log file (contains all other properties like forces)
`*0.val_bands`: File containing the solvated eigenenergies for DoS calculations
`*0_vacuum.val_bands`: File containing vacuum eigenenergies for DoS calculations
`*0.xyz`: Structure Geometry

`Dirac_deltas.csv`: A file containing the data used to produce Table II (Dirac Energies) in `Summer_Project_Report.pdf`

---

## 🔧 Installation and Environment Setup

We recommend using [`uv`](https://github.com/astral-sh/uv) to ensure a reproducible Python environment.

### 1. Install `uv`
If not already installed:
```bash
pip install uv
```

From the root of the repository, please run