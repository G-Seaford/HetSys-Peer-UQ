# HetSys-Peer-UQ: Dirac Energy Analysis and Uncertainty Quantification

This repository contains the code and workflow developed for the **PX915 Peer-to-Peer UQ exercise** and forms part of the reproducibility and uncertainty quantification (UQ) sections of my **HetSys Summer Project**. 

## Scope & Outcomes:
The aim for this repository is to aid in the reproduction of the Dirac energy analysis and UQ from the Summer Project using the provided ONETEP `.val_bands` data. In particular, the code provided has a full workflow, as seen in `main.py`, and additional python classes that can be used to reconstruct the workflow. These additional classes, as well as parameters used in the main script, can be found in the `Provided Code` subdirectory.

The goal is to reproduce the Dirac energies, with the uncertainty measure, outlined in Table II. This involves constructing the reduced density of states (DoS) curves (see `Figures/Stacked_DoS_2x2.png`), extracting the Dirac energy (relative to the Fermi level), and providing a rudimentary error quantification using a leave-one-out jackknife over the k-points.

> ⚠️ The jackknife is used here to quantify sensitivity of E_D to k-point omission. K-point weights are respected since leave-one-out is done at the k-point block level. This gives an internally-consistent 'uncertainty' for the given mesh (4×4×1 MP reduced to symmetry-distinct points). It does not assume strict statistical independence of k-points, but it does assume the estimator varies smoothly with the dataset and that the reduced mesh is a reasonable surrogate for the full Brillouin-zone average. This is **NOT**, in the strict sense, an uncertainty measure nor a true statistical confidence interval, but more of a *sensitivity diagnostic*.

From the *ONETEP* `.val_bands` files you should:
1.	Parse valence eigenvalues and k-point weights (with EF fallback from logs when missing).
2.	Build Gaussian-broadened densities of states (DoS) on a fixed energy grid.
3.	Extract the Dirac point energy $(E_D)$ as the DoS minimum within ±2 eV of E_F.
4.	Quantify uncertainty in $(E_D)$ using leave-one-out jackknife over k-points (report 2σ). A local quadratic fit near the minimum is computed as a diagnostic 1σ alternative.
5.	Generate figures:
	-	Stacked_DoS_2x2.png (comparison panel)
6.	Write CSVs:
	-	Uncertainty_Quantification_Values.csv (per-case metrics)
	-	Dirac_Energies.csv (ΔE_D table used for the report)

The full methodology and results are discussed in `Summer_Project_Report.pdf`.

---

## 📂 Repository Structure

For this diagram, the files and subdirectories that contain the virtual environment (`.venv`) and the setup for `uv` are omitted. A brief overview of the files found in this repo is available below.

```bash
HetSys-Peer-UQ/
├── Data/ # Main data directory.
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
│       ├── Site-000/ # H symmetry site
│       ├── Site-002/ # T symmetry site
│       └── Site-005/ # B symmetry site
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
├── Figures/ # Selected figures and files from the report for comparison.
│   ├── Dirac_Energies.csv
│   ├── Stacked_DoS_2x2.png
│   ├── Stacked_DoS_solv.png
│   ├── Stacked_DoS_vac.png
│   └── Uncertainty_Quantification_Values.csv
├── Provided Code/
│   ├── cli_config.py       # CLIConfig
│   ├── data_structures.py  # KPoint, ValBandsData
│   ├── dos_builder.py      # DoSBuilder
│   ├── dos_plotter.py      # DoSPlotter (optional to use)
│   ├── parameters.py       # System Parameters 
│   └── val_bands_parser.py # ValBandsParser
├── main.py
├── README.md # This file
└── Summer_Project_Report.pdf
```

### 📂 Files present:

##### Data Subdirectory:

- `$modules_list` : Auto-generated file containing modules loaded for ONETEP calculations.
- `*.dat`: ONETEP input file.
- `*.out`: ONETEP output file containing the grand-canonical free-energy for the vacuum and solvated systems.
- `*0.onetep`: ONETEP log file for the GC-eDFT run (contains all other properties like forces).
- `*0.val_bands`: File containing the solvated eigenenergies for DoS calculations in Hartree.
- `*0_vacuum.val_bands`: File containing vacuum eigenenergies for DoS calculations in Hartree.
- `*0.xyz`: Structure Geometry

##### Figures Subdirectory:
- `Dirac_Energies.csv`: A file containing the data used to produce Table II (Dirac Energies) in `Summer_Project_Report.pdf`.
- `Stacked_DoS_2x2.png`: The DoS plot used in Figure 2 in `Summer_Project_Report.pdf`.
- `Stacked_DoS_solv.png`: DoS plot for the solvated systems.
- `Stacked_DoS_vac.png`: DoS plot for the vacuum systems.
- `Uncertainty_Quantification_Values.csv` : A file containing differing UQ applied to the Dirac energy calculations.

##### Provided Code Subdirectory:
- `cli_config.py` : The CLIConfig class for command line arguments and I/O settings.
- `data_structures.py` : The KPoints and ValBandsData dataclasses used for data storage.
- `dos_builder.py` : The DoSBuilder class for constructing the DoS curves from the parsed .val_bands files
- `dos_plotter.py` : The DoSPlotter class for plotting the DoS curves.
- `parameters.py` : The parameters used in the `main.py` script, including matplotlib settings.
- `val_bands_parser.py` : The ValBandsParser class to read and store the data from .val_bands files.

##### Repository Root:
- `main.py` : A full pipeline for calculating the DoS and Dirac energies (with UQ) from the `.val_bands` files.
- `Summer_Project_Report.pdf` : A copy of the submitted summer project report.

---

## 🔧 Installation and Environment Setup

To run code in a self-contained virtual environment, I recommend using [`uv`](https://github.com/astral-sh/uv) to ensure a reproducible Python environment. The following steps will assume that the `uv` environment will be used.

### 1. Install `uv` and Initialise Environment:
If not already installed, please run the following commands:
```bash
pip install uv
cd <path_to_cloned_repo>
uv sync
```
This will first install the `uv` environment via pip and sync the environment using the packages listed in `requirements.txt`. If uv is already installed, the virtual environment can be initialised using:
```bash
cd <path_to_cloned_repo>
uv sync
```

The virtual environment can be activated using
``` bash
source .venv/bin/activate
```
and disabled using
```bash
deactivate
```

### 2. Running Scripts:
For a given python script, jobs can be run using the `uv` environment as follows:
```bash
uv run <script_name> <script_arguments>
```
For example, to check the command line arguments available to the `main.py` python script, one can run:
```bash
uv run main.py -h
```

### 3. Generating Results:

There are two possible methods for processing the `.val_bands` files. The first uses the provided workflow in `main.py`. The second requires you to write code to complete the workflow, using the CLIConfig, ValBandsParser, DoSBuilder, and DoSPlotter classes as well as the storage structures KPoints and ValBandsData. The code for these can be found in both the `main.py` scripts, as well as in the `Provided Code` subdirectory.

#### Using the Pre-built Script:

In order to produce results using the same script I used to generate the Dirac Energy and DoS figure in `Summer_Project_Report.pdf`,
one can run the python script `main.py`. To do this, one must run the following:

```bash
uv run main.py -v --data-root Data --figdir Results --sys-include '^Au-00[01](?:/|$)' --sites Site-000 Site-002 Site-005 --bias-regex '(?:-0\.10V|\+0\.00V|\+0\.10V)' --prefix-mode auto 
```

This calls `uv` to run the main.py script, which will take data from the `Data/` subdirectory, and write outputs into the `Results/` directory. The flags `--sys-include`, `--sites`, `--bias-regex`, and `--prefix-mode` provide guidance to the file search routine, and should be used as provided. A list of possible command line arguments for the CLIConfig class can be found using the `-h` flag (and used by running `uv run main.py -h`). Please do not worry too much about the other options, these are mainly for my use. 

---

## 🧭 End-to-end stages

The pipeline is split into five stages. Each stage is implemented in `main.py` and can be reused as is, or can be reconstructed using elements from the `Provided Code` directory with gaps filled in using custom code. Code for you to implement is denoted by the ✍️ symbol.

Stage 0 — Discovery & configuration
 - What happens: Discover systems and biases, resolve prefixes, set output folder, configure logging.
 - Inputs: --data-root, --sys-include/--sys-exclude, --sites, --heights, --bias-regex, --prefix-mode.
 - Provided code: CLIConfig class (`cli_config.py`).
 - ✍️ Self-coded: A tiny wrapper to call CLIConfig.parse_from_argv() and CLIConfig.configure_logging(), as seen in the main() function in `main.py`.

Stage 1 — Parse ONETEP val_bands
 - What happens: Read k-point blocks, weights, eigenvalues (Ha→eV). Recover EF from logs if missing. **Please ensure you convert the units from Hartree to eV**
 - Provided code: ValBandsParser class (data reading, fallback functionality).
 - Outputs: ValBandsData(nk, nspin, neig, ef_eV, kpoints[weight, eig_eV]).
 - ✍️ Self-coded (optional): A small script that calls the parser and prints nk, neig, and <EF>.

Stage 2 — Build DoS on a fixed grid
 - What happens: Flatten all eigenvalues + weights and apply Gaussian broadening (σ = 0.10 eV).
 - Provided code: DoSBuilder class (`dos_builder.py`).
 - Outputs: energy grid (absolute eV) and DoS arrays.
 - ✍️ Self-coded: Re-implement just gaussian_dos (1–2 functions), then check your curve matches the provided plots. The `DoSBuilder` class is provided for use in `dos_builder.py`, and can easily be replicated in a custom script.

Stage 3 — Dirac energy extraction + UQ
 - What happens:
	- Find global DoS minimum within ±2 eV relative to EF → E_D (both relative/absolute returned).
	- Compute jackknife SE (leave-one-out over k-points; report 2σ = 2×SE).
	- Optionally compute quadratic 1σ around the minimum (diagnostic).
 - Provided code: DiracAnalyser class (`main.py`).
 - ✍️ Self-coded: Recompute the Dirac energy using similar logic to that used in the DiracAnalyser class to calculate the Dirac energies with the associated Jackknife uncertainties. This is left as a exercise for you to reproduce; the jackknife equation is covered in the UQ section of `Summer_Project_Report.pdf`. The process for calculating E_D is outlined within the docstring contained in the DiracAnalyser.ed_by_minimum() function.

Stage 4 — Plotting and CSVs
 - What happens: Create stacked DoS panels & 2×2 comparison; write metrics CSVs.
 - Provided code: DoSPlotter class (`dos_plotter.py`).
 - Outputs (always):
    - Results/Uncertainty_Quantification_Values.csv
    - Results/Dirac_Energies.csv
 - Outputs (only if you choose to produce figures):
    - Results/Stacked_DoS_2x2.png
    - Results/Stacked_DoS_solv.png
    - Results/Stacked_DoS_vac.png
 - ✍️ Self-coded: The full pipeline is for you to implement. A good starting point would be the DoSPipeline class in `main.py`. The DoSPlotter class is provided as an optional tool for plotting tha can be used if desired.

---

## 🎨 Plotting Details

The production of figures is included as a simple visual aid to check whether the files are being read correctly. There are three paths you can take:

**Path A — Reuse the provided plotter (recommended if you want to focus on UQ):**  
Import `DoSPlotter` from `Provided Code/dos_plotter.py` and pass the parsed data (or precomputed DoS). This will draw lines corresponding to the Fermi energy and lay out the stacked panels for you. Reusing this is simply for your own benefit.

**Path B — Write your own plotting code (DIY):**  
Produce figures with the same filenames and general appearance. You may reuse `parameters.py` for consistent styling.

**Path C — Skip producing visualisations:**  

All paths are acceptable, and the choice is left entirely up to you!

---

## What you must self-code
Implement the following (you may reuse the provided data classes and helpers):

- **Dirac energy (E_D) estimator:** find the DoS minimum within ±2 eV of E_F; return both relative (to E_F) and absolute energies.
- **Jackknife UQ (leave-one-out over k-points):** report 2σ = 2 × SE; respect k-point weights.
- **Quadratic diagnostic (optional):** local quadratic fit near the minimum; report 1σ.
- **Outputs:** Produce the two CSVs with the specified headers (and the three figures if desired).

Your implementation should reproduce (or explain discrepancies with) Table II to a suitable tolerance. An example of the column formatting can be found in the .csv files under the `Figures` subdirectory. 

---

## ✅ Acceptance Criteria
- Match Table II values to within ±0.01eV (or provide a short rationale for any systematic deviations). Larger deviations may be due to the parameters used for the Gaussian broadening or Dirac point detection.
- Produce both CSVs with the exact headers and column order:
    - Uncertainty_Quantification_Values.csv
    - Dirac_Energies.csv
- (Optional) If figures are produced, save them with these filenames:
    - Results/Stacked_DoS_2x2.png
    - Results/Stacked_DoS_solv.png
    - Results/Stacked_DoS_vac.png
---
## 📝 Peer Checklist
Before submission, ensure you have:
- [ ] Run `uv sync` to install the environment.
- [ ] Processed the provided `Data/` folder without manual edits.
- [ ] Converted the eigenenergies in the `.val_bands` files from Hartree to eV!
- [ ] Generated both CSVs with exact column names and order.
- [ ] (Optional) Produced figures in the `Results/` folder.
- [ ] Documented any systematic differences with Table II.

Please drop me a message/email if you are stuck and need additional clarification.