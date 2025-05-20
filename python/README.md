# README

## Project: GEDI ICESat-2 DEM Regression and Analysis

This repository contains code for processing, analyzing, and visualizing Digital Elevation Model (DEM) errors using GEDI and ICESat-2 data. The main workflow is orchestrated by the `main_tipp.py` script.

---

## Main Script

### `main_tipp.py`

This script is the entry point for running the DEM regression and ablation analysis pipeline for the "Tipp" region. It performs the following tasks:

- Loads and preprocesses DEM and GEDI/ICESat-2 data.
- Trains regression models (Random Forest, SVR, KRR, etc.) to predict DEM errors.
- Evaluates model performance.
- Generates and saves prediction rasters.
- Produces visualizations such as histograms of elevation differences.

---

## Usage

### 1. Install Requirements

Make sure you have all dependencies installed. You can use `requirements.txt` if provided:

```bash
conda create -n test python==3.8.17
conda activate test
pip install -r requirements.txt
```

### 2. Prepare Data

- Place your DEM and GEDI/ICESat-2 data in the appropriate directories as expected by the scripts.
- Update paths in `main_tipp.py` and config files if necessary.

### 3. Run the Main Script

```bash
python main_tipp.py
```

You can modify parameters (e.g., regression method, resolutions) directly in `main_tipp.py`.

---

## Key Files

- **main_tipp.py**: Main workflow script for the Tipp region.
- **GEDI.py**: Contains functions for DEM error analysis and plotting.
- **regression.py**: Implements regression models and hyperparameter tuning.
- **general.py**: Utility functions for raster and data handling.

---

## Output

- Model performance metrics printed to the console.
- Prediction rasters saved to output directories.
- CSV files summarizing ablation effects.
- Plots (e.g., histograms) saved or displayed.

---

## Customization

- To change regression methods or parameters, edit the relevant sections in `main_tipp.py`.
- To add new ablation scenarios or resolutions, update the corresponding lists.
