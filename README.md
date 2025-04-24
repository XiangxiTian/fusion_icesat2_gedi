
# A python implementation on ICESat-2 Controlled Integration of GEDI and SRTM Data for Large-Scale Digital Elevation Model Generation

**Authors:** Xiangxi Tian & Jie Shan  
**Affiliation:** Purdue University

## Citation

If you find this repo useful and use this work in your research, please cite as:
```
X. Tian and J. Shan, “ICESat-2 Controlled Integration of GEDI and SRTM Data for Large-Scale Digital Elevation Model Generation,” in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024, Art. no. 5703414, doi: 10.1109/TGRS.2024.3389821.
```
or
```
@ARTICLE{10500859,
  author={Tian, Xiangxi and Shan, Jie},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={ICESat-2 Controlled Integration of GEDI and SRTM Data for Large-Scale Digital Elevation Model Generation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2024.3389821}}

```

---

## Overview
This repository implements a fusion framework that leverages the high-accuracy ICESat-2 ATL08 data to enhance GEDI L2A terrain measurements and improve existing global DEMs (e.g., SRTM) via spatial regression.

## Motivation & Objectives
1. Use ICESat-2 ATL08 sub‑meter terrain measurements to correct systematic biases in GEDI L2A heights.  
2. Develop a scalable regression framework to enhance global DEM quality at 30 m and 90 m resolutions.

## Study Areas & Datasets
| County      | Area (km²) | ATL08 (before → after)                        | GEDI L2A (before → after)                     |
|-------------|------------|-----------------------------------------------|-----------------------------------------------|
| Tippecanoe  | 1,303.4    | 37,904 (−0.38 ± 1.37 m) → 31,484 (−0.56 ± 1.00 m) | 218,106 (0.32 ± 853.21 m) → 80,014 (0.12 ± 2.15 m) |
| Mendocino   | 10,040.0   | 301,637 (−0.91 ± 7.47 m) → 125,174 (−1.62 ± 5.69 m) | 1,517,011 (−54.02 ± 804.60 m) → 771,418 (1.51 ± 9.33 m) |

> **Note:** Filtering removes non‑viable shots (63.3% in Tippecanoe; 48.7% in Mendocino) to retain high‑quality measurements.

## Methodology

<img src="src/workflow.png" alt="Workflow diagram" width="600"/>

1. **Problem Definition**  
   Treat DEM enhancement as a regression problem:
   $$
   \hat h(s_0) = 𝑓_𝜽 (𝑿(𝑠_0 ))
   $$
   where $h$ is the true terrain height, and $X$ collects multi‑source features.

2. **Predictor Variables**  
   - $X_{pos}$: geographic coordinates  
   - $X_{DEM}$': elevations from the baseline DEM at neighboring cells  
   - $X_{GEDI}$: distances & heights of nearby GEDI footprints  

3. **Regression Models**  
   - Random Forest Regression (RFR)  
   - Kernel Ridge Regression (KRR) with RBF kernel  
   - Support Vector Regression (SVR) with RBF kernel  

## Results

### RMSE Comparison (m)

| Model | Tippecanoe 30 m | Tippecanoe 90 m | Mendocino 30 m | Mendocino 90 m |
|-------|-----------------|-----------------|----------------|----------------|
| KRR   | 2.83            | 1.93            | 12.81          | 20.61          |
| RFR   | 2.15            | 1.25            |  8.25          | 14.38          |
| SVR   | 1.54            | 1.45            |  7.50          | 13.42          |

> All three models significantly outperform SRTM, achieving up to ~30% lower RMSE.

### Error Distributions & Bias
- **Tippecanoe (30 m)**: SVR yields median error ≈ −0.49 m, σ ≈ 3.39 m.  
- **Mendocino (30 m)**: SVR yields median ≈ −1.89 m, σ ≈ 26.03 m.  
- The fusion framework reduces DEM bias by ~85% compared to SRTM.

### Enhanced GEDI Data
- The filtered GEDI product becomes fully usable after bias correction.  
- Systematic errors are minimized, enabling dense, reliable point‑cloud coverage.

## Conclusion
We present a data fusion/regression framework that:
- Integrates ICESat-2 and GEDI for DEM refinement  
- Enhances SRTM’s vertical accuracy by ~30% RMSE reduction and ~85% bias decrease  
- Is scalable to additional spaceborne or irregular elevation datasets

## Repository Structure
```
.
├── data/                 # Raw ATL08, GEDI & reference DEM tiles
├── notebooks/            # EDA, feature‑importance & error analysis
├── src/
│   ├── preprocess.py     # Filtering & feature extraction
│   ├── train_model.py    # Train RFR, KRR, SVR
│   ├── predict_dem.py    # Generate enhanced DEMs
│   └── evaluate.py       # RMSE & error distribution
├── results/              # Generated DEMs & evaluation figures
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation & Dependencies
```bash
pip install -r requirements.txt
```

<!-- ## Usage

1. **Preprocess & filter data**  
   ```bash
   python src/preprocess.py      --icesat2 data/ATL08/      --gedi data/GEDI/      --out data/processed/
   ```
2. **Train regression models**  
   ```bash
   python src/train_model.py      --input data/processed/      --model_dir models/
   ```
3. **Generate enhanced DEM**  
   ```bash
   python src/predict_dem.py      --model models/svr.pkl      --resolution 30      --out results/dem_30m.tif
   ```
4. **Evaluate & visualize**  
   ```bash
   python src/evaluate.py      --dem results/dem_30m.tif      --reference data/3DEP/      --out results/metrics.csv
   ``` -->

