# Blunomy Data Science Test – LiDAR Wire Detection

This project is part of a technical assessment for a data science internship with Blunomy. The goal is to process drone-based LiDAR data and automatically identify and model the 3D shape of suspended wires using best-fit **catenary curves**.

---

## Problem Statement

Given 3D LiDAR point clouds representing power lines, the objective is to:

- Automatically **cluster points** corresponding to individual wires.
- Fit a **catenary curve** (2D cable model) to each cluster.
- **Reproject the fitted curve back to 3D** for visualization and output.

---

## Approach Summary

The pipeline is as follows:

1. **Preprocessing**:
   - Point cloud datasets are loaded from `.parquet` files.

2. **Clustering**:
   - To separate the wires, I first applied Principal Component Analysis (PCA) to the 3D point cloud. This allowed me to identify the main axis along which the wires are distributed — typically PC1 or PC2, depending on the dataset.
   - I then used DBSCAN, a density-based clustering algorithm, to group points based on their positions along the selected PCA component. Clustering was performed in the PCA-projected space, but the resulting labels were mapped back to the original 3D points, as PCA preserves the order of the data.
   - For certain datasets like medium, the second principal component (PC2) was used instead of the first, due to the disposition of the wires in the point cloud.

3. **Catenary Fitting**:
   - Each identified cluster is projected into a 2D plane using PCA, effectively aligning the wire so it can be viewed from the side — where its shape resembles a catenary (a hanging cable or "bowl"-like curve).
   - A catenary curve is then fitted to the 2D points using scipy.optimize.curve_fit, based on the catenary equation.
   - Finally, the fitted 2D curve is reprojected back into 3D space using the inverse of the original PCA transformation, preserving its orientation within the point cloud.

4. **Visualization & Output**:
   - Static 3D plots (`.png`) and interactive 3D plots (`.html`) are saved for each dataset.
   - Output files are stored in the `output/` directory.

---

## Project Structure

blunomy-wire-detection/
├── catenary_fitting/               # Python package containing all main code
│   ├── __init__.py                 # Makes this folder a Python package
│   ├── main.py                     # Entry point to run the full pipeline
│   ├── clustering.py               # Handles PCA + DBSCAN-based clustering
│   ├── fitting.py                  # Catenary model fitting and 3D projection
│   ├── visualization.py            # 3D plotting of clusters and fitted curves
│   └── utils.py                    # Utility functions for saving plots and files
├── datasets/                       # Folder for input .parquet LiDAR point cloud files
├── output/
│   ├── static_figures/             # Saved static 3D visualizations (.png)
│   └── interactive_figures/        # Saved interactive 3D plots (.html)
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Files to ignore in Git
└── README.md                       # Project description and usage instructions


---

## How to run ?

1. Place your .parquet files in the datasets/ folder.
2. Run the main script as a module from the project root: python -m catenary_fitting.main

This will generate your outputs inside the 'output/' directory, which contains:
    - static_figures/: Static 3D plots (.png)
    - interactive_figures/: Interactive 3D visualizations (.html)

Each plot shows the original LiDAR points, clustered per wire, along with the corresponding catenary curve fitting 😄
