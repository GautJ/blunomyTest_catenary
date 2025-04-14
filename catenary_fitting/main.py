import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from catenary_fitting.clustering import pca_and_dbscan_clustering
from catenary_fitting.visualization import visualize_point_cloud_3d, plot_3D_fitted_catenaries
from catenary_fitting.fitting import fit_catenaries_3D
from catenary_fitting.utils import save_plot_3D_fitted_catenaries, save_interactive_3D_plot

def main():
    # Directories to store output visualizations
    STATIC_DIR = os.path.join("output", "static_figures")
    INTERACTIVE_DIR = os.path.join("output", "interactive_figures")

    # Load dataset files from the 'datasets' folder
    DATA_DIR = os.path.join("datasets")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Folder '{DATA_DIR}' not found. Please make sure the dataset directory exists.")

    # Checks if the directory 'datasets' that holds the LIDAR 3D point cloud files exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Folder '{DATA_DIR}' not found. Please make sure the dataset directory exists.")

    dataset_files = [
        "lidar_cable_points_easy.parquet",
        "lidar_cable_points_medium.parquet",
        "lidar_cable_points_hard.parquet",
        "lidar_cable_points_extrahard.parquet"
    ]

    names = ['easy', 'medium', 'hard', 'extrahard']
    datasets = []

    for filename in dataset_files:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File missing: {path}")
        datasets.append(pd.read_parquet(path))


    # Cluster the point clouds using PCA + DBSCAN (more in clustering.py)
    eps = 0.1
    min_samples = 5

    for data, name in zip(datasets, names):
        # The 'medium' dataset needs a different PCA axis due to wire alignment
        pca_axis = 2 if name == "medium" else 1
        if name == "medium":
            data['cluster'] = pca_and_dbscan_clustering(data[['x', 'y', 'z']], eps, min_samples, pca_axis) 
        else:
            data['cluster'] = pca_and_dbscan_clustering(data[['x', 'y', 'z']], eps, min_samples, pca_axis)

        # Count number of valid clusters (ignore noise cluster -1)
        unique_clusters = set(data['cluster'])
        num_wires = len(unique_clusters - {-1})  # Remove noise label if present
        print(f"Dataset '{name}': Detected {num_wires} wires")

    # Fit catenary curves and save both static and interactive 3D plots in the 'ouput/' directory
    for data, name in zip(datasets, names):
        filenameStatic = f"static_fittedCatenaries_{name}.png"
        filenameInteractive = f"interactive_fittedCatenaries_{name}.html"
        cluster_curves = fit_catenaries_3D(data) # see fitting.py
        save_plot_3D_fitted_catenaries(cluster_curves, output_path=os.path.join(STATIC_DIR, filenameStatic),title=f"Catenary Fitting in 3D: {name}", name=name, show_points=True)
        save_interactive_3D_plot(cluster_curves, output_path=os.path.join(INTERACTIVE_DIR, filenameInteractive), title=f"3D Catenary Fits (Interactive): {name}", name=name, show_points=True)



# OPTIONAL : Visualize raw point clouds and fitted wire curves using matplotlib

    # ###  3D visualization of the raw datasets
    # for dataset, name in zip(datasets, names):
    #     visualize_point_cloud_3d(dataset, title=f"3D point cloud of dataset {name}")
    
    # ### 3D visualization of the fitted curves to each cluster (wire)
    # for data, name in zip(datasets, names):
    #     cluster_curves = fit_catenaries_3D(data)
    #     plot_3D_fitted_catenaries(cluster_curves, title=f"Catenary Fitting in 3D: {name}")


if __name__ == "__main__":
    main()


