import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
from scipy.optimize import OptimizeWarning

# Catenary model
def catenary(x, x0, y0, c):
    return y0 + c * (np.cosh((x - x0) / c) - 1)

# Function that fit a catenary curve to a 2D point cloud and returns the parameters of the curve
def fit_catenary(x, z):
    x0_init = np.mean(x)
    z0_init = np.min(z)
    c_init = 10  # arbitrary
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        x0_z0_c, _ = curve_fit(catenary, x, z, p0=[x0_init, z0_init, c_init])
    return x0_z0_c  # [x0, z0, c]


# Project a 3D cluster to 2D using PCA to have a plane where the curve of the wire is clearly seen from its side
def project_cluster_to_2D(X_3D):
    pca = PCA(n_components=2) # Only 2 components here, because the principal components of variation are the x,z coordinates in the plane where y is fixed!
    projected = pca.fit_transform(X_3D)
    return projected, pca

# Fit catenary for each cluster and reproject back to 3D 
def fit_catenaries_3D(data):
    cluster_curves = {}
    clusters = np.unique(data['cluster'])
    for cluster in tqdm(clusters, desc="Fitting catenaries"):
        cluster_xyz = data[data['cluster'] == cluster][['x', 'y', 'z']].values

        # Project in 2D plane (x,z with y fixed)
        projected_2D, pca = project_cluster_to_2D(cluster_xyz)
        x_proj, z_proj = projected_2D[:, 0], projected_2D[:, 1]

        try:
            # Fit catenary in the 2D plane given by PCA (x_proj, z_proj)
            params = fit_catenary(x_proj, z_proj)
            x_fit = np.linspace(x_proj.min(), x_proj.max(), 300)
            z_fit = catenary(x_fit, *params) # *params -> x0, z0, c
            fitted_2D = np.column_stack((x_fit, z_fit)) # 2D array of x_fit and z_fit coordinate that is used for projecting back in 3D

            # Project back to 3D
            fitted_3D = pca.inverse_transform(fitted_2D) # Gives the fitting curve in the initial 3D space

            cluster_curves[cluster] = {
                "fitted_curve": fitted_3D,
                "original_points": cluster_xyz,
                "params": params
            }

        except Exception as e:
            print(f"Cluster {cluster} fit failed: {e}")
            continue

    return cluster_curves
