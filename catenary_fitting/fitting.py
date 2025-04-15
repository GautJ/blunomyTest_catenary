import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
from scipy.optimize import OptimizeWarning

# Catenary model
def catenary(x, x0, y0, c):
    """
    Catenary curve equation.

    Parameters:
        x (array): x-values.
        x0 (float): x-coordinate of the curve's lowest point.
        y0 (float): y-coordinate of the curve's lowest point.
        c (float): Curvature parameter.

    Returns:
        array: Corresponding y-values of the catenary curve.
    """
    return y0 + c * (np.cosh((x - x0) / c) - 1)

# Function that fit a catenary curve to a 2D point cloud and returns the parameters of the curve
def fit_catenary(x, z):
    """
    Fit a catenary curve to 2D data using 'curve_fit' with the catenary model.

    Parameters:
        x (array): x-coordinates of points.
        z (array): z-coordinates of points.

    Returns:
        array: Fitted parameters [x0, z0, c].
    """
    x0_init = np.mean(x)
    z0_init = np.min(z)
    c_init = 10  # arbitrary
    # Avoid warnings in terminal
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        x0_z0_c, _ = curve_fit(catenary, x, z, p0=[x0_init, z0_init, c_init])
    return x0_z0_c  # [x0, z0, c]


# Flatten the 3D wire cluster to 2D using PCA — gives a side-on view where the curve shape is easiest to spot
def project_cluster_to_2D(X_3D):
    """
    Projects 3D cluster points onto a 2D plane using PCA.

    Parameters:
        X_3D (array): Nx3 array of 3D points.

    Returns:
        tuple: (2D projected points, fitted PCA object)
    """
    pca = PCA(n_components=2) # 2 components since most variation is in the x-z plane; y doesn't change much
    projected = pca.fit_transform(X_3D)
    return projected, pca

# Fit catenary for each cluster and reproject the result into the original 3D space
def fit_catenaries_3D(data):
    """
    Fit a 3D catenary curve to each cluster in the dataset.

    Parameters:
        data (DataFrame): Contains x, y, z coordinates and 'cluster' labels.

    Returns:
        dict: Mapping from cluster ID to fitted curve and parameters.
    """
    cluster_curves = {}
    clusters = np.unique(data['cluster'])
    for cluster in tqdm(clusters, desc="Fitting catenaries"):
        cluster_xyz = data[data['cluster'] == cluster][['x', 'y', 'z']].values

        # Flatten the 3D cluster to 2D (PCA gives a side view, usually x-z plane, bowl-shaped curve)
        projected_2D, pca = project_cluster_to_2D(cluster_xyz)
        x_proj, z_proj = projected_2D[:, 0], projected_2D[:, 1]

        try:
            # Fit a catenary curve in the 2D PCA plane (x_proj, z_proj)
            params = fit_catenary(x_proj, z_proj)
            x_fit = np.linspace(x_proj.min(), x_proj.max(), 300)
            z_fit = catenary(x_fit, *params) # params = (x0, z0, c)
            fitted_2D = np.column_stack((x_fit, z_fit)) # Combine x_fit and z_fit into a 2D curve

            # Reproject the 2D fit back into 3D using the inverse PCA transform
            fitted_3D = pca.inverse_transform(fitted_2D) 

            # Store both the fitted 3D curve and the original cluster points for later visualization
            cluster_curves[cluster] = {
                "fitted_curve": fitted_3D,
                "original_points": cluster_xyz,
                "params": params
            }

        except Exception as e:
            print(f"Cluster {cluster} fit failed: {e}")
            continue

    return cluster_curves
