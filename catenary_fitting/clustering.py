from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Apply PCA to determine the principal directions of variation in the 3D point cloud.
# Use DBSCAN on a selected PCA component (here PC1 or PC2) where wires are best separated.
# This effectively reduces the 3D clustering problem to a 1D density problem along a meaningful axis.
# Note: We keep 3 PCA components to handle datasets like 'medium', where the wire alignment is non-standard
# and the best separation may lie along PC2 or PC3 (not just PC1).
def pca_and_dbscan_clustering(data, eps=0.1, min_samples=5, pca_point_id=1):
    """
    Applies PCA and DBSCAN to cluster 3D points.

    Parameters:
        data (DataFrame): Input point cloud with 'x', 'y', 'z' columns.
        eps (float): DBSCAN epsilon value.
        min_samples (int): Minimum number of samples for a cluster.
        pca_point_id (int): Which PCA component to cluster on.

    Returns:
        array: Cluster labels for each point.
    """
    pca = PCA(n_components=3) 
    data_pca = pca.fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_pca[:, [pca_point_id]]) 
    labels = db.labels_ # store the labels given by DBSCAN
    return labels

