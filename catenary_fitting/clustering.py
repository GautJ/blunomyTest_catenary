from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Apply PCA to find a plane where the points are clearly separated (top view) and apply DBSCAN to cluster the points
def pca_and_dbscan_clustering(data, eps=0.1, min_samples=5, pca_point_id=1):
    pca = PCA(n_components=3) # 3 components because of dataset medium that shows different directions in variation 
                              # (the first 2 principal components doesnt allow a top view on the wires)
    data_pca = pca.fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_pca[:, [pca_point_id]]) # Apply DBSCAN on a selected PCA axis (e.g., PC1 or PC2), where the wires are best separated
                                                                                   # This simplifies the 3D clustering to a 1D density problem along the most informative direction
    labels = db.labels_ # store the labels given by DBSCAN
    return labels

