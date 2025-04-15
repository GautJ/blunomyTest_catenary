import numpy as np
import pandas as pd
from catenary_fitting.clustering import pca_and_dbscan_clustering

def test_dbscan_clustering_detects_two_lines():
    # Create two parallel lines of points
    line1 = np.array([[x, 0, 0] for x in range(10)])
    line2 = np.array([[x, 1.0, 0] for x in range(10)])
    data = np.vstack([line1, line2])
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])

    labels = pca_and_dbscan_clustering(df, eps=0.5, min_samples=2, pca_point_id=1) 
    unique_clusters = set(labels) - {-1}  # ignore noise
    # Assert that I have 2 clusters (the two lines)
    assert len(unique_clusters) == 2
