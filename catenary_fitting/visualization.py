import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Helper function that allow a 3D visualization of the (x,y,z) points
def visualize_point_cloud_3d(data, label_col=None, title="3D Point Cloud"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if label_col and label_col in data.columns:
        labels = np.unique(data[label_col])
        cmap = cm.get_cmap('tab10')
        for i, label in enumerate(labels):
            points = data[data[label_col] == label]
            color = 'black' if label == -1 else cmap(i % 10)
            ax.scatter(points['x'], points['y'], points['z'], s=3, color=color, label=f'{label_col} {label}')
        ax.legend()
    else:
        ax.scatter(data['x'], data['y'], data['z'], s=2, alpha=0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# --- Visualization ---
def plot_3D_fitted_catenaries(cluster_curves, title, show_points=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id, data in cluster_curves.items():
        curve_3d = data["fitted_curve"]
        original = data["original_points"]

        if show_points:
            ax.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, alpha=0.2, label=f"Points {cluster_id}")

        ax.plot(curve_3d[:, 0], curve_3d[:, 1], curve_3d[:, 2],
                label=f'Catenary Fit {cluster_id}', linewidth=2)

    num_clusters = len(cluster_curves)
    final_title = f"{title} ({num_clusters} wires)" if title else f"Catenary Fit ({num_clusters} clusters)"
    ax.set_title(final_title)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()
