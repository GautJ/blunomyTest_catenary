import matplotlib.pyplot as plt
import plotly.graph_objects as go

def save_plot_3D_fitted_catenaries(cluster_curves, output_path, title, name, show_points=True):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id, data in cluster_curves.items():
        curve_3d = data["fitted_curve"]
        original = data["original_points"]

        if show_points:
            ax.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, alpha=0.2)

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
    plt.savefig(output_path)
    print(f"Saved static plot to {output_path}")

def save_interactive_3D_plot(cluster_curves, output_path, title, name, show_points=True):
    fig = go.Figure()

    for cluster_id, data in cluster_curves.items():
        curve = data["fitted_curve"]
        points = data["original_points"]

        if show_points:
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=2, opacity=0.3),
                name=f"Cluster {cluster_id}"
            ))
        
        fig.add_trace(go.Scatter3d(
            x=curve[:, 0], y=curve[:,1], z=curve[:,2],
            mode='lines',
            line=dict(width=4),
            name=f"Cluster {cluster_id}"
        ))
    
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title=title,
        width=1000,
        height=700
    )

    fig.write_html(output_path)
    print(f"Saved interactive plot to {output_path}")

