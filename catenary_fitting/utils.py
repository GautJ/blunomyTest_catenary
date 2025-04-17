import matplotlib.pyplot as plt
import plotly.graph_objects as go

def save_plot_3D_fitted_catenaries(cluster_curves, output_path, title, show_points=True):
    """
    Saves a static 3D plot of the fitted catenaries to a PNG file.

    Parameters:
        cluster_curves (dict): Fitted curves per cluster, and original points.
        output_path (str): File path to save the PNG.
        title (str): Title of the plot.
        show_points (bool): If True, show original points too.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id, data in cluster_curves.items():
        curve_3d = data["fitted_curve"]
        original = data["original_points"]
        
        # Possibility to not show the points but only the fitted curve
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

    # Count number of wires (clusters)
    num_clusters = len(cluster_curves)

    # Add an annotation box showing total wires
    textstr = f'Total wires: {num_clusters}'
    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved static plot to {output_path}")

def save_interactive_3D_plot(cluster_curves, output_path, title, show_points=True):
    """
    Saves an interactive 3D plot of the fitted catenaries to a HTML file.

    Parameters:
        cluster_curves (dict): Fitted curves per cluster, and original points.
        output_path (str): File path to save the PNG.
        title (str): Title of the plot.
        show_points (bool): If True, show original points too.
    """

    fig = go.Figure()
    buttons = []
    visibility = []

    all_original_points = []
    all_clustered_points = []
    all_fitted_curves = []

    # Loop through clusters and add data
    for cluster_id, data in cluster_curves.items():
        curve = data["fitted_curve"]
        points = data["original_points"]

        # Original points (no color)
        orig = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.2, color='gray'),
            name=f"Original {cluster_id}",
            visible=True
        )
        all_original_points.append(orig)

        # Points clustered with PCA + DBSCAN (with label color)
        cluster = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.4),
            name=f"Cluster {cluster_id}",
            visible=False
        )
        all_clustered_points.append(cluster)

        # Fitted curves (catenary)
        curve_line = go.Scatter3d(
            x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
            mode='lines',
            line=dict(width=4),
            name=f"Fit {cluster_id}",
            visible=False
        )
        all_fitted_curves.append(curve_line)

    # Add all traces to figure
    for trace in all_original_points + all_clustered_points + all_fitted_curves:
        fig.add_trace(trace)

    n = len(cluster_curves)

    # Trace visibility order: [originals, clusters, curves]
    total_traces = 3 * n

    # Buttons setup
    buttons = [
        dict(label="Original Points",
             method="update",
             args=[{"visible": [i < n for i in range(total_traces)]},
                   {"title": f"{title} - Showing Original Points"}]),
        dict(label="Clustered Points",
             method="update",
             args=[{"visible": [n <= i < 2*n for i in range(total_traces)]},
                   {"title": f"{title} - Showing Clustered Points"}]),
        dict(label="Fitted Curves",
             method="update",
             args=[{"visible": [i >= 2*n for i in range(total_traces)]},
                   {"title": f"{title} - Showing Fitted Curves"}]),
        dict(label="All",
             method="update",
             args=[{"visible": [True]*total_traces},
                   {"title": f"{title} - Showing All"}])
    ]

    # Layout parameters
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="down",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="left",
            y=0.5,
            yanchor="top"
        )],
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title=title,
        width=1000,
        height=700
    )

    fig.write_html(output_path)
    print(f"Saved interactive plot to {output_path}")

