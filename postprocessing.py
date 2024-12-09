import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm


def planer_cluster(df):
    """
    Planar clustering method using RANSAC for plane segmentation.
    """
    # Convert DataFrame to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    points = df[["x", "y", "z"]].to_numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RANSAC for plane detection
    max_plane_idx = 1000
    segment_models = {}
    segments = {}
    rest = pcd

    # Loop for planar segmentation with progress tracking
    for i in tqdm(range(max_plane_idx), desc="Segmenting planes"):
        plane_model, inliers = rest.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        segment_models[i] = plane_model
        segments[i] = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)

        # Assign labels to inlier points
        inlier_points = np.asarray(segments[i].points)
        inlier_labels = np.full(inlier_points.shape[0], i)

        # Convert inlier_points to a set of tuples for fast lookups
        inlier_points_set = set(map(tuple, inlier_points))

        # Convert the DataFrame's x, y, z columns into tuples
        points_tuples = df[["x", "y", "z"]].to_numpy()

        # Create a boolean mask using NumPy's `isin` for multi-column comparison
        mask = np.array([tuple(point) in inlier_points_set for point in points_tuples])

        # Assign cluster labels to the matching rows
        df.loc[mask, "cluster"] = i

    # Handle remaining points as outliers
    remaining_points = np.asarray(rest.points)

    # Convert remaining points to a set of tuples for fast lookup
    remaining_points_set = set(map(tuple, remaining_points))

    # Convert the DataFrame's x, y, z columns into tuples
    points_tuples = df[["x", "y", "z"]].to_numpy()

    # Create a boolean mask for remaining points
    mask = np.array([tuple(point) in remaining_points_set for point in points_tuples])

    # Assign the outlier label (-1) to the matching rows
    df.loc[mask, "l"] = -1

    return df


def vote_cluster(df):
    """
    Vote clustering method each cluster votes together on their label and the label with the most votes wins.

    """
    # Group by cluster and count the number of occurrences of each label (l) in the cluster
    cluster_votes = df.groupby("cluster")["l"].value_counts()

    # Find the label with the most votes for each cluster
    cluster_votes = cluster_votes.reset_index(name='count')
    idx = cluster_votes.groupby('cluster')['count'].transform(max) == cluster_votes['count']
    cluster_votes = cluster_votes[idx]

    # Assign the winning label to each cluster
    cluster_votes = cluster_votes.set_index('cluster')
    df['cluster'] = df['cluster'].map(cluster_votes['l'])

