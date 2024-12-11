import pandas as pd
import matplotlib.pyplot as plt
from data_utils.DLRGroupDataLoader import DLRGroupDatasetWholeScenePoints
import numpy as np
import os
import cv2

plot_floor_gram = False
plot_floors = False
output_directory = r"D:\Datasets\PointClouds\floor_slices"

TRAIN_DATASET = DLRGroupDatasetWholeScenePoints(data_root=r"D:\Datasets\PointClouds\nps",split='train', test_project="MorrisCollege_Pinson", labels_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt")

d_i = 0
for dataset in TRAIN_DATASET:

    df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'r', 'g', 'b'])
    print("New Dataset")
    print(df.shape)
    # Compute histogram for 'z' values of floors
    hist, bin_edges = np.histogram(df['z'], bins=150)


    # detect outliers
    # Calculate mean and standard deviation
    mean_count = np.mean(hist)
    std_count = np.std(hist)

    # Identify outlier bins (e.g., > 3 standard deviations from the mean)
    outlier_threshold = mean_count + 3 * std_count
    outlier_bins = np.where(hist > outlier_threshold)[0]

    # Display results
    outlier_bins, hist[outlier_bins]

    floor_z_values = [(bin_edges[i], bin_edges[i + 1]) for i in outlier_bins]

    # Sort the bins by Z value for clarity
    floor_z_values = sorted(floor_z_values, key=lambda x: x[0])

    if plot_floor_gram == True:
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['z'], bins=150, edgecolor='black', alpha=0.7, histtype='step', label='Histogram')

        # Add vertical lines for the edges of the outlier bins
        # Add vertical lines for the largest bins
        for start, end in floor_z_values:
            plt.axvline(x=start, color='red', linestyle='--', linewidth=2, label=f'Floor: {start:.2f}-{end:.2f}')
            plt.axvline(x=end, color='red', linestyle='--', linewidth=2)

        # Add labels and legend
        plt.title('Histogram of z values with Outlier Bin Edges')
        plt.xlabel('z values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Initialize an empty dictionary to hold the DataFrames for each floor
    split_dfs = {}

    # Include the basement (below the first floor)
    split_dfs['Basement'] = df[df['z'] < floor_z_values[0][0]]

    # Assign floors
    for i, (start, end) in enumerate(floor_z_values):
        if i < len(floor_z_values) - 1:  # For intermediate floors
            next_start = floor_z_values[i + 1][0]
            split_dfs[f'Floor_{i + 1}'] = df[(df['z'] >= start) & (df['z'] < next_start)]
        else:  # For the topmost floor
            split_dfs[f'Floor_{i + 1}'] = df[df['z'] >= start]

    # Print the counts for verification
    for floor, floor_df in split_dfs.items():
        print(f"{floor}: {len(floor_df)} points")

    # Plot the heatmaps
    # Create a combined 2D heatmap and interactive 3D scatter plot for each floor
    if plot_floors == True:
        fig, axes = plt.subplots(len(split_dfs.keys()), 1, figsize=(12, 6 * len(split_dfs.keys())), constrained_layout=True)

    for fig_i, (floor, floor_df) in enumerate(split_dfs.items()):
        # Define bins for 2D histogram
        x_bins = np.linspace(0, 100, 512 + 1)
        y_bins = np.linspace(0, 100, 512 + 1)

        # Create 2D histogram for heatmap
        hist_xy, x_edges_xy, y_edges_xy = np.histogram2d(floor_df['x'], floor_df['y'], bins=[x_bins, y_bins])

        # Normalize the histogram
        hist_xy = hist_xy / hist_xy.sum()
        if plot_floors == True:
            # Plot 2D heatmap
            axes[fig_i].imshow(
                hist_xy.T,
                origin='lower',
                extent=[0, 100, 0, 100],
                cmap='viridis',
                aspect='equal'
            )
            axes[fig_i].set_title(f'Density Heatmap (X-Y Top View, {floor})')
            axes[fig_i].set_xlabel('X Coordinate')
            axes[fig_i].set_ylabel('Y Coordinate')

            # Set fixed limits for the heatmap
            axes[fig_i].set_xlim(0, 100)
            axes[fig_i].set_ylim(0, 100)

        # chunk into 126X126 images
        chunk_size = 128
        # Apply consistent normalization across all chunks
        vmin, vmax = hist_xy.min(), hist_xy.max()
        # Split the 512x512 array into 128x128 chunks
        # This will reshape it into a 4x4 grid of 128x128 images
        chunks = [
            hist_xy[i:i + chunk_size, j:j + chunk_size]
            for i in range(0, hist_xy.shape[0], chunk_size)
            for j in range(0, hist_xy.shape[1], chunk_size)
        ]

        # Save each chunk as a 1D image with consistent colormap scaling
        for i, chunk in enumerate(chunks):
            plt.imsave(
                os.path.join(output_directory, f"{floor}_{i}_{d_i}.png"),
                chunk,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,  # Ensures consistent application of the colormap
            )

    if plot_floors == True:
        plt.show()

    d_i += 1



