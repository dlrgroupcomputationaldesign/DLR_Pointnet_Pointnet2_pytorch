import pandas as pd
import matplotlib.pyplot as plt
from data_utils.DLRGroupDataLoader import DLRGroupDatasetWholeScenePoints
import numpy as np
import os
import cv2
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from processing import detect_and_label_walls






# def detect_and_label_walls(df):
#     plot_floor_gram = False
#     plot_floors = False
#     # Compute histogram for 'z' values of floors
#     hist, bin_edges = np.histogram(df['z'], bins=150)
#
#
#     # detect outliers
#     # Calculate mean and standard deviation
#     mean_count = np.mean(hist)
#     std_count = np.std(hist)
#
#     # Identify outlier bins (e.g., > 3 standard deviations from the mean)
#     outlier_threshold = mean_count + 3 * std_count
#     outlier_bins = np.where(hist > outlier_threshold)[0]
#
#     # Display results
#     outlier_bins, hist[outlier_bins]
#
#     floor_z_values = [(bin_edges[i], bin_edges[i + 1]) for i in outlier_bins]
#
#     # Sort the bins by Z value for clarity
#     floor_z_values = sorted(floor_z_values, key=lambda x: x[0])
#
#     if plot_floor_gram == True:
#         # Plot histogram
#         plt.figure(figsize=(10, 6))
#         plt.hist(df['z'], bins=150, edgecolor='black', alpha=0.7, histtype='step', label='Histogram')
#
#         # Add vertical lines for the edges of the outlier bins
#         # Add vertical lines for the largest bins
#         for start, end in floor_z_values:
#             plt.axvline(x=start, color='red', linestyle='--', linewidth=2, label=f'Floor: {start:.2f}-{end:.2f}')
#             plt.axvline(x=end, color='red', linestyle='--', linewidth=2)
#
#         # Add labels and legend
#         plt.title('Histogram of z values with Outlier Bin Edges')
#         plt.xlabel('z values')
#         plt.ylabel('Frequency')
#         plt.legend()
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.show()
#
#     # Initialize an empty dictionary to hold the DataFrames for each floor
#     split_dfs = {}
#
#     # Include the basement (below the first floor)
#     # split_dfs['Basement'] = df[df['z'] < floor_z_values[0][0]]
#
#     # Assign floors
#     for i, (start, end) in enumerate(floor_z_values):
#         if i < len(floor_z_values) - 1:  # For intermediate floors
#             next_start = floor_z_values[i + 1][0]
#             split_dfs[f'Floor_{i + 1}'] = df[(df['z'] >= start) & (df['z'] < next_start)]
#         else:  # For the topmost floor
#             split_dfs[f'Floor_{i + 1}'] = df[df['z'] >= start]
#
#     # Print the counts for verification
#     for floor, floor_df in split_dfs.items():
#         print(f"{floor}: {len(floor_df)} points")
#
#     # Plot the heatmaps
#     # Create a combined 2D heatmap and interactive 3D scatter plot for each floor
#     all_lines = {}
#     dfs = []
#     for fig_i, (floor, floor_df) in enumerate(split_dfs.items()):
#
#         all_min = min(floor_df['x'].min(), floor_df['y'].min(), floor_df['z'].min())
#         all_max = max(floor_df['x'].max(), floor_df['y'].max(), floor_df['z'].max())
#
#         # Plot points with Plotly 3D and visualize by wall_label
#         samp = min(50_000, floor_df.shape[0])
#         fig = px.scatter_3d(
#             floor_df.sample(samp),
#             x='x', y='y', z='z',
#             title=f'3D Point Cloud Floor'
#         )
#
#
#
#         fig.update_traces(marker=dict(size=1))
#
#         fig.update_layout(
#             scene=dict(
#                 aspectmode='manual',  # Use manual to enforce aspect ratios
#                 aspectratio=dict(x=1, y=1, z=1),  # Equal aspect ratio
#                 xaxis=dict(range=[all_min, all_max]),
#                 yaxis=dict(range=[all_min, all_max]),
#                 zaxis=dict(range=[all_min, all_max])
#             ),
#             legend=dict(
#                 itemsizing='constant',
#                 itemwidth=30,
#                 tracegroupgap=0,
#                 font=dict(size=12)
#             ))
#
#         # fig.show()
#
#
#         print(f"Floor has {floor_df.shape} points")
#         floor_max_x = int(np.ceil(floor_df['x'].max()))  # Round up
#         floor_min_x = int(np.floor(floor_df['x'].min()))  # Round down
#         floor_max_y = int(np.ceil(floor_df['y'].max()))  # Round up
#         floor_min_y = int(np.floor(floor_df['y'].min()))  # Round down
#
#         # Define bins for 2D histogram
#         x_bins = np.linspace(floor_min_x, floor_max_x, floor_max_x)
#         y_bins = np.linspace(floor_min_y, floor_max_y, floor_max_y)
#
#         # Create 2D histogram for heatmap
#         hist_xy, x_edges_xy, y_edges_xy = np.histogram2d(floor_df['x'], floor_df['y'], bins=[x_bins, y_bins])
#
#         # Normalize the histogram
#         # hist_xy = hist_xy / hist_xy.sum()
#
#         # Rescale to 0-255
#         # hist_xy_rescaled = (hist_xy * 255).astype(np.uint8)
#
#         hist_xy = 255 * (hist_xy - np.min(hist_xy)) / (np.max(hist_xy) - np.min(hist_xy))
#         hist_xy = hist_xy.astype(np.uint8)
#
#         # Flatten the array to 1D for histogram
#         hist_xy_flat = hist_xy.flatten()
#
#         # Plot the histogram of pixel values
#         plt.figure(figsize=(10, 6))
#         plt.hist(hist_xy_flat, bins=256, edgecolor='black', alpha=0.7, histtype='step')
#         plt.title('Pixel Value Distribution (Histogram)')
#         plt.xlabel('Pixel Value')
#         plt.ylabel('Frequency')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         # plt.show()
#
#         kernel_size = 5
#         blur_gray = cv2.GaussianBlur(hist_xy,(kernel_size, kernel_size),0)
#
#         if blur_gray.dtype != 'uint8':
#             blur_gray = cv2.convertScaleAbs(blur_gray)
#
#         low_threshold = 50
#         high_threshold = 150
#         edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
#
#         fig, axes = plt.subplots(2,1, figsize=(12, 6), constrained_layout=True)
#         axes[0].imshow(edges, cmap='gray')
#         axes[0].axis('off')
#
#         axes[1].imshow(hist_xy, cmap='gray')
#         axes[1].axis('off')
#         plt.title('Canny Edge Detection')
#
#         # plt.show()
#
#         rho = 1  # distance resolution in pixels of the Hough grid
#         theta = np.pi / 180  # angular resolution in radians of the Hough grid
#         threshold = 15  # minimum number of votes (intersections in Hough grid cell)
#         min_line_length = 3  # minimum number of pixels making up a line
#         max_line_gap = 20  # maximum gap in pixels between connectable line segments
#         line_image = np.copy(hist_xy) * 0  # creating a blank to draw lines on
#
#         # Run Hough on edge detected image
#         # Output "lines" is an array containing endpoints of detected line segments
#         lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                             min_line_length, max_line_gap)
#
#         # Plot hist_xy with lines drawn over it
#         plt.figure(figsize=(12, 6))
#         # plt.imshow(hist_xy, cmap='gray')
#
#         # Draw the lines with blue and endpoints with red
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Line in blue
#                 plt.scatter([x1, x2], [y1, y2], color='red', zorder=5)  # Endpoints in red
#
#         plt.axis('off')
#         plt.title('Lines Over Histogram')
#         # plt.show()
#
#
#         # now run it again on the line image
#         # edges = cv2.Canny(line_image, low_threshold, high_threshold)
#         #
#         # fig, axes = plt.subplots(2,1, figsize=(12, 6), constrained_layout=True)
#         # axes[0].imshow(edges, cmap='gray')
#         # axes[0].axis('off')
#         #
#         # axes[1].imshow(line_image, cmap='gray')
#         # axes[1].axis('off')
#         # plt.title('Canny Edge Detection')
#         #
#         # # plt.show()
#         #
#         #
#         # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#         #                     min_line_length, max_line_gap)
#         #
#         #
#         # rho = 1  # distance resolution in pixels of the Hough grid
#         # theta = np.pi / 180  # angular resolution in radians of the Hough grid
#         # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
#         # min_line_length = 10  # minimum number of pixels making up a line
#         # max_line_gap = 20  # maximum gap in pixels between connectable line segments
#         # line_image = np.copy(hist_xy) * 0  # creating a blank to draw lines on
#         #
#         # # Run Hough on edge detected image
#         # # Output "lines" is an array containing endpoints of detected line segments
#         # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#         #                     min_line_length, max_line_gap)
#         # # Plot hist_xy with lines drawn over it
#         # plt.figure(figsize=(12, 6))
#         # plt.imshow(hist_xy, cmap='gray', extent=(0, 100, 0, 100))
#         #
#         # # Draw the lines with blue and endpoints with red
#         # if lines is not None:
#         #     for line in lines:
#         #         for x1, y1, x2, y2 in line:
#         #             plt.plot([x1, x2], [y1, y2], color='blue', linewidth=2)  # Line in blue
#         #             plt.scatter([x1, x2], [y1, y2], color='red', zorder=5)  # Endpoints in red
#         #
#         # plt.axis('off')
#         # plt.title('Lines Over Histogram')
#         # plt.show()
#         # plt.close()
#
#         floor_df.reset_index(drop=True, inplace=True)
#         print(f"Floor has {floor_df.shape} points")
#         all_lines[floor] = lines
#         floor_df.loc[:, 'wall_label'] = -1
#
#         proximity_threshold = 1
#
#         inspect = -1
#
#         if lines is not None:
#
#             for idx, line in enumerate(lines):
#                 # Line segment endpoints
#                 y1, x1, y2, x2 = line[0]
#
#                 # Segment direction vector
#                 dx = x2 - x1
#                 dy = y2 - y1
#                 segment_length_sq = dx ** 2 + dy ** 2
#
#                 # Extract point coordinates from floor_df
#                 p_x = floor_df['x'].to_numpy()
#                 p_y = floor_df['y'].to_numpy()
#
#                 # Handle degenerate segment (zero length)
#                 if segment_length_sq == 0:
#                     # All distances are calculated relative to the single point (x1, y1)
#                     distances = np.sqrt((p_x - x1) ** 2 + (p_y - y1) ** 2)
#                 else:
#                     # Compute the projection parameter t for all points
#                     t = ((p_x - x1) * dx + (p_y - y1) * dy) / segment_length_sq
#
#                     # Clamp t to [0, 1] for finite segment projection
#                     t = np.clip(t, 0, 1)
#
#                     # Compute the closest points on the segment
#                     closest_x = x1 + t * dx
#                     closest_y = y1 + t * dy
#
#                     # Compute the Euclidean distances to the closest points
#                     distances = np.sqrt((p_x - closest_x) ** 2 + (p_y - closest_y) ** 2)
#
#                 # Find points within the proximity threshold
#                 close_points = distances <= proximity_threshold
#
#                 # Assign the current line's index (or any unique label) as the wall_label
#                 floor_df.loc[close_points, 'wall_label'] = idx
#                 all_min = min(floor_df['x'].min(), floor_df['y'].min(), floor_df['z'].min())
#                 all_max = max(floor_df['x'].max(), floor_df['y'].max(), floor_df['z'].max())
#                 if inspect == idx:
#                     floor_df['dist'] = distances
#                     fig = px.scatter_3d(
#                         floor_df.sample(samp),  # Adjust sample size if necessary
#                         x='x', y='y', z='z',
#                         color='dist',
#                         labels={'wall_label': 'Wall Label'},
#                         title='3D Point Cloud by Wall Label'
#                     )
#                 else:
#                     # Plot the floor_df with the point cloud
#                     fig = px.scatter_3d(
#                         floor_df.sample(samp),  # Adjust sample size if necessary
#                         x='x', y='y', z='z',
#                         color='wall_label',
#                         labels={'wall_label': 'Wall Label'},
#                         title='3D Point Cloud by Wall Label'
#                     )
#
#                 # Add lines to the plot (custom lines)
#                 line_trace = go.Scatter3d(
#                     x=[x1, x2], y=[y1, y2], z=[0, 0],  # Assume z remains constant
#                     mode='lines',
#                     line=dict(color='black', width=2),
#                     name=f"Line ({x1},{y1}) to ({x2},{y2})"
#                 )
#                 fig.add_trace(line_trace)
#
#                 # Add colored axes
#                 fig.add_trace(go.Scatter3d(
#                     x=[all_min, all_max], y=[0, 0], z=[0, 0],
#                     mode='lines',
#                     line=dict(color='red', width=3),
#                     name='X-Axis'
#                 ))
#                 fig.add_trace(go.Scatter3d(
#                     x=[0, 0], y=[all_min, all_max], z=[0, 0],
#                     mode='lines',
#                     line=dict(color='green', width=3),
#                     name='Y-Axis'
#                 ))
#                 fig.add_trace(go.Scatter3d(
#                     x=[0, 0], y=[0, 0], z=[all_min, all_max],
#                     mode='lines',
#                     line=dict(color='blue', width=3),
#                     name='Z-Axis'
#                 ))
#
#                 # Update layout
#                 fig.update_traces(marker=dict(size=3))
#                 fig.update_layout(
#                     scene=dict(
#                         aspectmode='manual',  # Use manual to enforce aspect ratios
#                         aspectratio=dict(x=1, y=1, z=1),  # Equal aspect ratio
#                         xaxis=dict(range=[all_min, all_max]),
#                         yaxis=dict(range=[all_min, all_max]),
#                         zaxis=dict(range=[all_min, all_max])
#                     ),
#                     updatemenus=[
#                         dict(
#                             type="buttons",
#                             direction="left",
#                             buttons=[
#                                 dict(
#                                     label="Show Point Cloud",
#                                     method="update",
#                                     args=[{"visible": [True, False, True, True, True]},
#                                           # Point Cloud visible, other traces controlled
#                                           {"title": "3D Point Cloud by Wall Label"}]
#                                 ),
#                                 dict(
#                                     label="Show Line",
#                                     method="update",
#                                     args=[{"visible": [False, True, True, True, True]},  # Line visible, axes visible
#                                           {"title": "Line by Coordinates"}]
#                                 ),
#                                 dict(
#                                     label="Show Both",
#                                     method="update",
#                                     args=[{"visible": [True, True, True, True, True]},  # All visible
#                                           {"title": "3D Point Cloud and Line"}]
#                                 )
#                             ],
#                             showactive=True,
#                         )
#                     ],
#                     legend=dict(
#                         itemsizing='constant',
#                         itemwidth=30,
#                         tracegroupgap=0,
#                         font=dict(size=12)
#                     )
#                 )
#
#                 if idx == inspect:
#                     fig.show()
#                     print('hi')
#
#
#
#
#
#             # for each line find points in floor_df that are close to the line in their x and y coordinates and assign them wall_cluster label
#             # Add a wall_label column initialized to -1 (default for no label)
#
#             # # Extract points
#             # points = floor_df[['x', 'y']].to_numpy()
#             #
#             # # Extract line start and end points
#             # x1, y1, x2, y2 = lines[:, 0, 0], lines[:, 0, 1], lines[:, 0, 2], lines[:, 0, 3]
#             #
#             # # Calculate the distances for all points to all lines
#             # A = y2 - y1
#             # B = -(x2 - x1)
#             # C = x2 * y1 - y2 * x1
#             #
#             # # Distance calculation
#             # distances = np.abs(A[:, np.newaxis] * points[:, 0] + B[:, np.newaxis] * points[:, 1] + C[:, np.newaxis])
#             # distances /= np.sqrt(A[:, np.newaxis] ** 2 + B[:, np.newaxis] ** 2)
#             #
#             # # Find the nearest line for each point within 5 units
#             # near_lines = np.where(distances <= 5)
#             #
#             # # Update the wall_label column
#             # floor_df.loc[near_lines[1], 'wall_label'] = near_lines[0]
#
#
#         print(f"Floor has {floor_df.shape} points")
#         # Randomly generate distinct colors for each wall_label
#         unique_labels = floor_df['wall_label'].unique()
#         num_labels = len(unique_labels)
#         colors = np.random.choice(px.colors.qualitative.Dark24, size=num_labels, replace=True)
#
#         # Map labels to colors
#         color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
#         floor_df['color'] = floor_df['wall_label'].map(color_map)
#         # Determine the overall range for all axes
#         all_min = min(floor_df['x'].min(), floor_df['y'].min(), floor_df['z'].min())
#         all_max = max(floor_df['x'].max(), floor_df['y'].max(), floor_df['z'].max())
#         samp = min(50_000, int(floor_df[floor_df['wall_label'] != -1].shape[0]))
#         # Plot points with Plotly 3D and visualize by wall_label
#         fig = px.scatter_3d(
#             floor_df[floor_df['wall_label'] != -1].sample(samp),
#             x='x', y='y', z='z',
#             color='wall_label',
#             color_discrete_map=color_map,
#             labels={'wall_label': 'Wall Label'},
#             title=f'3D Point Cloud by Wall Label'
#         )
#
#         # Add lines to the plot
#
#         for idx, line in enumerate(lines):
#             y1, x1, y2, x2 = line[0]
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=[x1, x2], y=[y1, y2], z=[all_min, all_min],  # Assume z remains constant
#                     mode='lines',
#                     line=dict(color='black', width=2),
#                     name=f"Line {idx} ({x1},{y1}) to ({x2},{y2})"
#                 )
#             )
#
#         fig.update_traces(marker=dict(size=3))
#
#         # Add colored axes
#         fig.add_trace(go.Scatter3d(
#             x=[all_min, all_max], y=[0, 0], z=[0, 0],
#             mode='lines',
#             line=dict(color='red', width=3),
#             name='X-Axis'
#         ))
#         fig.add_trace(go.Scatter3d(
#             x=[0, 0], y=[all_min, all_max], z=[0, 0],
#             mode='lines',
#             line=dict(color='green', width=3),
#             name='Y-Axis'
#         ))
#         fig.add_trace(go.Scatter3d(
#             x=[0, 0], y=[0, 0], z=[all_min, all_max],
#             mode='lines',
#             line=dict(color='blue', width=3),
#             name='Z-Axis'
#         ))
#
#         # Update layout
#         fig.update_traces(marker=dict(size=3))
#
#         fig.update_layout(
#             scene=dict(
#                 aspectmode='manual',  # Use manual to enforce aspect ratios
#                 aspectratio=dict(x=1, y=1, z=1),  # Equal aspect ratio
#                 xaxis=dict(range=[all_min, all_max]),
#                 yaxis=dict(range=[all_min, all_max]),
#                 zaxis=dict(range=[all_min, all_max])
#             ),
#             updatemenus=[
#                 dict(
#                     type="buttons",
#                     direction="left",
#                     buttons=[
#                         dict(
#                             label="Show Point Cloud",
#                             method="update",
#                             args=[{"visible": [True, False, True, True, True]},
#                                   # Point Cloud visible, other traces controlled
#                                   {"title": "3D Point Cloud by Wall Label"}]
#                         ),
#                         dict(
#                             label="Show Line",
#                             method="update",
#                             args=[{"visible": [False, True, True, True, True]},  # Line visible, axes visible
#                                   {"title": "Line by Coordinates"}]
#                         ),
#                         dict(
#                             label="Show Both",
#                             method="update",
#                             args=[{"visible": [True, True, True, True, True]},  # All visible
#                                   {"title": "3D Point Cloud and Line"}]
#                         )
#                     ],
#                     showactive=True,
#                 )
#             ],
#             legend=dict(
#                 itemsizing='constant',
#                 itemwidth=30,
#                 tracegroupgap=0,
#                 font=dict(size=12)
#             )
#         )
#         # fig.show()
#         print('hi')
#
#
#         dfs.append(floor_df)
#
#
#     df_floored = pd.concat(dfs)
#     return df_floored


if __name__ == "__main__":
    TRAIN_DATASET = DLRGroupDatasetWholeScenePoints(data_root=r"D:\Datasets\PointClouds\nps", split='train',
                                                    test_project="MorrisCollege_Pinson",
                                                    labels_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt")

    dataset = TRAIN_DATASET[5]
    df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'r', 'g', 'b'])



    # plot df
    all_min = min(df['x'].min(), df['y'].min(), df['z'].min())
    all_max = max(df['x'].max(), df['y'].max(), df['z'].max())
    z_max = df['z'].max()
    samp = min(50_000, int(df.shape[0]))
    # Plot points with Plotly 3D and visualize by wall_label
    fig = px.scatter_3d(
        df,
        x='x', y='y', z='z',
        # color_discrete_map=color_map,
        title=f'3D Point Cloud by Wall Label'
    )

    # Add lines to the plot

    fig.update_traces(marker=dict(size=1))

    # Add colored axes
    fig.add_trace(go.Scatter3d(
        x=[all_min, all_max], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=3),
        name='X-Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[all_min, all_max], z=[0, 0],
        mode='lines',
        line=dict(color='green', width=3),
        name='Y-Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[all_min, z_max],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Z-Axis'
    ))

    # Update layout
    fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        scene=dict(
            aspectmode='manual',  # Use manual to enforce aspect ratios
            aspectratio=dict(x=1, y=1, z=1 / 10),  # Equal aspect ratio
            xaxis=dict(range=[all_min, all_max]),
            yaxis=dict(range=[all_min, all_max]),
            zaxis=dict(range=[all_min, z_max])
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Show Point Cloud",
                        method="update",
                        args=[{"visible": [True, False, True, True, True]},
                              # Point Cloud visible, other traces controlled
                              {"title": "3D Point Cloud by Wall Label"}]
                    ),
                    dict(
                        label="Show Line",
                        method="update",
                        args=[{"visible": [False, True, True, True, True]},  # Line visible, axes visible
                              {"title": "Line by Coordinates"}]
                    ),
                    dict(
                        label="Show Both",
                        method="update",
                        args=[{"visible": [True, True, True, True, True]},  # All visible
                              {"title": "3D Point Cloud and Line"}]
                    )
                ],
                showactive=True,
            )
        ],
        legend=dict(
            itemsizing='constant',
            itemwidth=30,
            tracegroupgap=0,
            font=dict(size=12)
        )
    )
    fig.show()


    print("New Dataset")
    print(df.shape)

    df_floored = detect_and_label_walls(df)

    all_min = min(df_floored['x'].min(), df_floored['y'].min(), df_floored['z'].min())
    all_max = max(df_floored['x'].max(), df_floored['y'].max(), df_floored['z'].max())
    z_max = df_floored['z'].max()
    samp = min(50_000, int(df_floored[df_floored['wall_label'] != -1].shape[0]))
    # Plot points with Plotly 3D and visualize by wall_label
    fig = px.scatter_3d(
        df_floored[df_floored['wall_label'] != -1].sample(samp),
        x='x', y='y', z='z',
        color='wall_label',
        # color_discrete_map=color_map,
        labels={'wall_label': 'Wall Label'},
        title=f'3D Point Cloud by Wall Label'
    )

    # Add lines to the plot


    fig.update_traces(marker=dict(size=1))

    # Add colored axes
    fig.add_trace(go.Scatter3d(
        x=[all_min, all_max], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='red', width=3),
        name='X-Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[all_min, all_max], z=[0, 0],
        mode='lines',
        line=dict(color='green', width=3),
        name='Y-Axis'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[all_min, z_max],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Z-Axis'
    ))

    # Update layout
    fig.update_traces(marker=dict(size=3))

    fig.update_layout(
        scene=dict(
            aspectmode='manual',  # Use manual to enforce aspect ratios
            aspectratio=dict(x=1, y=1, z=1/10),  # Equal aspect ratio
            xaxis=dict(range=[all_min, all_max]),
            yaxis=dict(range=[all_min, all_max]),
            zaxis=dict(range=[all_min, z_max])
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Show Point Cloud",
                        method="update",
                        args=[{"visible": [True, False, True, True, True]},
                              # Point Cloud visible, other traces controlled
                              {"title": "3D Point Cloud by Wall Label"}]
                    ),
                    dict(
                        label="Show Line",
                        method="update",
                        args=[{"visible": [False, True, True, True, True]},  # Line visible, axes visible
                              {"title": "Line by Coordinates"}]
                    ),
                    dict(
                        label="Show Both",
                        method="update",
                        args=[{"visible": [True, True, True, True, True]},  # All visible
                              {"title": "3D Point Cloud and Line"}]
                    )
                ],
                showactive=True,
            )
        ],
        legend=dict(
            itemsizing='constant',
            itemwidth=30,
            tracegroupgap=0,
            font=dict(size=12)
        )
    )
    fig.show()

    print('hi')
