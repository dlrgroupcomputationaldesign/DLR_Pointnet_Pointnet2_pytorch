import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd

def visualize_npy(npy_file):
    # Load the npy file: x, y, z, r, g, b, l
    data = np.load(npy_file) 
    xyz = data[:, :3].astype(np.float32)  # Convert XYZ to float32
    rgb = data[:, 3:6].astype(np.float32) / 255.0  # Normalize RGB and convert to float32

    # Check if RGB has the correct shape
    if rgb.shape[1] != 3:
        raise ValueError(f"RGB values should have shape (N, 3), but got {rgb.shape}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)  # Assign points
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # Assign colors

    # Visualize
    o3d.visualization.draw_geometries([pcd])

def visualize_csv(csv_file, use_pred_labels=True):
    data = pd.read_csv(csv_file, skiprows=1, header=None)
    xyz = data.iloc[:, :3].values  # First 3 columns: x, y, z
    rgb = data.iloc[:, 3:6].values / 255.0  # Normalize RGB and convert to float32

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)  # Assign points
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # Assign colors

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name="Raw Point Cloud Visualization")

    gt_labels = data.iloc[:, -2].values.astype(int)  # two from the last column: ground truth label
    pred_labels = data.iloc[:, -1].values  # Last column: predicted label
    labels = pred_labels if use_pred_labels else gt_labels  # Choose which label to use
    # Get unique classes and generate distinct colors
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    
    # Generate a colormap (you can customize this)
    colormap = plt.get_cmap("tab10", num_classes)  # "tab10" provides 10 distinct colors

    # Assign a color to each class
    colors = np.array([colormap(i)[:3] for i in range(num_classes)])  # Get RGB colors

    # Map each point's label to a color
    point_colors = colors[labels]  # Assign colors based on labels

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float32))  # Ensure float32

    # Visualize
    o3d.visualization.draw_geometries([pcd])


npy_file = r"Datasets\PointClouds\nps2\clustered\00-10231-20_CortevaYorkTest\00-10231-20_CortevaYorkTest.npy"

visualize_npy(npy_file)

csv_file = "DLR_Pointnet_Pointnet2_pytorch/WyomingStateFair_Laramie_Output_Weds.csv"
visualize_csv(csv_file, use_pred_labels=False)




