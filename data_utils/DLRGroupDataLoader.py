# TODO Implement DLR Group Point Cloud dataset
"""
The DLR Groyp Point Cloud dataset should opperate just like Pointnet_Pointnet2_pytorch/data_utils/S3DISDataLoader.py S3DISDataLoader class,
but pull data in the form defined @ https://portal.azure.com/#@dlrgroup.com/resource/subscriptions/182a471a-5634-4d1a-a722-f7779cf5d470/resourceGroups/data_science/providers/Microsoft.Storage/storageAccounts/dlrdatalake/storagebrowser

Features:
 * be able to set a selection of projects via a list (default to all projects)
 * be ablel to define clustered vs unclustered, if clustered use the data in https://portal.azure.com/#@dlrgroup.com/resource/subscriptions/182a471a-5634-4d1a-a722-f7779cf5d470/resourceGroups/data_science/providers/Microsoft.Storage/storageAccounts/dlrdatalake/storagebrowser
   if unclusterd use the unclustered version in the unclustered folder


Creates:
 * A dataset that can be accessed and read in the same way that S3DISDataLoader.py/S3DISDataset is in train_semseg.py
"""

import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class DLRGroupDataset(Dataset):
    def __init__(self, labels_path, test_project, split='train', data_root='trainval_fullarea',
                 data_type="clustered",num_point=4096, block_size=2.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        # Read Label File
        with open(labels_path, 'r') as file:
            self.labels_length = len([line.strip() for line in file])
            
        # Build Complete path for all .npy files
        all_projects = []
        for root, dirs, files in os.walk(data_root):
            if data_type in root:
                for file in files:
                    if file.endswith('.npy'):
                        full_path = os.path.join(root, file)
                        all_projects.append(full_path)

        # Split Data Between Train and test
        if split == 'train':
            projects_split = []
            for project in all_projects:
                basename = os.path.basename(project)[:-4]
                if basename != test_project: # Test Project is the name of the project we need to test on eg. 00-10231-20_CortevaYork
                    projects_split.append(project) 
        else:
            projects_split = []
            for project in all_projects:
                basename = os.path.basename(project)[:-4]
                if basename == test_project:
                    projects_split.append(project) 
                    
        self.room_points, self.room_labels = [], []
        # Coord Min and Max
        self.room_coord_min, self.room_coord_max = [], []
        # All Number of points
        num_point_all = []
        # Class Weights
        labelweights = np.zeros(self.labels_length)
        # Iterate over each Room
        for room_path in tqdm(projects_split, total=len(projects_split)):
            # Load Room We are Considering One csv file as a Room / Project
            room_data = np.load(room_path)  # xyzrgbl, N*7
            
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            # Return distribution of labels
            tmp, _ = np.histogram(labels, range(self.labels_length+1))
            # Add to labelweights
            labelweights += tmp
            # Calculate min and max for points
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
            
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  
          
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(projects_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            # We have changed Black size from 1 to 2
            # Block size 1 does not return block where we have points > 1024 migt be due to Sparse Point Cloud
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0] 
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        
        
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        
        
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)
    
    
if __name__ == '__main__':
    data_root = r'E:\DLR_Pointnet_Pointnet2_pytorch-master\data\annotations' # Data is the path where you have saved all the .npy files it should be root folder not sub folders
    labels_path = r'E:\DLR_Pointnet_Pointnet2_pytorch-master\data_utils\meta\labels.txt'# Path where we have saved our labels
    num_point, block_size, sample_rate = 4096, 2.0, 1.0
    test_project = 'test' # Name of the Project which we want to consider for test
    
    point_data = DLRGroupDataset(split='train', labels_path=labels_path,
                                 data_root=data_root, num_point=num_point,
                                 test_project=test_project, block_size=block_size,
                                 sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
