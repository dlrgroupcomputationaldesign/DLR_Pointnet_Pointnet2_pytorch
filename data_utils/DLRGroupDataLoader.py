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