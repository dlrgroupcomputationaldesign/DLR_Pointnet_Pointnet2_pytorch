import os
import sys
import argparse
import numpy as np
import logging
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_subfiles(anno_path, project_folder, filename):
    """
    Creates subfiles based on element types and saves them as .npy files.

    Args:
    anno_path: Path to the annotation CSV file.
    project_folder: Directory where the subfiles should be saved.
    base_filename: Base filename for the subfiles.
    """
    required_columns = {'X', 'Y', 'Z', 'R', 'G', 'B', 'ElementType'}
    if os.path.exists(anno_path):
        # Reach each csv files
        df = pd.read_csv(anno_path)

        if required_columns.issubset(df.columns):
          # All Unique labels
          element_types = df["ElementType"].unique()
          # Save data for each unique class to .npy file
          for element_type in element_types:
              new_filename = filename + '_' + element_type + '.npy'
              output_filename = os.path.join(project_folder, new_filename)

              if not os.path.exists(output_filename):
                  filter_df = df[df["ElementType"] == element_type]

                  data = filter_df[['X', 'Y', 'Z', 'R', 'G', 'B', 'ElementType']].values
                  np.save(output_filename, data)
                  logging.info(f"Saved {output_filename}")
#-------------------------------------------------------------------------------------------------------


def extract_and_save_classes(main_folder, output_folder, output_filename="labels.txt"):
    """
    Extracts unique labels from .npy filenames and saves them to a text file.

    Args:
    main_folder: Main directory containing subfolders with .npy files.
    output_folder: Directory where the output text file should be saved.
    output_filename: Name of the output text file.
    """
    # set to store all labels
    labels_set = set()
    # Walk over the main folder which contain all the .npy files eg. C:\DLR_Pointnet_Pointnet2_pytorch-master\data\annotations
    for root, dirs, files in os.walk(main_folder):
        
        for file in files:
            if file.endswith('.npy'):
                base_filename = file[:-4]
                parts = base_filename.split('_')
                
                if len(parts) > 1:
                        last_label = parts[-1]
                        labels_set.add(last_label)
                        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Create the full path for the output file
    output_file_path = os.path.join(output_folder, output_filename)

    # Write the labels to the output text file
    with open(output_file_path, 'w') as f:
        for label in labels_set:
            f.write(f"{label}\n")
    logging.info(f"Labels saved to {output_file_path}")
    
    return labels_set

#-------------------------------------------------------------------------------------------------------


def create_and_save_labelled_data(output_dir, labels_set):
    """
    Creates labelled data by combining points with their corresponding labels and saves them.

    Args:
    output_dir: Directory where the labelled data should be saved.
    label_mapping: Dictionary mapping labels to numeric values.
    """
    for root, dirs, files in os.walk(output_dir):
        for dir in dirs:
            sub_folder_path = os.path.join(root, dir)
            points_list = []
            for filename in os.listdir(sub_folder_path):
                if filename.endswith('.npy'):
                # Load npy file
                    data_label = np.load(os.path.join(sub_folder_path, filename), allow_pickle=True)
                    # Delete the label as it is a string
                    data = np.delete(data_label, -1, axis=1)
                    # SPlit base file name to extract class
                    cls = filename[:-4].split('_')[-1]
                    labels = np.ones((data.shape[0],1)) * labels_set[cls]
                    points_list.append(np.concatenate([data, labels], 1)) # Nx7
                    os.remove(os.path.join(sub_folder_path, filename))
        
            if points_list:       
                data_label = np.concatenate(points_list, 0).astype(np.float64) # Convert it to float 64
                # Get min value for XYZ along axis 0
                xyz_min = np.amin(data_label, axis=0)[0:3]
                # Less min value from points
                data_label[:, 0:3] -= xyz_min
                np.save(os.path.join(sub_folder_path,  f'{dir}.npy'), data_label)
                logging.info(f"Saved labelled data to {sub_folder_path}/{dir}.npy")


#-------------------------------------------------------------------------------------------------------

def prepare_data(input_dir, output_dir, output_label_dir):
    '''
    Args:
    input_dir: Directory which contain all the datasets .csv
    output_dir: Directory where the output data should be saved
    output_label_dir: Director where the labels should be saved
    '''
    clust_outpu_dir = os.path.join(output_dir, "clustered")
    unclust_output_dir = os.path.join(output_dir, "unclustered")
    if not os.path.exists(clust_outpu_dir):
        os.makedirs(clust_outpu_dir)
    if not os.path.exists(unclust_output_dir):
        os.makedirs(unclust_output_dir)
        

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if 'unclustered' in root:
                    local_folder = unclust_output_dir
                else:
                    local_folder = clust_outpu_dir
    
                base_filename = file
                out_filename, _ = os.path.splitext(base_filename)
                annot_folder = os.path.join(local_folder, out_filename)
                
                if not os.path.exists(annot_folder):
                    os.makedirs(annot_folder)
                    

                create_subfiles(file_path, annot_folder, out_filename)
    # Dynamically create class names    
    class_names = extract_and_save_classes(output_dir, output_label_dir)
    # Assign numerical value to each class
    class2label = {cls: i for i,cls in enumerate(class_names)}
    # Merge all subfiles into one single .npy file and delete all subfiles
    create_and_save_labelled_data(output_dir, class2label)
     


def download_blobs(connection_string, container_name, output_dir):
    '''
    Download both Clustered and Unclustered data
    '''
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # Create a ContainerClient to interact with the container
    container_client = blob_service_client.get_container_client(container_name)

    for blob in container_client.list_blobs():
        blob_name = blob.name
        print(f"Processing blob: {blob_name}")

        # Check if the blob is in the 'unclustered' folder or directly in 'point_cloud'
        if blob_name.startswith('point_cloud/unclustered/') and not blob_name.endswith('/'):
            # Create the full local path for the downloaded file in the same structure
            local_file_path = os.path.join(output_dir, os.path.relpath(blob_name, 'point_cloud'))
        elif blob_name.count('/') == 1 and blob_name.endswith('.csv'):
            # Change the path to 'clustered' for CSV files directly under 'point_cloud'
            local_file_path = os.path.join(output_dir, 'clustered', os.path.relpath(blob_name, 'point_cloud'))
        else:
            continue
        local_dir = os.path.dirname(local_file_path)
        
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            print(f"Downloading: {blob_name} to {local_file_path}")
            with open(local_file_path, "wb") as download_file:
                download_file.write(container_client.download_blob(blob_name).readall())
        except Exception as e:
            print(f"Error downloading {blob_name}: {e}")


#-------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    connection_string = os.getenv("CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME")
    

    # Path to the input csv Directory where I want to download both clustered and unclustered data
    input_dir = r'C:\DLR_Pointnet_Pointnet2_pytorch-master\data\csv_files'
    # Path to the Output Directory
    output_dir = r'C:\DLR_Pointnet_Pointnet2_pytorch-master\data\annotations'
    # Path to the output label directory
    output_label_dir = r'C:\DLR_Pointnet_Pointnet2_pytorch-master\data_utils\meta'
    
    # I have a download file, i will make a copy of the same file for test
    # I will comment the below function as i already have some data
    # download_blobs(connection_string, container_name, input_dir)
    
    
    
    prepare_data(input_dir, output_dir, output_label_dir)


