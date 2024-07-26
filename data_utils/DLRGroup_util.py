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

# Classes
classes = ['Floor', 'Walls', 'Doors', 'Others']
# Class Labels
class2label = {cls: i for i, cls in enumerate(classes)}


def create_labels(anno_path, project_folder, filename, lookup_path):
    """
    Creates laebls and save the data to .npy file.

    Args:
    anno_path: Path to the annotation CSV file.
    project_folder: Directory where the subfiles should be saved.
    base_filename: Base filename for the subfiles.
    """

    df_lookup = pd.read_csv(lookup_path)
    lookup = df_lookup.set_index('Category')['Class'].to_dict()
    classes = pd.Series(list(lookup.values())).unique()
    class2label = {cls: i for i, cls in enumerate(classes)}
    required_columns = {'X', 'Y', 'Z', 'R', 'G', 'B', 'ElementType'}
    with open('labels_clean.txt', 'w') as file:
        for item in classes:
            file.write(f"{item}\n")

    if os.path.exists(anno_path):
        # Reach each csv files
        df = pd.read_csv(anno_path)

        points_list = []
        if required_columns.issubset(df.columns):
            df['ElementType'] = [lookup[v] for v in df['ElementType']]
            # All Unique labels
            element_types = df["ElementType"].unique()


            for element_type in element_types:
                if not os.path.exists(filename):
                    filter_df = df[df["ElementType"] == element_type]
                    data = filter_df[['X', 'Y', 'Z', 'R', 'G', 'B']].values
                    labels = np.ones((data.shape[0], 1)) * class2label[element_type]
                    points_list.append(np.concatenate([data, labels], 1))  # Nx7

            data_label = np.concatenate(points_list, 0).astype(np.float64)  # Convert it to float 64
            # Get min value for XYZ along axis 0
            xyz_min = np.amin(data_label, axis=0)[0:3]
            # Less min value from points
            data_label[:, 0:3] -= xyz_min
            output_filename = os.path.join(project_folder, filename)
            np.save(output_filename + '.npy', data_label)
            logging.info(f"Saved {output_filename}")


# -------------------------------------------------------------------------------------------------------


def prepare_data(input_dir, output_dir):
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

                create_labels(file_path, annot_folder, out_filename, "Label_Lookup.csv")


# -------------------------------------------------------------------------------------------------------


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


# -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    connection_string = os.getenv("CONNECTION_STRING")
    container_name = os.getenv("CONTAINER_NAME")

    # Path to the input csv Directory where we want to download both clustered and unclustered data
    input_dir = r'D:\Datasets\PointClouds\csvs'
    # Path to the Output Directory
    output_dir = r'D:\Datasets\PointClouds\nps'
    # Path to the output label directory

    # download_blobs(connection_string, container_name, input_dir)

    prepare_data(input_dir, output_dir)