import os
import pandas as pd
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
from data_utils.DLRGroupDataLoader import DLRGroupDataset, DLRDatasetWholeScene
from processing import planer_cluster, vote_cluster, postprocess
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'models'))


# D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\log\sem_seg\pointnet2_sem_seg\checkpoints\model_1.pth
# --test_project
# MorrisCollege_Pinson
# --data_type
# clustered
# --label_path
# D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt
# --data_dir
# D:\Datasets\PointClouds\nps
# --log_dir
# pointnet2_sem_seg
# --output_file
# D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\visualizer\output1.csv

# parser = argparse.ArgumentParser('Model')
#     parser.add_argument("--model_path", type=str, help="Path to the model you would like to test")
#     parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
#     parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
#     parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
#     parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
#     parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
#     parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
#     parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
#     parser.add_argument('--label_path', type=str, required=True,
#                         help='Path where the lables file is stored')  # Added argument for label path
#
#     # Path to the annotation directory
#     parser.add_argument('--data_dir', type=str, required=True,
#                         help='Directory where the data is stored')  # Added argument for data directory
#     parser.add_argument('--test_project', type=str, required=True,
#                         help='Name of the Test Project')  # Added argument for test_poject name
#     parser.add_argument('--data_type', type=str, required=True, help='Type of Data Clustered or Unclustered')
#     parser.add_argument('--output_file', type=str, required=True, help='csv of labeled point cloud')
#     return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def infer_points(points,
                 model="pointnet_sem_seg",
                 model_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\log\sem_seg\pointnet2_sem_seg\checkpoints\model_1.pth",
                 device=0,
                 num_points=4096,
                 label_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt",
                 batch_size=32,
                 log_dir="pointnet2_sem_seg",
                 visual=False,
                 num_votes=3,
                 ):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    experiment_dir = 'log/sem_seg/' + log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    with open(label_path, 'r') as file:
        classes = [line.strip() for line in file]
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    NUM_CLASSES = len(classes)
    NUM_POINT = num_points
    BATCH_SIZE = batch_size

    log_string("The number of test data is: %d" % len(points))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            whole_scene_rooms = TEST_DATASET_WHOLE_SCENE.room_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(num_votes), total=num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)
            # Assuming whole_scene_data and pred_label are already defined numpy arrays
            # Reshape pred_label to match the dimensions required for hstack
            pred_label_reshaped = pred_label.reshape(-1, 1)

            # Concatenate the arrays horizontally
            whole_scene_data_with_labels = np.hstack((whole_scene_data, pred_label_reshaped))

            # output data with labels
            df_whole_scene_data_with_labels = pd.DataFrame(whole_scene_data_with_labels)
            df_whole_scene_data_with_labels.columns = ["x", "y", "z", "r", "g", "b", "pred_l"]
            df_whole_scene_data_with_labels['Room'] = whole_scene_rooms['Room']
            df_whole_scene_data_with_labels['gt'] = whole_scene_label
            #  Cluster by plane and vote
            # df_whole_scene_data_with_labels_clustered = planer_cluster(df_whole_scene_data_with_labels)
            # df_whole_scene_data_with_labels_clean = vote_cluster(df_whole_scene_data_with_labels_clustered)

            return df_whole_scene_data_with_labels

if __name__ == "__main__":
    NUM_POINT = 4096

    TEST_DATASET_WHOLE_SCENE = DLRDatasetWholeScene(root=r"D:\Datasets\PointClouds\nps" , block_points=NUM_POINT, split='test', test_project="MorrisCollege_Pinson", stride=15.0, block_size=100.0, padding=0.001, labels_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt")

    points = TEST_DATASET_WHOLE_SCENE.scene_points_list
    df_points = pd.DataFrame(points[0], columns=['x', 'y', 'z', 'r', 'g', 'b'])

    df = infer_points(df_points,
                      model="pointnet2_sem_seg",
                      model_path=r"D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\log\sem_seg\pointnet2_sem_seg\checkpoints\best_model_BolaModel_30_epoch_blocksize_100.pth",
                      device=0,
                      num_points=4096,
                      label_path="D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\data_utils\labels_clean.txt",
                      batch_size=16,
                      log_dir="pointnet2_sem_seg",
                      visual=False,
                      num_votes=3)

    df = postprocess(df)


    df.to_csv("Morriscollge_Pinson_Output.csv")
    print('done')