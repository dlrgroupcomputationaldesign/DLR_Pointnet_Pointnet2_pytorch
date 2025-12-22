import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from .models import get_model_module

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B, N = pred_label.shape[0], pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def infer_whole_scenes(
    dataset_path,     # npy file
    csv_out_dir,      # output path csv file
    model,            # import model module 
    model_path,       # model checkpoint path
    device=0,         # GPU device
    num_points=4096,  # number of points per block
    batch_size=32,   
    log_dir="pointnet2_sem_seg",
    num_votes=3,      # number of votes
    stride=15.0,      # stride for block sampling
    block_size=100.0,  
    padding=0.001,    # padding for block sampling
    ):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    experiment_dir = os.path.join("log", "sem_seg", log_dir)
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Infer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(experiment_dir, "infer.txt"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    def log_string(msg):
        logger.info(msg)
        print(msg)

    labels = ['Other', 'Floor', 'Ceiling', 'Wall']
    num_classes = len(labels)

    log_string(f"Found {num_classes} classes from {labels}")

    # model
    MODEL = get_model_module(model)
    classifier = MODEL.get_model(num_classes).cuda().eval()
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint["model_state_dict"])

    data = np.load(dataset_path)
    points_xyzrgb = data[:, :6]

    with torch.no_grad():
        points = points_xyzrgb.copy()
        xyz = points[:, :3]

        coord_min = np.amin(xyz, axis=0)
        coord_max = np.amax(xyz, axis=0)

        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)

        data_room_list = []
        weight_list = []
        index_list = []

        for iy in range(grid_y):
            for ix in range(grid_x):
                s_x = coord_min[0] + ix * stride
                e_x = min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                s_y = coord_min[1] + iy * stride
                e_y = min(s_y + block_size, coord_max[1])
                s_y = e_y - block_size

                point_idxs = np.where(
                    (xyz[:, 0] >= s_x - padding) & (xyz[:, 0] <= e_x + padding) &
                    (xyz[:, 1] >= s_y - padding) & (xyz[:, 1] <= e_y + padding)
                )[0]
                if point_idxs.size == 0:
                    continue

                num_batch = int(np.ceil(point_idxs.size / num_points))
                point_size = int(num_batch * num_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)

                data_batch = points[point_idxs, :]  # (point_size, 6)
                
                norm_xyz = np.zeros((point_size, 3))
                norm_xyz[:, 0] = data_batch[:, 0] / (coord_max[0] if coord_max[0] != 0 else 1.0)
                norm_xyz[:, 1] = data_batch[:, 1] / (coord_max[1] if coord_max[1] != 0 else 1.0)
                norm_xyz[:, 2] = data_batch[:, 2] / (coord_max[2] if coord_max[2] != 0 else 1.0)

                data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
                data_batch[:, 3:6] /= 255.0  # rgb

                data_batch = np.concatenate((data_batch, norm_xyz), axis=1)  # (point_size, 9)

                # For inference, keep weight=1 for all points (vote all points equally)
                w = np.ones((point_size,), dtype=np.float32)
                
                # inside the (iy, ix) loops, after you build data_batch, w, point_idxs
                data_room_list.append(data_batch)          # (point_size, 9)
                weight_list.append(w)                      # (point_size,)
                index_list.append(point_idxs.astype(np.int64))  # (point_size,)

        if len(data_room_list) == 0:
            raise RuntimeError("No blocks were created (data_room_list is empty). Check stride/block_size/padding.")
        
        data_room = np.concatenate(data_room_list, axis=0)       # (total_points_in_blocks, 9)
        sample_weight = np.concatenate(weight_list, axis=0)      # (total_points_in_blocks,)
        index_room = np.concatenate(index_list, axis=0)          # (total_points_in_blocks,)

        assert data_room.ndim == 2, f"data_room should be 2D here, got {data_room.ndim}D {data_room.shape}"
        assert data_room.shape[1] == 9, f"expected 9 features, got {data_room.shape}"
        assert sample_weight.ndim == 1
        assert index_room.ndim == 1

        data_room = data_room.reshape((-1, num_points, data_room.shape[1]))
        sample_weight = sample_weight.reshape((-1, num_points))
        index_room = index_room.reshape((-1, num_points))

        num_blocks = data_room.shape[0]
        vote_label_pool = np.zeros((points.shape[0], num_classes), dtype=np.float32)

        for _ in tqdm(range(num_votes), total=num_votes):
            s_batch_num = (num_blocks + batch_size - 1) // batch_size

            for sb in range(s_batch_num):
                start = sb * batch_size
                end = min((sb + 1) * batch_size, num_blocks)
                real_bs = end - start

                batch_data = np.zeros((batch_size, num_points, 9), dtype=np.float32)
                batch_index = np.zeros((batch_size, num_points), dtype=np.int64)
                batch_w = np.zeros((batch_size, num_points), dtype=np.float32)

                batch_data[:real_bs] = data_room[start:end]
                batch_index[:real_bs] = index_room[start:end]
                batch_w[:real_bs] = sample_weight[start:end]

                torch_data = torch.from_numpy(batch_data).float().cuda().transpose(2, 1)
                seg_pred, _ = classifier(torch_data)
                pred = seg_pred.contiguous().cpu().data.max(2)[1].numpy()  # (B,N)

                vote_label_pool = add_vote(
                    vote_label_pool,
                    batch_index[:real_bs],
                    pred[:real_bs],
                    batch_w[:real_bs],
                )

        pred_label = np.argmax(vote_label_pool, axis=1).astype(np.int32)

        out = np.hstack([points_xyzrgb, pred_label.reshape(-1, 1)])
        df = pd.DataFrame(out, columns=["x", "y", "z", "r", "g", "b", "pred_class"])

        df.to_csv(csv_out_dir, index=False)
        print(f"Saved: {csv_out_dir}")

    log_string("Inference complete.")
    
    return df

