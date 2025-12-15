from __future__ import annotations

import os
import logging
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_utils.DLRGroupDataLoader import DLRDatasetWholeScene
from data_utils.indoor3d_util import g_label2color


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def run_inference(
    *,
    model: str = "pointnet",
    batch_size: int = 32,
    gpu: str = "0",
    num_point: int = 4096,
    log_dir: str,
    visual: bool = False,
    test_project: str = "MorrisCollege_Pinson",
    num_votes: int = 3,
    data_type: str = "clustered",
    data_dir: str,
    label_path: str,
    trained_model: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Library entry point. Call this from Python code.
    Returns a DataFrame with x,y,z,r,g,b,gt_label,pred_label and also writes CSV.
    """

    # --- env / device ---
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # --- read label list used for NUM_CLASSES ---
    with open(label_path, "r") as f:
        classes = [line.strip() for line in f]
    num_classes = len(classes)

    # --- experiment/visual dirs ---
    experiment_dir = Path("log/sem_seg") / log_dir
    visual_dir = experiment_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # --- logging ---
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(experiment_dir / "eval.txt")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def log_string(s: str):
        logger.info(s)
        print(s)

    log_string("PARAMETER ...")
    log_string(
        f"model={model}, batch_size={batch_size}, gpu={gpu}, num_point={num_point}, "
        f"log_dir={log_dir}, visual={visual}, test_project={test_project}, "
        f"num_votes={num_votes}, data_type={data_type}, data_dir={data_dir}, "
        f"label_path={label_path}, trained_model={trained_model}"
    )

    # --- dataset ---
    test_dataset = DLRDatasetWholeScene(
        root=data_dir,
        block_points=num_point,
        split="test",
        test_project=test_project,
        stride=15.0,
        block_size=100.0,
        padding=0.001,
        labels_path=label_path,
    )
    log_string(f"The number of test data is: {len(test_dataset)}")

    # --- model loading ---
    MODEL = importlib.import_module(f"{model}_sem_seg")
    classifier = MODEL.get_model(num_classes).cuda()

    ckpt_path = experiment_dir / "checkpoints" / trained_model
    checkpoint = torch.load(str(ckpt_path))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = [x[:-4] for x in test_dataset.file_list]
        num_batches = len(test_dataset)

        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        total_iou_deno_class = [0 for _ in range(num_classes)]

        log_string("---- EVALUATION WHOLE SCENE----")

        # NOTE: Your original code writes ONE CSV at the end using whole_scene_data/label
        # from the LAST batch only. If you want per-scene CSVs, move CSV writing inside loop.
        last_whole_scene_data = None
        last_whole_scene_label = None
        last_pred_label = None

        for batch_idx in range(num_batches):
            print(f"Inference [{batch_idx+1}/{num_batches}] {scene_id[batch_idx]} ...")

            total_seen_class_tmp = [0 for _ in range(num_classes)]
            total_correct_class_tmp = [0 for _ in range(num_classes)]
            total_iou_deno_class_tmp = [0 for _ in range(num_classes)]

            if visual:
                fout = open(visual_dir / f"{scene_id[batch_idx]}_pred.obj", "w")
                fout_gt = open(visual_dir / f"{scene_id[batch_idx]}_gt.obj", "w")

            whole_scene_data = test_dataset.scene_points_list[batch_idx]
            whole_scene_label = test_dataset.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], num_classes))

            for _ in tqdm(range(num_votes), total=num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = test_dataset[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + batch_size - 1) // batch_size

                batch_data = np.zeros((batch_size, num_point, 9))
                batch_label = np.zeros((batch_size, num_point))
                batch_point_index = np.zeros((batch_size, num_point))
                batch_smpw = np.zeros((batch_size, num_point))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * batch_size
                    end_idx = min((sbatch + 1) * batch_size, num_blocks)
                    real_batch_size = end_idx - start_idx

                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.tensor(batch_data, dtype=torch.float32).cuda().transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(
                        vote_label_pool,
                        batch_point_index[0:real_batch_size, ...],
                        batch_pred_label[0:real_batch_size, ...],
                        batch_smpw[0:real_batch_size, ...],
                    )

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(num_classes):
                total_seen_class_tmp[l] += np.sum(whole_scene_label == l)
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum((pred_label == l) | (whole_scene_label == l))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float32) + 1e-6)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string(f"Mean IoU of {scene_id[batch_idx]}: {tmp_iou:.4f}")

            # write txt prediction
            pred_txt = visual_dir / f"{scene_id[batch_idx]}.txt"
            with open(pred_txt, "w") as f:
                for i in pred_label:
                    f.write(f"{int(i)}\n")

            # optionally write objs
            if visual:
                for i in range(whole_scene_label.shape[0]):
                    color = g_label2color[pred_label[i]]
                    color_gt = g_label2color[whole_scene_label[i]]
                    fout.write(
                        "v %f %f %f %d %d %d\n"
                        % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2],
                           color[0], color[1], color[2])
                    )
                    fout_gt.write(
                        "v %f %f %f %d %d %d\n"
                        % (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2],
                           color_gt[0], color_gt[1], color_gt[2])
                    )
                fout.close()
                fout_gt.close()

            last_whole_scene_data = whole_scene_data
            last_whole_scene_label = whole_scene_label
            last_pred_label = pred_label

        # final metrics print (same as your script; you can expand as needed)
        log_string("Done!")

        # Build DataFrame (last scene, matching your original behavior)
        df = pd.DataFrame(last_whole_scene_data, columns=["x", "y", "z", "r", "g", "b"])
        df["gt_label"] = last_whole_scene_label
        df["pred_label"] = last_pred_label

        if output_csv is None:
            output_csv = f"{test_project}_Output.csv"
        df.to_csv(output_csv, index=False)

        return df
