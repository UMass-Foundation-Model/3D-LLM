import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tmp = torch.zeros(1).to(device)  # for device check


def normalize_pc(points):
    centroid = torch.mean(points, dim=0)
    points -= centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(torch.abs(points) ** 2, dim=-1)), dim=0)[0]
    points /= furthest_distance

    return centroid, furthest_distance, points


def uvd2xyz(depth, K, extrinsic, depth_trunc=np.inf):
    """
    depth: of shape H, W
    K: 3, 3
    extrinsic: 4, 4
    return points: of shape H, W, 3
    """
    depth[depth > depth_trunc] = 0
    H, W = depth.shape
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    x = np.arange(0, W) - cx
    y = np.arange(0, H) - cy
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    points = points * depth[..., None]
    points[..., 0] /= fx
    points[..., 1] /= fy
    points = points.reshape(-1, 3)
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    points = points @ np.linalg.inv(extrinsic).T
    points = points[:, :3].reshape(H, W, 3)
    return points


def gen_points(scene, path, file_list):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        print("no meta.json")
        return None
    meta = json.load(open(meta_path, "r"))

    points = []
    poses = []
    for file in file_list:
        key = os.path.join(path, file)
        value = meta[os.path.join(objaverse_save_dir, scene, file)]

        depth_path = f"{key.replace('view_', 'depth_0')}.exr"

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

        pose = np.array(value["extrinsic"]).reshape(4, 4)
        poses.append(pose)

        K = np.array(value["intrinsic"])

        depth = np.array(depth_raw)[..., 0]
        pcd = uvd2xyz(depth, K, pose)  # H, W, 3
        points.append(pcd)

    if len(points) == 0:
        return None
    points = np.stack(points, axis=0)
    return points


project_path = "./data"
scene_path = os.path.join(project_path, "objaverse_frame")
feat_path = os.path.join(project_path, "objaverse_2dfeat")
out_path = os.path.join(project_path, "objaverse_feat")

feat_folder = "nps_1024_hiddAve_ViTL"

scene_list = os.listdir(scene_path)
scene_list.sort()

objaverse_save_dir = "output/"
sample_num = 50000
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(out_path, "points"), exist_ok=True)
os.makedirs(os.path.join(out_path, "features"), exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job", default=0, type=int)
parser.add_argument("--all_jobs", default=160, type=int)
args = parser.parse_args()
N = len(scene_list)
jobs = args.all_jobs
ids = N // jobs + 1
for scene in tqdm(scene_list[args.job * ids : min(N, args.job * ids + ids)]):
    path = os.path.join(out_path, scene)
    scene_feat_path = os.path.join(feat_path, scene, feat_folder)
    if not os.path.exists(scene_feat_path) or len(os.listdir(scene_feat_path)) == 0:
        continue
    print(scene)
    feat_avail_file_list = [f.replace(".pt", "") for f in os.listdir(scene_feat_path) if f.endswith(".pt")]
    # hard code: remove outsied view 0
    feat_avail_file_list = [f for f in feat_avail_file_list if int(f[-1]) != 0 or "inside" in f]

    try:
        for render_type in ["outside"]:
            point_save_path = os.path.join(out_path, "points", f"{scene}_{render_type}.npy")
            feat_save_path = os.path.join(out_path, "features", f"{scene}_{render_type}.pt")
            if os.path.exists(point_save_path) and os.path.exists(feat_save_path):
                print("skip")
                continue
            os.remove(point_save_path) if os.path.exists(point_save_path) else None
            os.remove(feat_save_path) if os.path.exists(feat_save_path) else None

            file_list = sorted([f for f in feat_avail_file_list if render_type in f])
            points = gen_points(scene, os.path.join(scene_path, scene), file_list)  # N, H, W, 3
            if points is None:
                print("no valid points: points is None")
                continue
            points_norm = np.linalg.norm(points, axis=-1)  # N, H, W
            masks = np.logical_and(points_norm > 0, points_norm < 10000)  # N, H, W
            points = points[masks]  # N, 3
            if np.sum(masks) == 0:
                print("no valid points: mask all zero")
                continue

            features = []
            scene_feat_list = [f + ".pt" for f in file_list]
            failed_idx = []
            for i, pf in enumerate(scene_feat_list):
                try:
                    features.append(
                        torch.load(os.path.join(scene_feat_path, pf), map_location="cpu").numpy()
                    )  # H, H, 1024
                except:
                    failed_idx.append(i)
            if len(failed_idx) > 0:
                masks = np.delete(masks, failed_idx, axis=0)
            features = np.stack(features, axis=0)  # N, H, W, 1024
            features = features[masks]  # N, 1024
            features = torch.from_numpy(features)
            points = torch.from_numpy(points)

            c, v, points = normalize_pc(points)

            # remove the points with all zero features
            feature_sum = torch.sum(features, dim=1)  # N, 1024
            feature_indices = torch.where(feature_sum)[0]  # N

            final_points = points[feature_indices]
            final_features = features[feature_indices]
            # random sample
            N = final_points.shape[0]
            if N == 0:
                print("no valid points: all zero features")
                continue
            if N > sample_num:
                indices = np.random.choice(N, sample_num, replace=False)
                final_points = final_points[indices]
                final_features = final_features[indices]

            # ======== voxelize the features and points ========
            points = final_points.detach().cpu().numpy()
            features = final_features

            points = (points * 128 + 128).astype(int)  # N, 3
            point_dict = defaultdict(list)
            for point, feature in zip(points, features):
                point_dict[(point[0], point[1], point[2])].append(feature)

            myKeys = list(point_dict.keys())
            myKeys.sort()
            point_dict = {i: point_dict[i] for i in myKeys}

            final_points = []
            final_features = []

            for point, feature_list in point_dict.items():
                feature_list = torch.stack(feature_list)
                final_points.append([point[0], point[1], point[2]])
                feature = torch.mean(feature_list, 0)
                final_features.append(feature)

            final_points = np.array(final_points)
            final_features = torch.stack(final_features).detach().cpu().numpy()
            print(final_points.shape, final_features.shape)

            np.save(point_save_path, final_points)
            torch.save(final_features, feat_save_path)
            # print("saved to", point_save_path, feat_save_path)

            # remove scene_feat_path
            print(f"remove {scene_feat_path}")
            for f in os.listdir(scene_feat_path):
                os.remove(os.path.join(scene_feat_path, f))

        # del points, features, point_dict, final_points, final_features
    except Exception as e:
        print(f"{scene} error: {e}")
