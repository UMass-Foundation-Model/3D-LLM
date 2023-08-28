import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=None, type=str)
parser.add_argument("--model", choices=["blip", "clip"], default="clip", type=str)
args = parser.parse_args()


project_path = "./data/"
scene_path = os.path.join(project_path, "objaverse_frame_cap3d")
feat_path = os.path.join(project_path, "objaverse_2dfeat_cap3d")
out_path = os.path.join(project_path, "objaverse_feat_cap3d")

if args.model == "blip":
    feat_folder = "nps_blip"
else:
    feat_folder = "nps_1024_hiddAve_ViTL"

scene_list = os.listdir(scene_path)
scene_list.sort()

objaverse_save_dir = "output/"
sample_num = 50000
os.makedirs(out_path, exist_ok=True)
os.makedirs(os.path.join(out_path, "points"), exist_ok=True)
os.makedirs(os.path.join(out_path, "features"), exist_ok=True)

scene = args.scene
path = os.path.join(out_path, scene)
scene_feat_path = os.path.join(feat_path, scene, feat_folder)
if not os.path.exists(scene_feat_path) or len(os.listdir(scene_feat_path)) == 0:
    print("no feature")
    exit()
print(scene)
feat_avail_file_list = [f.replace(".pt", "") for f in os.listdir(scene_feat_path) if f.endswith(".pt")]
# hard code: remove outsied view 0
feat_avail_file_list = [f for f in feat_avail_file_list if int(f[-1]) != 0 or "inside" in f]

for render_type in ["outside"]:
    point_save_path = os.path.join(out_path, "points", f"{scene}_{render_type}.npy")
    feat_save_path = os.path.join(out_path, "features", f"{scene}_{render_type}.pt")

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
    for pf in scene_feat_list:
        features.append(torch.load(os.path.join(scene_feat_path, pf), map_location="cpu").numpy())  # H, W, 1024
    features = np.stack(features, axis=0)  # N, H, W, 1024
    features = features[masks]  # N, 1024
    features = torch.from_numpy(features)
    points = torch.from_numpy(points)  # N, 3

    # visualize points
    # use tsne to visualize features
    # the color of the points above should be the tsne result
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    tsne = TSNE(n_components=1, random_state=0)
    tsne_result = tsne.fit_transform(features)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=tsne_result[:, 0], cmap="viridis")
    save_path = f"vis/image/{scene}_{render_type}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print("save fig")
