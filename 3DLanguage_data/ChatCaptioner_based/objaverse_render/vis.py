import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import torch
import numpy as np
import open3d as o3d
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tmp = torch.zeros(1).to(device)  # for device check
LOAD_IMG_HEIGHT = 240
LOAD_IMG_WIDTH = 320


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
    mask = depth > 0
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
    points = points[mask]
    return points


def gen_points(path):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        print("no meta.json")
        return None
    meta = json.load(open(meta_path, "r"))

    points = []
    poses = []
    file_list = os.listdir(path)
    file_list = [file for file in file_list if file.endswith(".png") and "rgb" in file]
    for file in file_list:
        key = os.path.join(path, file)
        value = meta["view_params"][key]

        depth_path = key.replace("rgb", "depth").replace("png", "exr")

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)

        pose = np.array(value["extrinsic"]).reshape(4, 4)
        poses.append(pose)

        K = np.array(value["intrinsic"])

        depth = np.array(depth_raw)[..., 0]
        pcd = uvd2xyz(depth, K, pose, 1000)  # H, W, 3
        pcd = pcd.reshape(-1, 3)  # N, 3
        points.append(pcd)

    if len(points) == 0:
        return None
    points = np.concatenate(points, axis=0)  # N, 3
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()

    path = f"output/{args.uid}"
    point = gen_points(path)
    poincloud = point.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(poincloud)
    save_path = "pointcloud.ply"
    o3d.io.write_point_cloud(save_path, pcd)


if __name__ == "__main__":
    main()
