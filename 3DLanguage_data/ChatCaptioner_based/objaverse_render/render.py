import os
import bpy
import json
import math
import objaverse
from pyblend.object import load_obj
from pyblend.lighting import config_world, create_light
from pyblend.camera import get_camera_para
from pyblend.utils import BlenderRemover, ArgumentParserForBlender
from pyblend.transform import look_at, normalize_obj, random_loc
from pyblend.render import (
    config_render,
    render_image,
    enable_depth_render,
    config_cycle_gpu,
)


def load_objaverse(uids, download_processes=1):
    objects = objaverse.load_objects(
        uids=uids,
        download_processes=download_processes,
    )
    return objects


def main(args):
    uids = [args.uid]
    objects = load_objaverse(uids)
    uid, path = list(objects.items())[0]

    # ======== Config ========
    config_render(res_x=320, res_y=240, transparent=True, enable_gpu=False)
    remover = BlenderRemover()
    remover.clear_all()
    config_world(1)
    camera = bpy.data.objects["Camera"]
    exr_depth_node, png_depth_node = enable_depth_render(f"output/{uid}", reverse=False)

    # ======== Set up scene ========
    # load object
    obj = load_obj(path, "object", center=False, join=True)
    obj.location = (0, 0, 0)
    normalize_obj(obj)

    # ======== Render ========
    camera_dict = {
        "base_dir": os.path.abspath(f"./"),
        "view_params": {},
    }
    # outside
    theta_view = [
        [-1 / 4, -1 / 4],
        [1 / 4, 1 / 4],
        [3 / 4, 3 / 4],
        [-3 / 4, -3 / 4],
    ]
    phi_view = [
        [1 / 12, 1 / 12],
        [1 / 12, 1 / 12],
        [1 / 12, 1 / 12],
        [1 / 12, 1 / 12],
    ]

    for camera_idx in range(4):
        camera.location = random_loc((0, 0, 0), (3, 3), theta=theta_view[camera_idx], phi=phi_view[camera_idx])
        look_at(camera, (0, 0, 0))
        bpy.context.view_layer.update()
        camera_para = get_camera_para(camera)
        intr = camera_para["intrinsic"]  # (3, 3)
        extr = camera_para["extrinsic"]  # (4, 4)
        bpy.context.scene.frame_current = camera_idx
        png_depth_node.file_slots[0].path = f"outside_depth_norm_"
        exr_depth_node.file_slots[0].path = f"outside_depth_"
        png_path = f"output/{uid}/outside_view_{camera_idx:03d}.png"
        render_image(png_path)
        camera_dict["view_params"][png_path] = {
            "intrinsic": intr.tolist(),
            "extrinsic": extr.tolist(),
        }
    with open(f"output/{uid}/meta.json", "w") as f:
        json.dump(camera_dict, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    args = parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    main(args)
