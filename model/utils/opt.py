import argparse


def get_default_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL CONFIG
    group = parser.add_argument_group("general")
    group.add_argument('--device', required=False, default="cuda")
    group.add_argument('--exp_name', type=str, default='full_trajectory')
    group.add_argument('--trajectory_file', type=str, default='model/trajectories/examples/living_room.json')
    group.add_argument('--models_path', required=False, default="checkpoints")
    group.add_argument('--out_path', required=False, default="output")
    group.add_argument('--input_image_path', required=False, type=str, default=None)
    group.add_argument('--n_images', required=False, default=10, type=int)
    group.add_argument('--save_scene_every_nth', required=False, default=50, type=int)

    # DEPTH CONFIG
    group = parser.add_argument_group("depth")
    group.add_argument('--iron_depth_type', required=False, default="scannet", type=str, help="model type of iron_depth")
    group.add_argument('--iron_depth_iters', required=False, default=20, type=int, help="amount of refinement iterations per prediction")

    # PROJECTION CONFIG
    group = parser.add_argument_group("projection")
    group.add_argument('--fov', required=False, default=55.0, type=float, help="FOV in degrees for all images")
    group.add_argument('--blur_radius', required=False, default=0, type=float, help="if render_mesh: blur radius from pytorch3d rasterization")

    # STABLE DIFFUSION CONFIG
    group = parser.add_argument_group("stable-diffusion")
    group.add_argument('--prompt', required=False, default="a living room with a lit furnace, couch and cozy curtains, bright lamps that make the room look well-lit")
    group.add_argument('--negative_prompt', required=False, default="blurry, bad art, blurred, text, watermark, plant, nature")
    group.add_argument('--guidance_scale', required=False, default=7.5, type=float)
    group.add_argument('--num_inference_steps', required=False, default=50, type=int)

    # INPAINTING MASK CONFIG
    group = parser.add_argument_group("inpaint")
    group.add_argument('--erode_iters', required=False, default=1, type=int, help="how often to erode the inpainting mask")
    group.add_argument('--dilate_iters', required=False, default=2, type=int, help="how often to dilate the inpainting mask")
    group.add_argument('--boundary_thresh', required=False, default=10, type=int, help="how many pixels from image boundaries may not be dilated")
    group.add_argument('--update_mask_after_improvement', required=False, default=False, action="store_true", help="after erosion/dilation/fill_contours -- directly update inpainting mask")

    # MESH UPDATE CONFIG
    group = parser.add_argument_group("fusion")
    group.add_argument('--edge_threshold', required=False, default=0.1, type=float, help="only save faces whose edges _all_ have a smaller l2 distance than this. Default: -1 (=skip)")
    group.add_argument('--surface_normal_threshold', required=False, default=0.1, type=float, help="only save faces whose normals _all_ have a bigger dot product to view direction than this. Default: -1 (:= do not apply threshold)")
    group.add_argument('--faces_per_pixel', required=False, default=8, type=int, help="how many faces per pixel to render (and accumulate)")
    group.add_argument('--replace_over_inpainted', required=False, default=False, action="store_true", help="remove existing faces at the inpainting mask. Note: if <update_mask_after_improvement> is not set, will update mask before performing this operation.")
    group.add_argument('--remove_faces_depth_threshold', required=False, default=0.1, type=float, help="during <replace_over_inpainted>: only remove faces that are within the threshold to rendered depth")
    group.add_argument('--clean_mesh_every_nth', required=False, default=20, type=int, help="run several cleaning steps on the mesh to remove noise, small connected components, etc.")
    group.add_argument('--min_triangles_connected', required=False, default=15000, type=int, help="during <clean_mesh_every_nth>: minimum number of connected triangles in a component to not be removed")
    group.add_argument('--poisson_depth', required=False, default=12, type=int, help="depth value to use for poisson surface reconstruction")
    group.add_argument('--max_faces_for_poisson', required=False, default=10_000_000, type=int, help="after poisson surface reconstruction: save another version of the mesh that has at most this many triangles")

    # completion parameters
    group = parser.add_argument_group("completion")
    group.add_argument('--n_voxels', type=int, default=1000, help="how many voxels we use for this scene")
    group.add_argument('--n_dir', type=int, default=8, help="how many rotation directions we use for a camera inside one voxel")
    group.add_argument('--core_ratio_x', type=float, default=0.4, help="The ratio of regions we sample cameras compared to the bbox along x direction")
    group.add_argument("--core_ratio_y", type=float, default=0.4, help="The ratio of regions we sample cameras compared to the bbox along y direction")
    group.add_argument("--core_ratio_z", type=float, default=0.4, help="The ratio of regions we sample cameras compared to the bbox along y direction")
    group.add_argument('--minimum_completion_pixels', type=int, default=10000, help="we only inpaint images once there are over this many pixels to be inpainted")
    group.add_argument("--min_camera_distance_to_mesh", type=float, default=0.1, help="a sampled camera's position must be at least this far away from the mesh")
    group.add_argument("--min_depth_quantil_to_mesh", type=float, default=2.5, help="a sampled camera's observed image must have a 10% depth quantil of at least this depth")
    group.add_argument("--max_inpaint_ratio", type=float, default=0.2, help="a sampled camera's observed image must have at most this many unobserved pixels (fraction in percent)")
    group.add_argument("--completion_dilate_iters", type=int, default=8, help="repeat mask dilation this many times during completion")

    return parser
