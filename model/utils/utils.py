import torch
import cv2
import os
import json
import numpy as np
import time
import pymeshlab
import imageio

from PIL import Image

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_TURBO):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        if (x > 0).any():
            mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
            ma = np.max(x)
        else:
            mi = 0.0
            ma = 0.0
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def load_sd_inpaint(args):
    model_path = os.path.join(args.models_path, "stable-diffusion-2-inpainting")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Next Image"
    })

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe


def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img


def generate_first_image(args):
    model_path = os.path.join(args.models_path, "stable-diffusion-2-1")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Start Image"
    })

    return pipe(args.prompt).images[0]


def save_image(image, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    image.save(file_out)
    return file_with_ext


def save_rgbd(image, depth, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    dst = Image.new('RGB', (image.width + depth.width, image.height))
    dst.paste(image, (0, 0))
    dst.paste(depth, (image.width, 0))
    dst.save(file_out)
    return file_with_ext


def save_settings(args):
    with open(os.path.join(args.out_path, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def save_animation(image_folder_path, prefix=""):
    gif_name = os.path.join(image_folder_path, prefix + 'animation.gif')
    images = [os.path.join(image_folder_path, img) for img in sorted(os.listdir(image_folder_path), key=lambda x: int(x.split(".")[0].split("_")[-1])) if "rgb" in img]

    with imageio.get_writer(gif_name, mode='I', duration=0.2) as writer:
        for filename in images:
            image = imageio.v3.imread(filename)
            writer.append_data(image)


def save_poisson_mesh(mesh_file_path, depth=12, max_faces=10_000_000):
    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file_path)
    print("loaded", mesh_file_path)

    # compute normals
    start = time.time()
    ms.compute_normal_for_point_clouds()
    print("computed normals")

    # run poisson
    ms.generate_surface_reconstruction_screened_poisson(depth=depth)
    end_poisson = time.time()
    print(f"finish poisson in {end_poisson - start} seconds")

    # save output
    parts = mesh_file_path.split(".")
    out_file_path = ".".join(parts[:-1])
    suffix = parts[-1]
    out_file_path_poisson = f"{out_file_path}_poisson_meshlab_depth_{depth}.{suffix}"
    ms.save_current_mesh(out_file_path_poisson)
    print("saved poisson mesh", out_file_path_poisson)

    # quadric edge collapse to max faces
    start_quadric = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
    end_quadric = time.time()
    print(f"finish quadric decimation in {end_quadric - start_quadric} seconds")

    # save output
    out_file_path_quadric = f"{out_file_path}_poisson_meshlab_depth_{depth}_quadric_{max_faces}.{suffix}"
    ms.save_current_mesh(out_file_path_quadric)
    print("saved quadric decimated mesh", out_file_path_quadric)

    return out_file_path_poisson
