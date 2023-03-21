import torch
import math
import os
import cv2
from PIL import Image
import numpy as np
import json
from datetime import datetime
from tqdm.auto import tqdm
import trimesh

from model.mesh_fusion.util import (
    get_pinhole_intrinsics_from_fov,
    torch_to_trimesh
)

from model.mesh_fusion.render import (
    features_to_world_space_mesh,
    render_mesh,
    save_mesh,
    load_mesh,
    clean_mesh,
    edge_threshold_filter
)

from model.iron_depth.predict_depth import load_iron_depth_model, predict_iron_depth

from model.depth_alignment import depth_alignment

from model.trajectories import trajectory_util, pose_noise_util
from model.trajectories.convert_to_nerf_convention import convert_pose_to_nerf_convention, convert_pose_from_nerf_convention

from model.utils.utils import (
    visualize_depth_numpy,
    save_image,
    pil_to_torch,
    save_rgbd,
    load_sd_inpaint,
    save_settings,
    save_animation
)


class Text2RoomPipeline(torch.nn.Module):
    def __init__(self, args, setup_models=True, offset=0, first_image_pil=None, H=512, W=512, start_pose=None):
        super().__init__()
        # setup (create out_dir, save args)
        self.args = args
        self.orig_n_images = self.args.n_images
        self.orig_prompt = self.args.prompt
        self.orig_negative_prompt = self.args.negative_prompt
        self.orig_surface_normal_threshold = self.args.surface_normal_threshold
        self.H = H
        self.W = W
        self.bbox = [torch.ones(3) * -1.0, torch.ones(3) * 1.0]  # initilize bounding box of meshs as [-1.0, -1.0, -1.0] -> [1.0, 1.0, 1.0]
        self.setup_output_directories()

        assert H == 512 and W == 512, "stable_diffusion inpainting model can process only 512x512 images"

        # load models if required
        if setup_models:
            self.setup_models()

        # initialize global point-cloud / mesh structures
        self.rendered_depth = torch.zeros((H, W), device=self.args.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((H, W), device=self.args.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        self.vertices = torch.empty((3, 0), device=args.device)
        self.colors = torch.empty((3, 0), device=args.device)
        self.faces = torch.empty((3, 0), device=args.device, dtype=torch.long)
        self.pix_to_face = None

        # initialize trajectory
        self.trajectory_fn = trajectory_util.forward()
        self.trajectory_dict = {}
        self.world_to_cam = torch.eye(4, dtype=torch.float32, device=self.args.device) if start_pose is None else start_pose.to(self.args.device)
        self.K = get_pinhole_intrinsics_from_fov(H=self.H, W=self.W, fov_in_degrees=self.args.fov).to(self.world_to_cam)

        # initialize all visited camera poses
        self.seen_poses = []

        # initialize nerf output
        self.nerf_transforms_dict = self.build_nerf_transforms_header()

        # save start image if specified
        if first_image_pil is not None:
            self.setup_start_image(first_image_pil, offset)
        else:
            print("WARN: no start image specified, should call load_mesh() before rendering images!")

    def setup_start_image(self, first_image_pil, offset):
        # save & convert first_image
        self.current_image_pil = first_image_pil
        self.current_image_pil = self.current_image_pil.resize((self.W, self.H))
        self.current_image = pil_to_torch(self.current_image_pil, self.args.device)
        save_image(self.current_image_pil, "rgb", offset, self.args.rgb_path)

        # predict depth, add 3D structure
        self.add_next_image(pos=0, offset=offset)

        # add to seen poses
        self.seen_poses.append(self.world_to_cam)

    def setup_models(self):
        # construct inpainting stable diffusion pipeline
        self.inpaint_pipe = load_sd_inpaint(self.args)

        # construct depth model
        self.iron_depth_n_net, self.iron_depth_model = load_iron_depth_model(self.args.iron_depth_type, self.args.iron_depth_iters, self.args.models_path, self.args.device)

    def remove_models(self):
        self.inpaint_pipe = None
        self.iron_depth_model = None
        self.iron_depth_n_net = None
        torch.cuda.empty_cache()

    def setup_output_directories(self):
        prompt_folder_name = self.orig_prompt[:40]
        prompt_folder_name = prompt_folder_name.replace(" ", "_")

        if os.path.isfile(self.args.trajectory_file):
            trajectory_name = os.path.basename(self.args.trajectory_file)
            trajectory_name = trajectory_name.split(".")[0]
            self.args.out_path = os.path.join(self.args.out_path, trajectory_name)

        self.args.out_path = os.path.join(self.args.out_path, self.args.exp_name, prompt_folder_name)

        if self.args.input_image_path:
            file_name = self.args.input_image_path.split("/")[-1]
            self.args.out_path = os.path.join(self.args.out_path, file_name)
        else:
            self.args.out_path = os.path.join(self.args.out_path, "no_input_image_file")

        now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%fZ')
        self.args.out_path = os.path.join(self.args.out_path, now_str)
        os.makedirs(self.args.out_path, exist_ok=True)
        self.args.rgb_path = os.path.join(self.args.out_path, "rgb")
        self.args.rgbd_path = os.path.join(self.args.out_path, "rgbd")
        self.args.rendered_path = os.path.join(self.args.out_path, "rendered")
        self.args.depth_path = os.path.join(self.args.out_path, "depth")
        self.args.fused_mesh_path = os.path.join(self.args.out_path, "fused_mesh")
        self.args.mask_path = os.path.join(self.args.out_path, "mask")
        self.args.output_rendering_path = os.path.join(self.args.out_path, "output_rendering")
        self.args.output_depth_path = os.path.join(self.args.out_path, "output_depth")
        os.makedirs(self.args.rgb_path, exist_ok=True)
        os.makedirs(self.args.rgbd_path, exist_ok=True)
        os.makedirs(self.args.rendered_path, exist_ok=True)
        os.makedirs(self.args.depth_path, exist_ok=True)
        os.makedirs(self.args.fused_mesh_path, exist_ok=True)
        os.makedirs(self.args.mask_path, exist_ok=True)
        os.makedirs(self.args.output_rendering_path, exist_ok=True)
        os.makedirs(self.args.output_depth_path, exist_ok=True)
        save_settings(self.args)

    def build_nerf_transforms_header(self):
        return {
            "fl_x": self.K[0, 0].cpu().numpy().item(),
            "fl_y": self.K[1, 1].cpu().numpy().item(),
            "cx": self.K[0, 2].cpu().numpy().item(),
            "cy": self.K[1, 2].cpu().numpy().item(),
            "w": self.W,
            "h": self.H,
            "camera_angle_x": self.args.fov * math.pi / 180.0,
            "aabb_scale": 4,
            "integer_depth_scale": 10000,
            "frames": []
        }

    def save_poses(self, pose_file_path, poses=None):
        if poses is None:
            poses = self.seen_poses
        pose_dict = {i: p.cpu().numpy().tolist() for i, p in enumerate(poses)}
        with open(pose_file_path, "w") as f:
            json.dump(pose_dict, f, indent=4)

    def load_poses(self, pose_file_path, convert_from_nerf=False, replace_existing=True):
        with open(pose_file_path, "r") as f:
            poses = json.load(f)
            if 'frames' in poses:
                poses = poses['frames']
                poses = [torch.from_numpy(np.array(p['transform_matrix'])).to(self.args.device).float() for p in poses]
            else:
                poses = [torch.from_numpy(np.array(p)).to(self.args.device).float() for i, p in poses.items()]

            if convert_from_nerf:
                poses = [convert_pose_from_nerf_convention(p) for p in poses]

            if replace_existing or not hasattr(self, "seen_poses") or self.seen_poses is None:
                self.seen_poses = poses
            else:
                self.seen_poses.extend(poses)

    def append_nerf_extrinsic(self, rgb_dir_name, rgb_file_name, depth_dir_name, depth_file_name):
        p = convert_pose_to_nerf_convention(self.world_to_cam)

        self.nerf_transforms_dict["frames"].append({
            "transform_matrix": p.cpu().numpy().tolist(),
            "file_path": f"{rgb_dir_name}/{rgb_file_name}",
            "depth_file_path": f"{depth_dir_name}/{depth_file_name}"
        })

    def save_nerf_transforms(self):
        nerf_transforms_file = os.path.join(self.args.out_path, 'transforms.json')

        with open(nerf_transforms_file, "w") as f:
            json.dump(self.nerf_transforms_dict, f, indent=4)

        return nerf_transforms_file

    def save_mesh(self, name='fused_final.ply'):
        target_path = os.path.join(self.args.fused_mesh_path, name)

        save_mesh(
            vertices=self.vertices,
            faces=self.faces,
            colors=self.colors,
            target_path=target_path
        )

        return target_path

    def load_mesh(self, rgb_path):
        vertices, faces, rgb = load_mesh(rgb_path)
        self.vertices = vertices.to(self.vertices)
        self.colors = rgb.to(self.colors)
        self.faces = faces.to(self.faces)

    def save_animations(self, prefix=""):
        save_animation(self.args.rgb_path, prefix=prefix)
        save_animation(self.args.rgbd_path, prefix=prefix)

    def predict_depth(self):
        # use the IronDepth method to predict depth: https://github.com/baegwangbin/IronDepth
        predicted_depth, _ = predict_iron_depth(
            image=self.current_image_pil,
            K=self.K,
            device=self.args.device,
            model=self.iron_depth_model,
            n_net=self.iron_depth_n_net,
            input_depth=self.rendered_depth,
            input_mask=self.inpaint_mask,
            fix_input_depth=True
        )

        return predicted_depth

    def depth_alignment(self, predicted_depth):
        aligned_depth = depth_alignment.scale_shift_linear(
            rendered_depth=self.rendered_depth,
            predicted_depth=predicted_depth,
            mask=~self.inpaint_mask,
            fuse=True)

        return aligned_depth

    def add_vertices_and_faces(self, predicted_depth):
        if self.inpaint_mask.sum() == 0:
            # when no pixels were masked out, we do not need to add anything, so skip this call
            return

        vertices, faces, colors = features_to_world_space_mesh(
            colors=self.current_image,
            depth=predicted_depth,
            fov_in_degrees=self.args.fov,
            world_to_cam=self.world_to_cam,
            mask=self.inpaint_mask,
            edge_threshold=self.args.edge_threshold,
            surface_normal_threshold=self.args.surface_normal_threshold,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices
        )

        faces += self.vertices.shape[1]  # add face offset

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def remove_masked_out_faces(self):
        if self.pix_to_face is None:
            return

        # get faces to remove: those faces that project into the inpaint_mask
        faces_to_remove = self.pix_to_face[:, self.inpaint_mask, :]

        # only remove faces whose depth is close to actual depth
        if self.args.remove_faces_depth_threshold > 0:
            depth = self.rendered_depth[self.inpaint_mask]
            depth = depth[None, ..., None]
            depth = depth.repeat(faces_to_remove.shape[0], 1, faces_to_remove.shape[-1])
            zbuf = self.z_buf[:, self.inpaint_mask, :]
            mask_zbuf = (zbuf - depth).abs() < self.args.remove_faces_depth_threshold
            faces_to_remove = faces_to_remove[mask_zbuf]

        faces_to_remove = torch.unique(faces_to_remove.flatten())
        faces_to_remove = faces_to_remove[faces_to_remove > -1].long()

        # select the faces that were hit in the mask
        # this does not catch all faces because some faces that project into the mask are not visible from current viewpoint (e.g. behind another face)
        # this _should not_ catch those faces though - they might not be wanted to be removed.
        keep_faces_mask = torch.ones_like(self.faces[0], dtype=torch.bool)
        keep_faces_mask[faces_to_remove] = False

        # remove the faces
        self.faces = self.faces[:, keep_faces_mask]

        # remove left-over too long faces
        self.apply_edge_threshold_filter()

        # set to None since pix_to_face has now changed
        # this is actually desired behavior: we do not fuse together new faces with current mesh, because it is too difficult anyways
        self.pix_to_face = None

    def set_trajectory(self, trajectory_dict):
        self.trajectory_dict = trajectory_dict
        fn = getattr(trajectory_util, trajectory_dict["fn_name"])
        self.trajectory_fn = fn(**trajectory_dict["fn_args"])
        self.args.n_images = trajectory_dict.get("n_images", self.orig_n_images)
        self.args.prompt = trajectory_dict.get("prompt", self.orig_prompt)
        self.args.negative_prompt = trajectory_dict.get("negative_prompt", self.orig_negative_prompt)
        self.args.surface_normal_threshold = trajectory_dict.get("surface_normal_threshold", self.orig_surface_normal_threshold)

    def get_next_pose_in_trajectory(self, i=0):
        return self.trajectory_fn(i, self.args.n_images).to(self.args.device)

    def project(self):
        # project mesh into pose and render (rgb, depth, mask)
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.args.fov,
            RT=self.world_to_cam,
            blur_radius=self.args.blur_radius,
            faces_per_pixel=self.args.faces_per_pixel
        )

        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        return rendered_image_tensor, rendered_image_pil, inpaint_mask_pil

    def inpaint(self, rendered_image_pil, inpaint_mask_pil):
        m = np.asarray(inpaint_mask_pil)[..., 0].astype(np.uint8)

        # inpaint with classical method to fill small gaps
        rendered_image_numpy = np.asarray(rendered_image_pil)
        rendered_image_pil = Image.fromarray(cv2.inpaint(rendered_image_numpy, m, 3, cv2.INPAINT_TELEA))

        # remove small seams from mask
        kernel = np.ones((7, 7), np.uint8)
        m2 = m
        if self.args.erode_iters > 0:
            m2 = cv2.erode(m, kernel, iterations=self.args.erode_iters)
        if self.args.dilate_iters > 0:
            m2 = cv2.dilate(m2, kernel, iterations=self.args.dilate_iters)

        # do not allow mask to extend to boundaries
        if self.args.boundary_thresh > 0:
            t = self.args.boundary_thresh
            h, w = m2.shape
            m2[:t] = m[:t]  # top
            m2[h - t:] = m[h - t:]  # bottom
            m2[:, :t] = m[:, :t]  # left
            m2[:, w - t:] = m[:, w - t:]  # right

        # close inner holes in dilated masks -- find out-most contours and fill everything inside
        contours, hierarchy = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(m2, contours, -1, 255, thickness=cv2.FILLED)

        # convert back to pil & save updated mask
        inpaint_mask_pil = Image.fromarray(m2).convert("RGB")
        self.eroded_dilated_inpaint_mask = torch.from_numpy(m2).to(self.inpaint_mask)

        # update inpaint mask to contain all updates
        if self.args.update_mask_after_improvement:
            self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

        # inpaint large missing areas with stable-diffusion model
        inpainted_image_pil = self.inpaint_pipe(
            prompt=self.args.prompt,
            negative_prompt=self.args.negative_prompt,
            num_images_per_prompt=1,
            image=rendered_image_pil,
            mask_image=inpaint_mask_pil,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps
        ).images[0]

        return inpainted_image_pil, inpaint_mask_pil

    def apply_depth_smoothing(self, image, mask):

        def dilate(x, k=3):
            x = torch.nn.functional.conv2d(
                x.float()[None, None, ...],
                torch.ones(1, 1, k, k).to(x.device),
                padding="same"
            )
            return x.squeeze() > 0

        def sobel(x):
            flipped_sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).to(x.device)
            flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(1)

            x_pad = torch.nn.functional.pad(x.float()[None, None, ...], (1, 1, 1, 1), mode="replicate")

            x = torch.nn.functional.conv2d(
                x_pad,
                flipped_sobel_x,
                padding="valid"
            )
            dx, dy = x.unbind(dim=-3)
            # return torch.sqrt(dx**2 + dy**2).squeeze()
            # new content is created mostly in x direction, sharp edges in y direction are wanted (e.g. table --> wall)
            return dx.squeeze()

        edges = sobel(mask)
        dilated_edges = dilate(edges, k=21)

        img_numpy = image.float().cpu().numpy()
        blur_bilateral = cv2.bilateralFilter(img_numpy, 5, 140, 140)
        blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
        blur_gaussian = torch.from_numpy(blur_gaussian).to(image)

        image_smooth = torch.where(dilated_edges, blur_gaussian, image)
        return image_smooth

    def add_next_image(self, pos, offset, save_files=True, file_suffix=""):
        # predict & align depth of current image
        predicted_depth = self.predict_depth()
        predicted_depth = self.depth_alignment(predicted_depth)
        predicted_depth = self.apply_depth_smoothing(predicted_depth, self.inpaint_mask)
        self.predicted_depth = predicted_depth

        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(self.rendered_depth.cpu().numpy())[0].astype(np.uint8))
        depth_pil = Image.fromarray(visualize_depth_numpy(predicted_depth.cpu().numpy())[0].astype(np.uint8))
        if save_files:
            save_image(rendered_depth_pil, f"rendered_depth{file_suffix}", offset + pos, self.args.depth_path)
            save_image(depth_pil, f"depth{file_suffix}", offset + pos, self.args.depth_path)
            save_rgbd(self.current_image_pil, depth_pil, f'rgbd{file_suffix}', offset + pos, self.args.rgbd_path)

        # remove masked-out faces. If we use erosion in the mask it means those points will be removed.
        if self.args.replace_over_inpainted:
            # only now update mask: predicted depth will still take old positions as anchors (they are still somewhat correct)
            # otherwise if we erode/dilate too much we could get depth estimates that are way off
            if not self.args.update_mask_after_improvement:
                self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

            self.remove_masked_out_faces()

        # add new points (novel content)
        self.add_vertices_and_faces(predicted_depth)

        # save current meshes
        if save_files and self.args.save_scene_every_nth > 0 and (offset + pos) % self.args.save_scene_every_nth == 0:
            self.save_mesh(f"fused_until_frame{file_suffix}_{offset + pos:04}.ply")

    def project_and_inpaint(self, pos=0, offset=0, save_files=True, file_suffix="", inpainted_image_pil=None):
        # project to next pose
        _, rendered_image_pil, inpaint_mask_pil = self.project()
        if "adaptive" in self.trajectory_dict:
            def update_pose(reverse=False):
                # update the args in trajectory dict
                for d in self.trajectory_dict["adaptive"]:
                    arg = d["arg"]
                    delta = d["delta"] if not reverse else -d["delta"]
                    self.trajectory_dict["fn_args"][arg] += delta

                    if "min" in d:
                        self.trajectory_dict["fn_args"][arg] = max(d["min"], self.trajectory_dict["fn_args"][arg])
                    if "max" in d:
                        self.trajectory_dict["fn_args"][arg] = min(d["max"], self.trajectory_dict["fn_args"][arg])

                # update pose
                self.set_trajectory(self.trajectory_dict)
                self.world_to_cam = self.get_next_pose_in_trajectory(pos)
                self.seen_poses[-1] = self.world_to_cam

                # render new images
                return self.project()

            for i in range(10):
                # increase args as long as depth does not get smaller again
                # can extend this to allow multiple comparisons: e.g., add "as long as mean depth not smaller than X"
                old_std_depth, old_mean_depth = torch.std_mean(self.rendered_depth)
                _, rendered_image_pil, inpaint_mask_pil = update_pose()
                current_std_depth, current_mean_depth = torch.std_mean(self.rendered_depth)

                if current_mean_depth <= old_mean_depth:
                    # go back one step and end search
                    _, rendered_image_pil, inpaint_mask_pil = update_pose(reverse=True)
                    break

        # inpaint projection result
        if inpainted_image_pil is None:
            inpainted_image_pil, eroded_dilated_inpaint_mask_pil = self.inpaint(rendered_image_pil, inpaint_mask_pil)
            if save_files:
                save_image(eroded_dilated_inpaint_mask_pil, f"mask_eroded_dilated{file_suffix}", offset + pos, self.args.mask_path)
        else:
            self.eroded_dilated_inpaint_mask = torch.zeros_like(self.inpaint_mask)

        # update images
        self.current_image_pil = inpainted_image_pil
        self.current_image = pil_to_torch(inpainted_image_pil, self.args.device)

        if save_files:
            save_image(rendered_image_pil, f"rendered{file_suffix}", offset + pos, self.args.rendered_path)
            save_image(inpaint_mask_pil, f"mask{file_suffix}", offset + pos, self.args.mask_path)
            save_image(self.current_image_pil, f"rgb{file_suffix}", offset + pos, self.args.rgb_path)

        # predict depth, add to 3D structure
        self.add_next_image(pos, offset, save_files, file_suffix)

        # update bounding box
        self.calc_bounding_box()

    def clean_mesh(self):
        self.vertices, self.faces, self.colors = clean_mesh(
            vertices=self.vertices,
            faces=self.faces,
            colors=self.colors,
            edge_threshold=self.args.edge_threshold,
            min_triangles_connected=self.args.min_triangles_connected,
            fill_holes=True
        )

    def apply_edge_threshold_filter(self):
        self.faces = edge_threshold_filter(
            vertices=self.vertices,
            faces=self.faces,
            edge_threshold=self.args.edge_threshold
        )

    def forward(self, pos=0, offset=0, save_files=True):
        # get next pose
        self.world_to_cam = self.get_next_pose_in_trajectory(pos)
        self.seen_poses.append(self.world_to_cam.clone())

        # render --> inpaint --> add to 3D structure
        self.project_and_inpaint(pos, offset, save_files)

        if self.args.clean_mesh_every_nth > 0 and (pos + offset) % self.args.clean_mesh_every_nth == 0:
            self.clean_mesh()

    def refine(self, pos=0, offset=0, repeat_iters=1):
        # save old values
        old_replace_over_inpainted = self.args.replace_over_inpainted
        old_min_triangles_connected = self.args.min_triangles_connected
        old_surface_normal_threshold = self.args.surface_normal_threshold
        old_erode_iters = self.args.erode_iters
        old_dilate_iters = self.args.dilate_iters
        old_prompt = self.args.prompt
        old_negative_prompt = self.args.negative_prompt

        # project_and_inpaint -- but with replace_over_inpainted option and huge dilate_iters and no erode_iters
        self.args.replace_over_inpainted = True
        self.args.erode_iters = 0
        self.args.dilate_iters = self.args.completion_dilate_iters
        self.args.min_triangles_connected = -1
        self.args.surface_normal_threshold = -1
        self.args.prompt = self.orig_prompt
        self.args.negative_prompt = "humans, animals, text, distortions, blurry"
        self.project_and_inpaint(pos, offset)

        # reset to old values
        self.args.replace_over_inpainted = old_replace_over_inpainted
        self.args.erode_iters = old_erode_iters
        self.args.dilate_iters = old_dilate_iters

        # repeat to fill in remaining holes
        pbar_repeat = tqdm(range(repeat_iters), leave=True, desc=f"Repeat Refine Image {pos}")
        for i in pbar_repeat:
            self.project_and_inpaint(pos, offset, save_files=(i + 1) == repeat_iters, file_suffix=f"_repeat_{i}", inpainted_image_pil=self.current_image_pil)

        if self.args.clean_mesh_every_nth > 0 and (pos + offset) % self.args.clean_mesh_every_nth == 0:
            self.clean_mesh()

        # reset to old values
        self.args.min_triangles_connected = old_min_triangles_connected
        self.args.surface_normal_threshold = old_surface_normal_threshold
        self.args.prompt = old_prompt
        self.args.negative_prompt = old_negative_prompt

    def generate_images(self, offset=0):
        # generate images with forward-warping
        pbar = tqdm(range(self.args.n_images))
        for pos in pbar:
            pbar.set_description(f"Image [{pos}/{self.args.n_images - 1}]")
            self.forward(pos, offset)

        # reset gpu memory
        torch.cuda.empty_cache()

        return offset + self.args.n_images

    def save_seen_trajectory_renderings(self, apply_noise=False, chunk_size=10, r_max=5.0, t_max=0.05, add_to_nerf_images=False):
        old_world_to_cam = self.world_to_cam.clone()

        poses = self.seen_poses if not apply_noise else pose_noise_util.apply_noise(self.seen_poses, chunk_size, r_max, t_max)
        self.save_poses(os.path.join(self.args.out_path, "seen_poses.json" if not apply_noise else "seen_poses_noise.json"), poses)
        pbar = tqdm(poses, desc=f"Save Renderings [Noise={apply_noise}]")
        for i, p in enumerate(pbar):
            self.world_to_cam = p
            _, rendered_image_pil, inpaint_mask_pil = self.project()
            filename = save_image(rendered_image_pil, "rendering" if not apply_noise else "rendering_noise", i, self.args.output_rendering_path)
            filename_depth = save_image(
                Image.fromarray(self.rendered_depth.squeeze().detach().cpu().numpy().astype(np.uint16)),
                "depth" if not apply_noise else "depth_noise", i, self.args.output_depth_path)
            if add_to_nerf_images:
                self.append_nerf_extrinsic(os.path.basename(self.args.output_rendering_path), filename, os.path.basename(self.args.output_depth_path), filename_depth)

        self.world_to_cam = old_world_to_cam

    def calc_bounding_box(self):
        """
        Calculate the bounding box of existing meshes. 
        We use the most simply version to calculate: [x_min, y_min, z_min] -> [x_max, y_max, z_max]
        """
        min_bound = torch.amin(self.vertices, dim=-1)
        max_bound = torch.amax(self.vertices, dim=-1)
        self.bbox = [min_bound, max_bound]

    def N_to_reso(self, n_voxels, adjusted_grid=True):
        """
        Given the n_voxels and length along x,y,z, we calculate the resolution along each dimension.
        """
        if adjusted_grid:
            xyz_min, xyz_max = self.bbox
            voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
            return ((xyz_max - xyz_min) / voxel_size).long().tolist()
        else:
            # grid_each = n_voxels.pow(1 / 3)
            grid_each = math.pow(n_voxels, 1 / 3)
            return [int(grid_each), int(grid_each), int(grid_each)]

    def random_direction_sample_no_phi(self):
        theta = torch.rand(self.args.n_dir) * 360
        phi = torch.rand(self.args.n_dir) * 0
        c2w = [trajectory_util.pose_spherical(theta[i], phi[i], 1) for i in range(self.args.n_dir)]
        c2w = torch.stack(c2w, dim=0)  # [n_dir, 4, 4]
        return c2w

    def _completion_heuristic_sample_points(self):
        self.calc_bounding_box()

        voxel_reso = self.N_to_reso(self.args.n_voxels, adjusted_grid=True)  # [N_voxel_x, N_voxel_y, N_voxel_z]
        min_bound, max_bound = self.bbox
        core_ratio = torch.tensor([self.args.core_ratio_x, self.args.core_ratio_y, self.args.core_ratio_z]).to(min_bound.device)

        self.core_bbox = [min_bound * core_ratio, max_bound * core_ratio]
        self.core_bbox_length = (max_bound - min_bound) * core_ratio

        device = self.core_bbox_length[0].device
        X = torch.linspace(0, 1, voxel_reso[0], device=device) * self.core_bbox_length[0] + self.core_bbox[0][0]
        Y = torch.linspace(0, 1, voxel_reso[1], device=device) * self.core_bbox_length[1] + self.core_bbox[0][1]
        Z = torch.linspace(0, 1, voxel_reso[2], device=device) * self.core_bbox_length[2] + self.core_bbox[0][2]

        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
        grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # get voxelized positions grid_xyz, [N_1, N_2, N_3, 3]
        return grid_xyz.view(-1, 3).contiguous()

    def complete_mesh(self, offset=0):
        max_inpaint = 512 * 512 * self.args.max_inpaint_ratio

        # Get sampled points in [N, 3] shape
        grid_xyz = self._completion_heuristic_sample_points()

        # convert to trimesh
        mesh = torch_to_trimesh(self.vertices, self.faces, self.colors)

        # Filter out inappropriate points
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, grid_xyz.detach().cpu().numpy())
        filtering_mask = distance >= self.args.min_camera_distance_to_mesh
        grid_xyz = grid_xyz[torch.tensor(filtering_mask).to(grid_xyz.device)]
        device = grid_xyz.device

        # complete from each sampled point
        pos = 0
        for i in tqdm(range(grid_xyz.shape[0]), desc="Completion"):
            # sample multiple cameras for each point
            camera_pos = grid_xyz[i]
            c2w = self.random_direction_sample_no_phi()
            c2w[:, 0:3, 3] = camera_pos
            RT = torch.inverse(c2w)

            # render from each camera
            inpaint_masks = []
            rendered_images = []
            depth_min_quantile = []
            for index in range(RT.shape[0]):
                self.world_to_cam = RT[index].to(device)
                _, rendered_image_pil, inpaint_mask_pil = self.project()

                inpaint_masks.append(np.array(inpaint_mask_pil)[..., 0:1] / 255.0)
                rendered_images.append(rendered_image_pil)

                depth = self.rendered_depth
                depth = depth[depth != 0]
                if depth.shape[0] == 0:
                    depth_min_quantile.append(torch.zeros(1, device=depth.device, dtype=depth.dtype)[0])
                else:
                    depth_min_quantile.append(torch.quantile(depth, 0.1))

            # filter cameras that contain too much novel content (e.g. looking outside of the scene)
            inpaint_masks = np.stack(inpaint_masks, axis=0)
            inpaint_masks = inpaint_masks.reshape(RT.shape[0], -1).sum(-1)  # [n_dir]
            mask = inpaint_masks < max_inpaint

            # filter cameras that are too close to existing geometry
            depth_min_quantile = torch.stack(depth_min_quantile).cpu().numpy()  # [n_dir]
            depth_mask = depth_min_quantile >= self.args.min_depth_quantil_to_mesh
            mask = mask * depth_mask

            # apply filters
            inpaint_masks = inpaint_masks[mask]
            RT = RT[mask]
            if len(inpaint_masks) == 0:
                continue

            # take the pose that views most unobserved regions
            max_index = np.argmax(inpaint_masks)

            # refine from that pose
            if inpaint_masks[max_index] >= self.args.minimum_completion_pixels:
                self.world_to_cam = RT[max_index].to(device)
                self.seen_poses.append(self.world_to_cam.clone())
                self.refine(pos, offset, repeat_iters=1)
                pos += 1

                # reset gpu memory
                torch.cuda.empty_cache()

        return offset + pos
