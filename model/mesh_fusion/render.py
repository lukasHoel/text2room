import torch
import numpy as np
import open3d as o3d

from model.mesh_fusion.util import (
    unproject_points,
    get_camera,
    o3d_pcd_to_torch,
    o3d_mesh_to_torch,
    torch_to_o3d_pcd,
    torch_to_o3d_mesh,
    torch_to_trimesh,
    trimesh_to_torch,
    o3d_to_trimesh
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex
)
from pytorch3d.renderer.mesh.shader import HardDepthShader, ShaderBase, BlendParams
from pytorch3d.renderer.blending import hard_rgb_blend, softmax_rgb_blend


def load_mesh(file_name):
    mesh = o3d.io.read_triangle_mesh(file_name)
    mesh.remove_unreferenced_vertices()
    return o3d_mesh_to_torch(mesh)


def save_mesh(vertices, faces, colors, target_path):
    colors = colors[:3, ...]
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(target_path, mesh, compressed=True, write_vertex_colors=True, print_progress=True)


def save_world_space_mesh(image, depth, fov_in_degrees, world_to_cam, target_path):
    """
    project image to mesh in world space and save as .ply file

    :param image: (3, H, W) in 0...1 range
    :param depth: (H, W)
    :param fov_in_degrees: fov_in_degrees
    :param world_to_cam: (4, 4)
    :param target_path: out_path
    """
    vertices, faces, colors = features_to_world_space_mesh(image, depth, fov_in_degrees, world_to_cam)
    save_mesh(vertices, faces, colors, target_path)


def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, edge_threshold: float = 0.1, min_triangles_connected: int = -1, fill_holes: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Performs the following steps to clean the mesh:

    1. edge_threshold_filter
    2. remove_duplicated_vertices, remove_duplicated_triangles, remove_degenerate_triangles
    3. remove small connected components
    4. remove_unreferenced_vertices
    5. fill_holes

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param fill_holes: If true, will perform trimesh fill_holes step, otherwise not.

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    if edge_threshold > 0:
        # remove long edges
        faces = edge_threshold_filter(vertices, faces, edge_threshold)

    # cleanup via open3d
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    if min_triangles_connected > 0:
        # remove small components via open3d
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles_connected
        mesh.remove_triangles_by_mask(triangles_to_remove)

    # cleanup via open3d
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        # misc cleanups via trimesh
        mesh = o3d_to_trimesh(mesh)
        mesh.process()
        mesh.fill_holes()

        return trimesh_to_torch(mesh, v=vertices, f=faces, c=colors)
    else:
        return o3d_mesh_to_torch(mesh, v=vertices, f=faces, c=colors)


def edge_threshold_filter(vertices, faces, edge_threshold=0.1):
    """
    Only keep faces where all edges are smaller than edge_threshold.
    Will remove stretch artifacts that are caused by inconsistent depth at object borders

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param edge_threshold: maximum length per edge (otherwise removes that face).

    :return: filtered faces
    """

    p0, p1, p2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
    d01 = torch.linalg.vector_norm(p0 - p1, dim=0)
    d02 = torch.linalg.vector_norm(p0 - p2, dim=0)
    d12 = torch.linalg.vector_norm(p1 - p2, dim=0)

    mask_small_edge = (d01 < edge_threshold) * (d02 < edge_threshold) * (d12 < edge_threshold)
    faces = faces[:, mask_small_edge]

    return faces


def calculate_face_normal(vertices, faces):
    """
    Calculate face normal from vertices and faces.
    Vertices has shape [N, 3], faces has shape [M, 3]
    Output has shape [M, 3].
    """
    face_pos = vertices[[faces]] #shape [M, 3, 3]

    BA = face_pos[:, 1] - face_pos[:, 0]
    BA = BA / torch.norm(BA, dim=-1, keepdim=True) # Prevent too small values
    CA = face_pos[:, 2] - face_pos[:, 0]
    CA = CA / torch.norm(CA, dim=-1, keepdim=True)

    normals = torch.cross(BA, CA, dim=-1)

    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    return normals


def surface_normal_filter(vertices, faces, H, W, world_to_cam, fov_in_degrees, surface_normal_threshold=0.1, pix_to_face=None):
    """
    Only keep faces where the dot product between surface normal and viewing direction is larger than the threshold.
    Will remove stretch artifacts that are caused by bad viewing angles towards surfaces.

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param H: image plane height
    :param W: image plane width
    :param world_to_cam: (4, 4)
    :param fov_in_degrees: fov_in_degrees
    :param surface_normal_threshold: the dot product threshold (applied on normalized vectors, so should be a value in 0..1)
    :param pix_to_face: If specified, will calculate based on view_direction from corresponding pixel, else from image-plane-center
     LongTensor of shape (N, image_size, image_size, faces_per_pixel) giving the indices of the nearest faces at each pixel, sorted in ascending z-order.
     Concretely pix_to_face[n, y, x, k] = f means that faces_verts[f] is the kth closest face (in the z-direction) to pixel (y, x).
     Pixels that are hit by fewer than faces_per_pixel are padded with -1.

    :return: filtered faces
    """

    # calculate face normals
    surface_normal = calculate_face_normal(vertices.permute(1, 0), faces.permute(1, 0))  # [faces_number, 3]

    # Get view directions
    depth = torch.ones((H, W)).to(vertices)
    world_points = unproject_points(world_to_cam, fov_in_degrees, depth, H, W).permute(1, 0)
    camera = get_camera(world_to_cam, fov_in_degrees)
    camera_center = camera.get_camera_center()
    view_direction = world_points - camera_center
    view_direction = view_direction / torch.norm(view_direction, dim=-1, keepdim=True)

    if pix_to_face is None:
        # use center view direction since we do not know which face maps to which pixel
        select_view_direction = view_direction[view_direction.shape[0] // 2].unsqueeze(0)
        dot_map = torch.sum(select_view_direction * surface_normal, dim=-1)
        dot_map = torch.abs(dot_map)
    else:
        # calculate dot product between per-pixel view_direction and the surface normals of all faces that project into this pixel
        pix_to_face = pix_to_face.squeeze()
        invalid_mask = pix_to_face < 0
        pix_to_face[invalid_mask] = 0  # for indexing to work, but gets filtered later
        pix_to_surface_normal = surface_normal[pix_to_face]  # (H, W, faces_per_pixel, 3)
        view_direction = view_direction.reshape(H, W, 3).unsqueeze(-2).repeat(1, 1, pix_to_surface_normal.shape[2], 1)  # (H, W, faces_per_pixel, 3)
        dot_map = (pix_to_surface_normal * view_direction).sum(dim=-1).abs()  # (H, W, faces_per_pixel)

        # a face can be used in multiple pixels, so final dot product is the average from all pixels
        per_face_dot_product_sum = torch.zeros(faces.shape[1], device=faces.device)  # (M)
        per_face_observed_count = torch.zeros(faces.shape[1], device=faces.device)  # (M)

        # add contribution to the specified face positions
        index = pix_to_face[~invalid_mask].flatten()  # M * (H, W, faces_per_pixel) --> (P)
        dot_map = dot_map[~invalid_mask].flatten()  # M * (H, W, faces_per_pixel) --> (P)
        per_face_dot_product_sum.scatter_add_(dim=0, index=index, src=dot_map)
        per_face_observed_count.scatter_add_(dim=0, index=index, src=torch.ones_like(dot_map))

        # compute final average
        dot_map = per_face_dot_product_sum / per_face_observed_count.clamp(min=1e-8)

    faces_remove_mask = dot_map < surface_normal_threshold
    faces = faces[:, ~faces_remove_mask]

    return faces, faces_remove_mask


def remesh_poisson(vertices, colors, faces=None, max_faces_for_poisson=4_000_000, edge_threshold=0.1, min_triangles_connected=100, poisson_depth=10, density_quantil_threshold=0.005, n_threads=-1):
    """
    Remeshes the input mesh or point-cloud using the following steps:

    1. If faces are specified: simplify_quadric_decimation such that mesh has at most <max_faces_for_poisson> faces.
    2. If faces are specified: calculate vertex_normals from mesh. Else: calculate point_normals.
    3. Run poisson surface reconstruction on the point-cloud with normals and colors
    4. Call clean_mesh()

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param faces: (3, M) torch.Tensor of type torch.long. If not specified, will not do simplify_quadric_decimation and compute normals from points only.
    :param max_faces_for_poisson: maximum number of faces before running poisson (such that it does not take forever)
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param depth: depth at which to run poisson (see o3d.geometry.TriangleMesh.create_from_point_cloud_poisson)
    :param density_quantil_threshold: after running poisson, filter the vertices whose density is below this quantil_threshold
    :param n_threads: how many threads to use for poisson (see o3d.geometry.TriangleMesh.create_from_point_cloud_poisson)

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    if faces is not None:
        mesh = torch_to_o3d_mesh(vertices, faces, colors)
        if faces.shape[1] > max_faces_for_poisson:
            # make mesh smaller
            mesh = mesh.simplify_quadric_decimation(max_faces_for_poisson)
        mesh.compute_vertex_normals()  # normal estimation from mesh -- somehow raises an error "invalid data"

        # build point cloud from mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        pcd.normals = mesh.vertex_normals
    else:
        pcd = torch_to_o3d_pcd(vertices, colors)
        pcd.estimate_normals()  # normal estimation from point cloud -- makes normals outside, but we want inside for better shading in meshlab
        pcd.orient_normals_consistent_tangent_plane(k=100)
        #pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

    # poisson surface reconstruction from point cloud
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd=pcd,
        depth=poisson_depth,
        scale=1.1,
        linear_fit=False,
        n_threads=n_threads
    )

    # filter vertices with density quantile
    vertices_to_remove = densities < np.quantile(densities, density_quantil_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # get tensors from o3d mesh object (still on cpu because we call clean_mesh afterwards anyways)
    device, dtype = vertices.device, vertices.dtype
    vertices, faces, colors = o3d_mesh_to_torch(mesh)
    faces = faces.long()

    # clean mesh
    vertices, faces, colors = clean_mesh(
        vertices,
        faces,
        colors,
        edge_threshold=edge_threshold,
        min_triangles_connected=min_triangles_connected,
        fill_holes=False)

    vertices = vertices.to(device=device, dtype=dtype)
    faces = faces.to(device=device, dtype=torch.long)
    colors = colors.to(device=device, dtype=dtype)

    return vertices, faces, colors


def features_to_world_space_mesh(colors, depth, fov_in_degrees, world_to_cam, mask=None, edge_threshold=-1, surface_normal_threshold=-1, pix_to_face=None, faces=None, vertices=None):
    """
    project features to mesh in world space and return (vertices, faces, colors) result by applying simple triangulation from image-structure.

    :param colors: (C, H, W)
    :param depth: (H, W)
    :param fov_in_degrees: fov_in_degrees
    :param world_to_cam: (4, 4)
    :param mask: (H, W)
    :param edge_threshold: only save faces whose edges _all_ have a smaller l2 distance than this. Default: -1 (:= do not apply threshold)
    :param surface_normal_threshold: only save faces whose normals _all_ have a bigger dot product to view direction than this. Default: -1 (:= do not apply threshold)
    :param pix_to_face: LongTensor of shape
          (N, image_size, image_size, faces_per_pixel)
          giving the indices of the nearest faces at each pixel,
          sorted in ascending z-order.
          Concretely ``pix_to_face[n, y, x, k] = f`` means that
          ``faces_verts[f]`` is the kth closest face (in the z-direction)
          to pixel (y, x). Pixels that are hit by fewer than
          faces_per_pixel are padded with -1.
    :param faces: all currently existing faces (referenced by pix_to_face)
    :param vertices: all currently existinc vertices (referenced by faces)
    """

    # get point cloud from depth map
    C, H, W = colors.shape
    colors = colors.reshape(C, -1)
    world_space_points = unproject_points(world_to_cam, fov_in_degrees, depth, H, W)

    # define vertex_ids for triangulation
    '''
    00---01
    |    |
    10---11
    '''
    vertex_ids = torch.arange(H*W).reshape(H, W).to(colors.device)
    vertex_00 = remapped_vertex_00 = vertex_ids[:H-1, :W-1]
    vertex_01 = remapped_vertex_01 = (remapped_vertex_00 + 1)
    vertex_10 = remapped_vertex_10 = (remapped_vertex_00 + W)
    vertex_11 = remapped_vertex_11 = (remapped_vertex_00 + W + 1)

    if mask is not None:
        def dilate(x, k=3):
            x = torch.nn.functional.conv2d(
                x.float()[None, None, ...],
                torch.ones(1, 1, k, k).to(mask.device),
                padding="same"
            )
            return x.squeeze() > 0

        # need dilated mask for "connecting vertices", e.g. face at the mask-edge connected to next masked-out vertex
        '''
        x---x---o
        | / | / |  
        x---o---o
        '''
        mask_dilated = dilate(mask, k=5)
        if pix_to_face is not None and faces is not None and vertices is not None:
            # replace dilated world_points with existing vertices (pix_to_face --> face --> closest vertex to world_points)
            # TODO: could also choose the existing vertex id for that face then (instead of duplicating the vertex) -- do not increase that face id with offset afterwards though
            mask_only_dilated = mask_dilated * ~mask
            face_verts = vertices.T[faces.T]  # (M, 3, 3) where face_verts[k, i] is i-th vertex of face k
            pix_to_face_only_dilated = pix_to_face[0, mask_only_dilated][..., 0]  # (N)
            face_verts = face_verts[pix_to_face_only_dilated]  # (N, 3, 3) where vertices[k, i] is "i-th vertex of face k" (:= V_ik)
            p_pred = world_space_points[:, mask_only_dilated.flatten()].T  # (N, 3) where p_pred[k] is "unprojected point corresponding to face k" (:= P_k)
            p_pred = p_pred[:, None, :].repeat(1, 3, 1)  # (N, 3, 3) where p_pred[k, i] is P_k for all i in [0..2]
            d = torch.linalg.vector_norm(p_pred - face_verts, dim=2)  # (N, 3) where d[k, i] is l2_dist(V_ik, P_k)
            closest_vertex_index = torch.argmin(d, dim=1)  # (N) where closest_vertex_index[k] is in [0..2] and signaling which i of V_ik is closest to P_k
            closest_vertex_index = closest_vertex_index[..., None, None].repeat(1, 1, 3)  # (N, 1, 3)
            closest_vertex = torch.gather(input=face_verts, dim=1, index=closest_vertex_index)  # (N, 1, 3) where closest_vertex[k, 0] is V_ik with i=closest_vertex_index[k]
            closest_vertex = closest_vertex.squeeze().t()  # (3, N)
            world_space_points[:, mask_only_dilated.flatten()] = closest_vertex  # replace P_k with V_ik

        # only keep vertices/features for faces that need to be added (i.e. are masked) -- rest of the faces are already present in 3D
        colors = colors[:, mask_dilated.flatten()]
        world_space_points = world_space_points[:, mask_dilated.flatten()]

        # remap vertex id's to shortened list of vertices
        remap = torch.bucketize(vertex_ids, vertex_ids[mask_dilated])
        remap[~mask_dilated] = -1  # mark invalid vertex_ids with -1 --> due to dilation + triangulation, a few faces will contain -1 values --> need to filter them
        remap = remap.flatten()
        mask_dilated = mask_dilated[:H-1, :W-1]
        vertex_00 = vertex_00[mask_dilated]
        vertex_01 = vertex_01[mask_dilated]
        vertex_10 = vertex_10[mask_dilated]
        vertex_11 = vertex_11[mask_dilated]
        remapped_vertex_00 = remap[vertex_00]
        remapped_vertex_01 = remap[vertex_01]
        remapped_vertex_10 = remap[vertex_10]
        remapped_vertex_11 = remap[vertex_11]

    # triangulation: upper-left and lower-right triangles from image structure
    faces_upper_left_triangle = torch.stack(
        [remapped_vertex_00.flatten(), remapped_vertex_10.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )
    faces_lower_right_triangle = torch.stack(
        [remapped_vertex_10.flatten(), remapped_vertex_11.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )

    # filter faces with -1 vertices and combine
    mask_upper_left = torch.all(faces_upper_left_triangle >= 0, dim=0)
    faces_upper_left_triangle = faces_upper_left_triangle[:, mask_upper_left]
    mask_lower_right = torch.all(faces_lower_right_triangle >= 0, dim=0)
    faces_lower_right_triangle = faces_lower_right_triangle[:, mask_lower_right]
    faces = torch.cat([faces_upper_left_triangle, faces_lower_right_triangle], dim=1)

    # apply surface normal threshold
    use_surface_normal_filter = surface_normal_threshold > -1
    if use_surface_normal_filter:
        # construct pix_to_face by exploiting triangulation scheme:
        # (1) get the face_ids for upper_left and lower_right branch
        face_id = torch.arange(faces.shape[1], dtype=torch.long, device=faces.device)  # (P + Q)
        upper_left_face_id = face_id[:faces_upper_left_triangle.shape[1]]  # (P)
        lower_right_face_id = face_id[faces_upper_left_triangle.shape[1]:]  # (Q)

        # (2) make pix_to_face a tensor with 6 dimensions, because we will map each vertex_id separately
        pix_to_face = torch.zeros((6, H*W), dtype=torch.long, device=faces.device) - 1

        # (3) each face consists of 3 vertex_ids (== flattened pixel coordinates), e.g. face_0 = [p0, p1, p3], face_1 = [p3, p1, p4]
        #     need to "scatter", but with variable amount of elements, e.g. p0 -> [face_0], p1 -> [face_0, face_1], ...
        #     implement it by calling scatter 2 times!
        #       (a) start with faces_upper_left. scatter the vertex_ids
        #       (d) repeat for faces_lower_right

        # upper_left triangle
        flattened_pixel_coordinates_upper_left = torch.stack([
            # upper_left triangle
            vertex_00[mask_upper_left],
            vertex_10[mask_upper_left],
            vertex_01[mask_upper_left]
        ], dim=0)  # (3, P)
        upper_left_face_id = torch.stack([upper_left_face_id]*3, dim=0)  # (3, P)
        pix_to_face[0:3].scatter_(dim=1, index=flattened_pixel_coordinates_upper_left, src=upper_left_face_id)

        # lower_right triangle
        flattened_pixel_coordinates_lower_right = torch.stack([
            vertex_10[mask_lower_right],
            vertex_11[mask_lower_right],
            vertex_01[mask_lower_right]
        ], dim=0)  # (3, Q)
        lower_right_face_id = torch.stack([lower_right_face_id]*3, dim=0)  # (3, Q)
        pix_to_face[3:6].scatter_(dim=1, index=flattened_pixel_coordinates_lower_right, src=lower_right_face_id)

        # (4) reshape pix_to_face to required shape: (H, W, 6)
        pix_to_face = pix_to_face.t().reshape(H, W, -1)

        # (5) filter out faces with surface_normal_filter and using the constructed pix_to_face for per-pixel view direction
        faces, _ = surface_normal_filter(
            vertices=world_space_points,
            faces=faces,
            H=H,
            W=W,
            world_to_cam=world_to_cam,
            fov_in_degrees=fov_in_degrees,
            surface_normal_threshold=surface_normal_threshold,
            pix_to_face=pix_to_face
        )

    # clean mesh
    world_space_points, faces, colors = clean_mesh(
        world_space_points,
        faces,
        colors,
        edge_threshold=edge_threshold,
        min_triangles_connected=1000 if use_surface_normal_filter else -1,
        fill_holes=use_surface_normal_filter
    )

    return world_space_points, faces, colors


class VertexColorShader(ShaderBase):
    def __init__(self, blend_soft=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blend_soft = blend_soft

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_soft:
            return softmax_rgb_blend(texels, fragments, blend_params)
        else:
            return hard_rgb_blend(texels, fragments, blend_params)


def render_mesh(vertices, faces, vertex_features, H, W, fov_in_degrees, RT, blur_radius=0.0, faces_per_pixel=1):
    """
    Renders a mesh using its vertex-features (e.g. rgb colors) into a novel view-point.

    :param vertices: points in world-space. shape=(3, N)
    :param faces: mesh faces. shape=(3, M)
    :param vertex_features: features for each vertex. shape=(F, N)
    :param H: target image height
    :param W: target image width
    :param fov_in_degrees: fov in degrees
    :param RT: world-to-cam matrix where cam specifies the target view-point
    :param blur_radius: Float distance in the range [0, 2] used to expand the face
            bounding boxes for rasterization. Setting blur radius
            results in blurred edges around the shape instead of a
            hard boundary. Set to 0 for no blur.
    :param faces_per_pixel: (int) Number of faces to keep track of per pixel.
            We return the nearest faces_per_pixel faces along the z-axis.
    :return: output_image, output_depth, output_mask
    """
    # create mesh
    texture = TexturesVertex(verts_features=[vertex_features.T])
    mesh = Meshes(verts=[vertices.T], faces=[faces.T], textures=texture)

    # Initialize a camera
    camera = get_camera(RT, fov_in_degrees)

    if vertex_features.shape[0] == 3:
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))
    elif vertex_features.shape[0] == 4:
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0, 0))

    # Define the settings for rasterization and shading
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True
    )

    # Create a renderer by composing a rasterizer and a shader
    # We simply render vertex colors through the custom VertexColorShader (no lighting, materials are used)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=VertexColorShader(
            blend_soft=False,
            device=RT.device,
            cameras=camera,
            blend_params=blend_params
        )
    )

    # renderer.shader.blend_params.background_color = torch.zeros_like(vertex_features[:,0])

    # Create a depth shader
    depth_shader = HardDepthShader(device=RT.device, cameras=camera)

    # render RGB and depth, get mask
    images, fragments = renderer(mesh)
    mask = (fragments.pix_to_face[..., 0] < 0).squeeze()
    depth = depth_shader(fragments, mesh).squeeze()
    depth[mask] = 0

    return images[0].permute(2, 0, 1), depth, mask, fragments.pix_to_face, fragments.zbuf
