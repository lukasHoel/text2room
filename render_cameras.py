from model.text2room_pipeline import Text2RoomPipeline
from model.utils.opt import get_default_parser


def main(args):
    # do not need models, only inference
    pipeline = Text2RoomPipeline(args, setup_models=False)

    # load existing poses that we want to render
    pipeline.load_poses(args.cameras_file, convert_from_nerf=args.convert_cameras_from_nerf_convention, replace_existing=True)
    pipeline.args.n_images = len(pipeline.seen_poses)

    # load mesh that we want to render
    pipeline.load_mesh(args.mesh_file)

    # render the images
    pipeline.save_seen_trajectory_renderings(add_to_nerf_images=True)
    pipeline.save_nerf_transforms()


if __name__ == "__main__":
    parser = get_default_parser()

    # GENERAL CONFIG
    parser.add_argument('--mesh_file', '-m', required=True)
    parser.add_argument('--cameras_file', '-c', required=True)
    parser.add_argument('--convert_cameras_from_nerf_convention', required=False, action="store_true")

    args = parser.parse_args()

    main(args)
