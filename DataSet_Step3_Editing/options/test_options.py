from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(phase='test')
        parser.add_argument('--network_pkl',
                            type=str,
                            default='../checkpoints/stylegan_model/stylegan2-ffhq-config-f.pkl')
        parser.add_argument('--max_result_snapshots', default=30, help='max result snapshots')

        parser.add_argument(
            "--proj_data_dir",
            type=str,
            required=True,
            help="The directory of the project data, which should inculde 'latents/lights/attributes' sub-directorys.")
        parser.add_argument("--flow_model_path",
                            type=str,
                            default='../checkpoints/styleflow_model/modellarge10k.pt',
                            help="The path of the pretrained styleflow model.")
        parser.add_argument("--exp_direct_path",
                            type=str,
                            default='../checkpoints/styleflow_model/expression_direction.pt',
                            help="The path of the expression direction.")
        parser.add_argument("--exp_recognition_path",
                            type=str,
                            default='../checkpoints/exprecog_model/FacialExpRecognition_model.t7',
                            help="The path of the expression recognition model.")
        parser.add_argument(
            "--edit_items",
            type=str,
            default='delight,norm_attr,multi_yaw',
            help="The edited items, optional:[delight, norm_attr, multi_yaw, multi_pitch], joint by ','.")
        parser.add_argument("--device", type=str, default='cuda', help="The device, optional: cup/cuda.")

        return parser