import torch
import torch.nn as nn

from .recon_network import ReconNetWrapper
from .parametric_face_model import ParametricFaceModel
from .renderer import compute_224projXY_norm_by_pin_hole


class Face3d_Recon_API(nn.Module):

    def __init__(self,
                 pfm_model_path,
                 recon_model_path='./epoch_latest.pth',
                 focal=1015.0,
                 camera_distance=10.0,
                 device='cuda'):
        '''
        Args:
            pfm_model_path: str. The file path of parametric face model (3DMM).
            recon_model_path: str. The pretrained model of reconstruction network.
            focal: float. The render setting.
            camera_distance: float. The render setting.
            device: str. The device.
        '''
        super(Face3d_Recon_API, self).__init__()

        self.focal = focal
        self.camera_distance = camera_distance
        self.device = device

        self.facemodel = ParametricFaceModel(model_path=pfm_model_path, device=device)

        fc_dim_dict = {
            'id_dims': self.facemodel.id_dims,
            'exp_dims': self.facemodel.exp_dims,
            'tex_dims': self.facemodel.tex_dims
        }
        self.recon_net = ReconNetWrapper(fc_dim_dict=fc_dim_dict, pretrain_model_path=recon_model_path, device=device)

        self.facemodel.to(device)
        self.recon_net.eval().to(device)

    def to(self, device):
        self.device = device
        self.facemodel.to(device)
        self.recon_net.to(device)

    def pred_coeffs(self, input_img):
        '''
        Predict 3DMM coefficients from input image.

        Args:
            input_img: torch.Tensor, (B, 3, 224, 224). The input image.
        Returns:
            pred_coeffs_dict: dict. It contains:
                'id': torch.Tensor, (B, 532).
                'exp': torch.Tensor, (B, 45).
                'tex': torch.Tensor, (B, 439).
                'angle': torch.Tensor, (B, 3).
                'trans': torch.Tensor, (B, 3).
                'gamma': torch.Tensor, (B, 3, 9).
        '''
        with torch.no_grad():
            pred_coeffs_dict = self.recon_net(input_img)
            return pred_coeffs_dict

    def compute_224projXY_norm_by_pin_hole(self, coeffs_dict):
        '''
        Compute project XY coordinates (224x224) and normal vectors for each vertex by pin hole camera.

        Args:
            coeffs_dict: dict.
        Returns:
            face_projXY: torch.Tensor, (B, N, 2). The project XY coordinates (224x224) for each vertex.
            face_norm: torch.Tensor, (B, N, 3). The normal vector for each vertex.
        '''
        face_shape, _ = self.facemodel.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
        mesh_info = {'v': face_shape, 'f_v': self.facemodel.head_buf, 'v_f': self.facemodel.point_buf}
        face_projXY, face_norm = compute_224projXY_norm_by_pin_hole(mesh_info=mesh_info,
                                                                    angle=coeffs_dict['angle'],
                                                                    trans=coeffs_dict['trans'],
                                                                    camera_distance=self.camera_distance,
                                                                    focal=self.focal)
        return face_projXY, face_norm

    def save_mesh(self, coeffs_dict, mesh_path, mtl_path, uv_name, is_neutral=False):
        '''
        Save 3D mesh to obj file.

        Args:
            coeffs_dict: dict.
            mesh_path: str. The file path of saved mash obj file.
            mtl_path: str. The file path of saved mtl file.
            uv_name: str. The file name of saved UV map.
            is_neutral: bool. If true, save neutral shape.
        '''
        face_shape, id_shape = self.facemodel.compute_shape(coeffs_dict['id'], coeffs_dict['exp'])
        if is_neutral:
            save_shape = id_shape.cpu().numpy()[0]
        else:
            save_shape = face_shape.cpu().numpy()[0]

        mesh_info = {
            'v': save_shape,
            'vt': self.facemodel.vt_list.cpu().numpy(),
            'fv': self.facemodel.head_buf.cpu().numpy(),
            'fvt': self.facemodel.head_tri_vt.cpu().numpy(),
            'mtl_name': mtl_path.split('/')[-1]
        }
        self.creat_mtl(mtl_path, uv_name)
        self.write_mesh_obj(mesh_info, mesh_path)

    def creat_mtl(self, mtl_path, uv_path='albedo.png'):
        with open(mtl_path, 'w') as fp:
            fp.write('newmtl blinn1SG\n')
            fp.write('Ka 0.200000 0.200000 0.200000\n')
            fp.write('Kd 1.000000 1.000000 1.000000\n')
            fp.write('Ks 1.000000 1.000000 1.000000\n')
            fp.write('map_Kd ' + uv_path)

    def write_mesh_obj(self, mesh_info, file_path):
        v = mesh_info['v']
        vt = mesh_info['vt'] if 'vt' in mesh_info else None
        vn = mesh_info['vn'] if 'vn' in mesh_info else None
        fv = mesh_info['fv']
        fvt = mesh_info['fvt'] if 'fvt' in mesh_info else None
        fvn = mesh_info['fvn'] if 'fvn' in mesh_info else None
        mtl_name = mesh_info['mtl_name'] if 'mtl_name' in mesh_info else None

        with open(file_path, 'w') as fp:
            # write mtl info
            if mtl_name:
                fp.write(f'mtllib {mtl_name}\n')
            # write vertices
            for x, y, z in v:
                fp.write('v %f %f %f\n' % (x, y, z))
            # write vertex textures (UV coordinates)
            if vt is not None:
                for u, v in vt:
                    fp.write('vt %f %f\n' % (u, v))
            # write vertex normal
            if vn is not None:
                for x, y, z in vn:
                    fp.write('vn %f %f %f\n' % (x, y, z))
            # write faces
            if fvt is None and fvn is None:  # fv only
                for v_list in fv:
                    v_list = v_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        fp.write('f %d %d %d\n' % (v1, v2, v3))
                    else:
                        v1, v2, v3, v4 = v_list
                        fp.write('f %d %d %d %d\n' % (v1, v2, v3, v4))
            elif fvn is None:  # fv/fvt
                for v_list, vt_list in zip(fv, fvt):
                    v_list = v_list + 1
                    vt_list = vt_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        t1, t2, t3 = vt_list
                        fp.write('f %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3))
                    else:
                        v1, v2, v3, v4 = v_list
                        t1, t2, t3, t4 = vt_list
                        fp.write('f %d/%d %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3, v4, t4))
            else:  # fv/fvt/fvn
                for v_list, vt_list, vn_list in zip(fv, fvt, fvn):
                    v_list = v_list + 1
                    vt_list = vt_list + 1
                    vn_list = vn_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        t1, t2, t3 = vt_list
                        n1, n2, n3 = vn_list
                        fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (v1, t1, n1, v2, t2, n2, v3, t3, n3))
                    else:
                        v1, v2, v3, v4 = v_list
                        t1, t2, t3, t4 = vt_list
                        n1, n2, n3, n4 = vn_list
                        fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                                 (v1, t1, n1, v2, t2, n2, v3, t3, n3, v4, t4, n4))
