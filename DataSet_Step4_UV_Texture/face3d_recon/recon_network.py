import numpy as np
import torch
import torch.nn as nn

from .resnet_backbone import func_dict, conv1x1, conv1x1_relu


class ReconNetWrapper(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 use_last_fc=False,
                 fc_dim_dict=None,
                 limit_exp_range=True,
                 pretrain_model_path='./epoch_latest.pth',
                 init_SH=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 device='cuda'):
        super(ReconNetWrapper, self).__init__()

        self.use_last_fc = use_last_fc
        self.fc_dim_dict = fc_dim_dict
        self.device = device

        self.fc_dim = fc_dim_dict['id_dims'] + fc_dim_dict['exp_dims'] + fc_dim_dict['tex_dims'] + 3 + 27 + 2 + 1

        # define backbone network
        if backbone_name not in func_dict:
            return NotImplementedError('network [%s] is not implemented', backbone_name)
        func, backbone_last_dim = func_dict[backbone_name]
        self.backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)

        # define last fc network
        if not use_last_fc:
            if limit_exp_range:
                self.final_layers = nn.ModuleList([
                    conv1x1(backbone_last_dim, fc_dim_dict['id_dims'], bias=True),  # id layer
                    conv1x1_relu(backbone_last_dim, fc_dim_dict['exp_dims'], bias=True),  # exp layer
                    conv1x1(backbone_last_dim, fc_dim_dict['tex_dims'], bias=True),  # tex layer
                    conv1x1(backbone_last_dim, 3, bias=True),  # angle layer
                    conv1x1(backbone_last_dim, 27, bias=True),  # gamma layer
                    conv1x1(backbone_last_dim, 2, bias=True),  # tx, ty
                    conv1x1(backbone_last_dim, 1, bias=True)  # tz
                ])
            else:
                self.final_layers = nn.ModuleList([
                    conv1x1(backbone_last_dim, fc_dim_dict['id_dims'], bias=True),  # id layer
                    conv1x1(backbone_last_dim, fc_dim_dict['exp_dims'], bias=True),  # exp layer
                    conv1x1(backbone_last_dim, fc_dim_dict['tex_dims'], bias=True),  # tex layer
                    conv1x1(backbone_last_dim, 3, bias=True),  # angle layer
                    conv1x1(backbone_last_dim, 27, bias=True),  # gamma layer
                    conv1x1(backbone_last_dim, 2, bias=True),  # tx, ty
                    conv1x1(backbone_last_dim, 1, bias=True)  # tz
                ])

            def init_weights(mm):
                if type(mm) == nn.Conv2d:
                    nn.init.constant_(mm.weight, 0.)

            for m in self.final_layers:
                try:
                    nn.init.constant_(m.weight, 0.)
                    nn.init.constant_(m.bias, 0.)
                except:
                    m.apply(init_weights)

        # the initial SH coeff that need to be added to the output SH coeffs.
        self.init_SH = torch.from_numpy(init_SH.reshape([1, 1, -1]).astype(np.float32)).float().to(device)

        self.backbone.to(device)
        self.final_layers.to(device)
        self.load_state_dict(torch.load(pretrain_model_path, map_location=device)['net_recon'])
        print('loading the recon model from %s' % pretrain_model_path)

    def to(self, device):
        self.device = device
        self.init_SH = self.init_SH.to(device)
        self.backbone.to(device)
        self.final_layers.to(device)

    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        coeffs_dict = self.get_coeffs(x)
        return coeffs_dict

    def get_coeffs(self, net_output):
        id_dims = self.fc_dim_dict['id_dims']
        exp_dims = self.fc_dim_dict['exp_dims']
        tex_dims = self.fc_dim_dict['tex_dims']

        id_coeffs = net_output[:, :id_dims]
        exp_coeffs = net_output[:, id_dims:id_dims + exp_dims]
        tex_coeffs = net_output[:, id_dims + exp_dims:id_dims + exp_dims + tex_dims]
        angle = net_output[:, id_dims + exp_dims + tex_dims:id_dims + exp_dims + tex_dims + 3]
        gamma = net_output[:, id_dims + exp_dims + tex_dims + 3:id_dims + exp_dims + tex_dims + 3 + 27]
        translations = net_output[:, id_dims + exp_dims + tex_dims + 3 + 27:]

        gamma = gamma.reshape(-1, 3, 9)
        gamma = gamma + self.init_SH

        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angle,
            'gamma': gamma,
            'trans': translations
        }
