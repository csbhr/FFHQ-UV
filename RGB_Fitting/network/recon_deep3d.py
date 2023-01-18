import torch
import torch.nn as nn

from .resnet.backbone import func_dict, conv1x1, conv1x1_relu


def define_net_recon_deep3d(net_recon, use_last_fc=False, fc_dim_dict=None, pretrained_path=None):
    net = ReconNetWrapper(net_recon=net_recon,
                          use_last_fc=use_last_fc,
                          fc_dim_dict=fc_dim_dict,
                          pretrained_path=pretrained_path)
    net.eval()
    return net


class ReconNetWrapper(nn.Module):

    def __init__(self, net_recon, use_last_fc=False, fc_dim_dict=None, pretrained_path=None):
        super(ReconNetWrapper, self).__init__()

        self.use_last_fc = use_last_fc
        self.fc_dim_dict = fc_dim_dict

        self.fc_dim = fc_dim_dict['id_dims'] + fc_dim_dict['exp_dims'] + fc_dim_dict['tex_dims'] + 3 + 27 + 2 + 1

        # define backbone network
        if net_recon not in func_dict:
            return NotImplementedError('network [%s] is not implemented', net_recon)
        func, backbone_last_dim = func_dict[net_recon]
        self.backbone = func(use_last_fc=use_last_fc, num_classes=self.fc_dim)

        # define last fc network
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(backbone_last_dim, fc_dim_dict['id_dims'], bias=True),  # id layer
                conv1x1_relu(backbone_last_dim, fc_dim_dict['exp_dims'], bias=True),  # exp layer
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

        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu')['net_recon'])
            print('loading the pretrained recon model from %s' % pretrained_path)

    def forward(self, x, rgb_light=True):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        if rgb_light:
            return x
        else:
            id_exp_tex_angle = x[:, :-30]
            gamma = x[:, -30:-3]
            trans = x[:, -3:]
            gamma = gamma.reshape([-1, 3, 9])
            gamma_mean = torch.mean(gamma, dim=1)
            x_new = torch.cat([id_exp_tex_angle, gamma_mean, trans], dim=1)
            return x_new
