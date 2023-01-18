import torch
import torch.nn as nn
import numpy as np
from typing import List
import nvdiffrast.torch as dr


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0], [0, n / -x, 0, 0], [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


class MeshRenderer(nn.Module):

    def __init__(self, fov, znear=0.1, zfar=10, rasterize_size=224):
        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(ndc_projection(x=x, n=znear,
                                                    f=zfar)).matmul(torch.diag(torch.tensor([1., -1, -1, 1])))
        self.rasterize_size = rasterize_size
        self.ctx = None

    def forward(self, vertex, tri, feat=None, uv_map=None):
        '''
        Rasterize 3D mesh (from camera coordinate system to imaging coordinate system).

        The vertex texture and UV texture are supported.
        If uv_map is None, the feat is the vertex color, (B, N, 3);
        Else, the feat is the UV indices (512x512) and the vertex shading, (B, N, 5).

        Args:
            vertex: torch.Tensor, (B, N, 3). The vertices in the camera coordinate system.
            tri: torch.Tensor, (B, M, 3). The faces that need to be rendered.
            feat: torch.Tensor, (B, N, ?). The vertex features that need to be interpolated.
            uv_map: torch.Tensor, (B, 3, uvsize, uvsize). The UV texture map.
        Returns:
            mask: torch.Tensor, (B, 1, rsize, rsize). The mask of rendered pixels which are 3D mesh.
            depth: torch.Tensor, (B, 1, rsize, rsize). The depth of each rendered pixel.
            image: torch.Tensor, (B, 3, rsize, rsize). The rendered image.
        '''
        device = vertex.device
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)

        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=device)
            print("create cuda ctx on device cuda:%d" % (device.index))

        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        # from camera coordinate system to imaging coordinate system
        vertex_ndc = vertex @ ndc_proj.t()

        # handle that each vertex case has a separate tri
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device)
            fstartidx = torch.cumsum(fnum, dim=0) - fnum
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i * vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vertex: [B*N, 4], tri: [B*M, 3], for instance_mode vertex: [B, N, 4], tri: [M, 3]
        vertex_ndc = vertex_ndc.contiguous()
        tri = tri.type(torch.int32).contiguous()
        # Rasterization (B, rsize, rsize, 4), the face index and the barycentric coordinates
        rast_out, _ = dr.rasterize(self.ctx, vertex_ndc, tri, resolution=[rsize, rsize], ranges=ranges)

        # the mask of rendered pixels which are 3D mesh (the face index is not -1)
        # (B, 1, rsize, rsize)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)

        # the depth of each rendered pixel
        # (B, rsize, rsize, 1)
        depth, _ = dr.interpolate(vertex[..., 2:3].contiguous(), rast_out, tri)
        # (B, 1, rsize, rsize)
        depth = depth.permute(0, 3, 1, 2)
        depth = mask * depth

        # the rendered image
        image = None
        if feat is not None:

            # vertex texture
            if uv_map is None:
                image, _ = dr.interpolate(feat, rast_out, tri)
                image = image.permute(0, 3, 1, 2)
                image = image * mask

            # UV texture
            else:
                feat, _ = dr.interpolate(feat, rast_out, tri)
                feat = feat.permute(0, 3, 1, 2)
                uv_coord = feat[:, :2, :, :]
                shading = feat[:, 2:, :, :]

                # transfer UV coordinates range from 0~512 to 0~uv_size
                uv_bs, _, uv_size, _ = uv_map.size()
                uv_coord = (uv_coord * uv_size) / 512
                uv_coord = torch.clamp(uv_coord, 0, uv_size).long()

                # flatten UV map and UV coordinates, (...uv_size, uv_size...) -> (...uv_size**2...)
                uv_map_flat = uv_map.reshape([uv_bs, 3, -1])  # (B, 3, uv_size**2)
                uv_coord_flat = uv_coord[:, 0, :, :] * uv_size + uv_coord[:, 1, :, :]
                uv_coord_flat = uv_coord_flat.reshape([uv_bs, -1])  # (B, uv_size**2)

                # remap texture
                image = torch.zeros([uv_bs, 3, rsize * rsize]).to(device)
                for b in range(uv_bs):
                    image[b, 0, :] = torch.take(uv_map_flat[b, 0, :], uv_coord_flat[b, :])
                    image[b, 1, :] = torch.take(uv_map_flat[b, 1, :], uv_coord_flat[b, :])
                    image[b, 2, :] = torch.take(uv_map_flat[b, 2, :], uv_coord_flat[b, :])
                image = image.reshape([uv_bs, 3, rsize, rsize])
                image = image * shading * mask

        return mask, depth, image
