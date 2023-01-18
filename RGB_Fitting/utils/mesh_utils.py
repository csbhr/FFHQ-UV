import numpy as np
import cv2
import re
import skimage

from .laplacian_pyramid import LaplacianPyramid as LP
from .data_utils import img2mask


def unwrap_vertex_to_uv(attr_vertex, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw):
    '''
    Unwrap the 3D attributes from vertex formation to UV formation, use 'bilinear' for interpolation.
    Each UV map pixel corresponding to 3 vertices and their barycentric coordinates

    Args:
        attr_vertex: numpy.array (N, *). The input 3D attributes in vertex formation.
        unwrap_uv_idx_v_idx: numpy.array (unwrap_size, unwrap_size, 3). The vertex indices for each UV pixel.
        unwrap_uv_idx_bw: numpy.array (unwrap_size, unwrap_size, 3). The barycentric coordinates for each UV pixel.
    Returns:
        attr_uv: numpy.array (unwrap_size, unwrap_size, *). The remapped 3D attributes in UV formation.
    '''
    # reshape (N, *) -> (N, 1, *)
    N, C = attr_vertex.shape
    attr_vertex = np.reshape(attr_vertex, (N, 1, C))

    # the indices of the 1st-axis (N)
    uv_ver_map_y0 = unwrap_uv_idx_v_idx[:, :, 0].astype(np.float32)
    uv_ver_map_y1 = unwrap_uv_idx_v_idx[:, :, 1].astype(np.float32)
    uv_ver_map_y2 = unwrap_uv_idx_v_idx[:, :, 2].astype(np.float32)
    # the indices of the 2nd-axis (1)
    uv_ver_map_x = np.zeros_like(uv_ver_map_y0).astype(np.float32)

    # remap attributes from vertex form to UV form
    attr_uv_0 = cv2.remap(attr_vertex, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_LINEAR)
    attr_uv_1 = cv2.remap(attr_vertex, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_LINEAR)
    attr_uv_2 = cv2.remap(attr_vertex, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_LINEAR)
    # inerpolate using barycentric coordinates
    attr_uv = \
        attr_uv_0 * unwrap_uv_idx_bw[:, :, 0:1] + \
        attr_uv_1 * unwrap_uv_idx_bw[:, :, 1:2] + \
        attr_uv_2 * unwrap_uv_idx_bw[:, :, 2:3]

    return attr_uv


def laplacian_pyramid_blend(template_tex, input_tex, mask, times=5):
    '''
    Blend using Laplacian Pyramid.

    Args:
        template_tex: numpy.array (unwrap_size, unwrap_size, 3). The template texture.
        input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input texture.
        mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input texture.
        times: numpy.array (unwrap_size, unwrap_size, 3). The number of Laplacian Pyramid levels.
    Returns:
        blend_tex: numpy.array (unwrap_size, unwrap_size, 3). The blended texture.
    '''
    pyramids_template = LP.buildLaplacianPyramids(template_tex, times)
    pyramids_input = LP.buildLaplacianPyramids(input_tex, times)
    mask_list = LP.downSamplePyramids(mask, times)
    pyramids_blend = []
    for i in range(len(pyramids_template)):
        mask = np.clip(mask_list[i], 0, 1)
        pyramids_blend.append(pyramids_input[i] * mask + pyramids_template[i] * (1 - mask))
    blend_tex = LP.reconstruct(pyramids_blend)
    blend_tex = np.clip(blend_tex, 0, 255)
    return blend_tex


def linear_blend(template_tex, input_tex, mask):
    '''
    Blend with linear manner. c = a * m + b * (1 - m)

    Args:
        template_tex: numpy.array (unwrap_size, unwrap_size, 3). The template texture.
        input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input texture.
        mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input texture.
    Returns:
        blend_tex: numpy.array (unwrap_size, unwrap_size, 3). The blended texture.
    '''
    return input_tex * mask + template_tex * (1 - mask)


def match_color_in_yuv(src_tex, dst_tex, mask):
    '''
    Match color (src_tex -> dst_tex) in YUV color space.

    Args:
        src_tex: numpy.array (unwrap_size, unwrap_size, 3). The source UV texture (template texture).
        dst_tex: numpy.array (unwrap_size, unwrap_size, 3). The target UV texture (input texture).
        mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of valid region.
    Returns:
        match_tex: numpy.array (unwrap_size, unwrap_size, 3). The matched UV texture.
    '''
    # rgb -> yuv
    dst_tex_yuv = skimage.color.convert_colorspace(dst_tex, "rgb", "yuv")
    src_tex_yuv = skimage.color.convert_colorspace(src_tex, "rgb", "yuv")
    # status
    is_valid = mask[:, :, 0] > 0.5
    mu_dst = np.mean(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    std_dst = np.std(dst_tex_yuv[is_valid], axis=0, keepdims=True)
    mu_src = np.mean(src_tex_yuv[is_valid], axis=0, keepdims=True)
    std_src = np.std(src_tex_yuv[is_valid], axis=0, keepdims=True)
    # match
    match_tex_yuv = (src_tex_yuv - mu_src) / std_src
    match_tex_yuv = (match_tex_yuv / 1.5) * std_dst + mu_dst
    # yuv -> rgb
    match_tex = skimage.color.convert_colorspace(match_tex_yuv, "yuv", "rgb")
    match_tex = np.clip(match_tex, 0, 255)
    return match_tex


def blend_uv_with_template(res_uv, template_uv, hair_mask, blend_mask):
    # for blur kernel size
    uv_size = res_uv.shape[0]
    r = float(uv_size) / 1024.

    # match color
    template_uv_match_color = match_color_in_yuv(src_tex=template_uv, dst_tex=res_uv, mask=blend_mask)

    # blend with template
    ks = int(31 * r)
    ks = ks if ks % 2 == 1 else ks - 1
    blend_mask_blur = cv2.blur(blend_mask, (ks, ks), 0)
    blend_uv = laplacian_pyramid_blend(template_tex=template_uv_match_color, input_tex=res_uv, mask=blend_mask_blur)

    # cover hair
    ks = int(15 * r)
    ks = ks if ks % 2 == 1 else ks - 1
    hair_mask_erode = cv2.blur(hair_mask, (ks, ks), 0)
    hair_mask_erode = img2mask(hair_mask_erode, thre=1., mode='greater-equal')
    hair_mask_blur = cv2.blur(hair_mask_erode, (ks, ks), 0)
    blend_result = linear_blend(template_tex=template_uv, input_tex=blend_uv, mask=(1 - hair_mask_blur))
    blend_result = np.clip(blend_result, 0, 255)

    return blend_result


def write_mtl(mtl_path, uv_path='albedo.png'):
    with open(mtl_path, 'w') as fp:
        fp.write('newmtl blinn1SG\n')
        fp.write('Ka 0.200000 0.200000 0.200000\n')
        fp.write('Kd 1.000000 1.000000 1.000000\n')
        fp.write('Ks 1.000000 1.000000 1.000000\n')
        fp.write('map_Kd ' + uv_path)


def read_mesh_obj(file_path):
    vertices = []  # v
    vertices_texture = []  # vt
    vertices_normal = []  # vn

    face_v = []  # f 1 2 3
    face_vt = []  # f 1/1 2/2 3/3
    face_vn = []  # f 1/1/1 2/2/2 3/3/3

    lines = open(file_path, 'r').readlines()
    for line in lines:
        line = re.sub(' +', ' ', line)
        if line.startswith('v '):
            toks = line.strip().split(' ')[1:]
            try:
                vertices.append([float(toks[0]), float(toks[1]), float(toks[2])])
            except Exception:
                print(toks)
        elif line.startswith('vt '):
            toks = line.strip().split(' ')[1:]
            vertices_texture.append([float(toks[0]), float(toks[1])])
        elif line.startswith('vn '):
            toks = line.strip().split(' ')[1:]
            vertices_normal.append([float(toks[0]), float(toks[1]), float(toks[2])])
        elif line.startswith('f '):
            toks = line.strip().split(' ')[1:]
            if len(toks) == 3:  # tri faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2]], np.int32) - 1)

            if len(toks) == 4:  # quad faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')
                faces4 = toks[3].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0], faces4[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1], faces4[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2], faces4[2]], np.int32) - 1)

    results = {}
    results['v'] = np.array(vertices, np.float32)
    if len(vertices_texture) > 0:
        results['vt'] = np.array(vertices_texture, np.float32)
    if len(vertices_normal) > 0:
        results['vn'] = np.array(vertices_normal, np.float32)

    if len(face_v) > 0:
        results['fv'] = face_v
    if len(face_vt) > 0:
        results['fvt'] = face_vt
    if len(face_vn) > 0:
        results['fvn'] = face_vn

    return results


def write_mesh_obj(mesh_info, file_path):
    v = mesh_info['v']
    vt = mesh_info['vt'] if 'vt' in mesh_info else None
    vn = mesh_info['vn'] if 'vn' in mesh_info else None
    fv = mesh_info['fv'] if 'fv' in mesh_info else None
    fvt = mesh_info['fvt'] if 'fvt' in mesh_info else None
    fvn = mesh_info['fvn'] if 'fvn' in mesh_info else None
    mtl_name = mesh_info['mtl_name'] if 'mtl_name' in mesh_info else None

    if vt is None:
        rgb_tex = False
    elif vt.shape[1] == 2:
        rgb_tex = False
    elif vt.shape[1] == 3:
        rgb_tex = True

    with open(file_path, 'w') as fp:
        # write mtl info
        if mtl_name is not None:
            fp.write(f'mtllib {mtl_name}\n')

        # write vertices
        if rgb_tex:
            for (x, y, z), (r, g, b) in zip(v, vt):
                fp.write('v %f %f %f %f %f %f\n' % (x, y, z, r, g, b))
        else:
            for x, y, z in v:
                fp.write('v %f %f %f\n' % (x, y, z))

        # write vertex textures (UV coordinates)
        if vt is not None and not rgb_tex:
            for u, v in vt:
                fp.write('vt %f %f\n' % (u, v))

        # write vertex normal
        if vn is not None:
            for x, y, z in vn:
                fp.write('vn %f %f %f\n' % (x, y, z))

        # write faces
        if fv is not None:  # have face
            if rgb_tex or (fvt is None and fvn is None):  # fv only
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
