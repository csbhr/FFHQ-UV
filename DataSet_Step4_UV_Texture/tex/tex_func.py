import numpy as np
import cv2
import skimage

from .laplacian_pyramid import LaplacianPyramid as LP
from utils import img2mask


def match_color_in_yuv(src_tex, dst_tex, mask, w=1.5):
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
    match_tex_yuv = (match_tex_yuv / w) * std_dst + mu_dst
    # yuv -> rgb
    match_tex = skimage.color.convert_colorspace(match_tex_yuv, "yuv", "rgb")
    match_tex = np.clip(match_tex, 0, 255)
    return match_tex


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


def remap_tex_from_input2D(input_img, seg_mask, projXY, norm, unwrap_uv_idx_v_idx, unwrap_uv_idx_bw):
    '''
    Remap texture from input 2D image to UV map.

    Args:
        input_img: numpy.array (h, w, 3). The input 2D image.
        seg_mask: numpy.array (h, w, 3). The parsing mask of facial parts (without eyes and mouth).
        projXY: numpy.array (N, 2). The project XY coordinates (h x w) for each vertex.
        norm: numpy.array (N, 3). The normal vector for each vertex.
        unwrap_uv_idx_v_idx: numpy.array (unwrap_size, unwrap_size, 3). The vertex indices for each UV pixel.
            It contains 3 vertices of the face on which the UV pixel is located. 
        unwrap_uv_idx_bw: numpy.array (unwrap_size, unwrap_size, 3). The barycentric coordinates for each UV pixel.
    Returns:
        remap_tex: numpy.array (unwrap_size, unwrap_size, 3). The remapped UV texture.
        remap_mask: numpy.array (unwrap_size, unwrap_size, 1). The mask of remapped texture.
    '''
    n_ver = projXY.shape[0]

    # Step 1. Find 2D image pixel coordinates for each UV map pixel
    # mesh vertex --> 2D image pixel (N, 1, 2)
    ver_XY = np.reshape(projXY, (n_ver, 1, 2))

    # UV map pixel --> mesh vertex (unwrap_size, unwrap_size)
    # each UV map pixel corresponding to 3 vertices and its barycentric coordinates
    uv_ver_map_y0 = unwrap_uv_idx_v_idx[:, :, 0].astype(np.float32)
    uv_ver_map_y1 = unwrap_uv_idx_v_idx[:, :, 1].astype(np.float32)
    uv_ver_map_y2 = unwrap_uv_idx_v_idx[:, :, 2].astype(np.float32)
    uv_ver_map_x = np.zeros_like(uv_ver_map_y0).astype(np.float32)

    # UV map pixel --> 2D image pixel (unwrap_size, unwrap_size, 2)
    # each UV map pixel corresponding to 3 2D image pixels and its barycentric coordinates
    uv_XY_0 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
    uv_XY_1 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
    uv_XY_2 = cv2.remap(ver_XY, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
    uv_XY = \
        uv_XY_0 * unwrap_uv_idx_bw[:, :, 0:1] + \
        uv_XY_1 * unwrap_uv_idx_bw[:, :, 1:2] + \
        uv_XY_2 * unwrap_uv_idx_bw[:, :, 2:3]

    # Step 2. remap texture pixel from 2D image to UV map (unwrap_size, unwrap_size, 2)
    remap_tex = cv2.remap(input_img, uv_XY[:, :, 0], uv_XY[:, :, 1], cv2.INTER_LINEAR)
    remap_tex = np.clip(remap_tex, 0., 255.)

    # Step 3. remap seg face masks to UV map, then get the segmentation mask (unwrap_size, unwrap_size, 1)
    remap_seg_mask0 = cv2.remap(seg_mask, uv_XY_0[:, :, 0], uv_XY_0[:, :, 1], cv2.INTER_LINEAR)
    remap_seg_mask1 = cv2.remap(seg_mask, uv_XY_1[:, :, 0], uv_XY_1[:, :, 1], cv2.INTER_LINEAR)
    remap_seg_mask2 = cv2.remap(seg_mask, uv_XY_2[:, :, 0], uv_XY_2[:, :, 1], cv2.INTER_LINEAR)
    remap_seg_mask = \
        remap_seg_mask0 * unwrap_uv_idx_bw[:, :, 0:1] + \
        remap_seg_mask1 * unwrap_uv_idx_bw[:, :, 1:2] + \
        remap_seg_mask2 * unwrap_uv_idx_bw[:, :, 2:3]
    remap_seg_mask = img2mask(remap_seg_mask, thre=0.5)

    # Step 5. remap visible vertices to UV map, then get the visible mask (unwrap_size, unwrap_size, 1)
    # compute visible vertices according to normal vectors (N, 1, 1)
    ver_vis_mask = np.reshape((norm[:, 2] > 0.1).astype(np.float32), (n_ver, 1, 1))
    remap_vis_mask0 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y0, cv2.INTER_NEAREST)
    remap_vis_mask1 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y1, cv2.INTER_NEAREST)
    remap_vis_mask2 = cv2.remap(ver_vis_mask, uv_ver_map_x, uv_ver_map_y2, cv2.INTER_NEAREST)
    remap_vis_mask = \
        remap_vis_mask0 * unwrap_uv_idx_bw[:, :, 0] + \
        remap_vis_mask1 * unwrap_uv_idx_bw[:, :, 1] + \
        remap_vis_mask2 * unwrap_uv_idx_bw[:, :, 2]
    remap_vis_mask = img2mask(remap_vis_mask, thre=0.5)

    # Step 6. combine the segmentation mask and the visible mask, then get the final remap mask
    remap_mask = remap_seg_mask * remap_vis_mask

    # Step 7. apply remap mask on the remap texture
    remap_tex = remap_tex * remap_mask

    return remap_tex, remap_mask


def blur_tex_mask_with_major_minor_valid_mask(tex_mask, major_valid_mask, minor_valid_mask):
    '''
    Blur the tex_mask, where consider the major_valid_mask and minor_valid_mask.
        major_valid_mask is contained in minor_valid_mask.
    For the minor_valid_mask region:
        After blur, the value of mask is smaller.
    For the major_valid_mask region:
        After blur, the value of mask is larger.
    Step:
        1. Use minor_valid_mask to restrict the tex_mask region;
        2. Blur the mask with a large blur kernel, then give a small weight (0.8);
        3. Set the major_valid_mask region back to one;
        4. Blur the mask again, but with a small blur kernel.

    Args:
        tex_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input facial texture.
        major_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of major valid region.
        minor_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of minor valid region.
    Returns:
        blur_mask_minor: numpy.array (unwrap_size, unwrap_size, 3). Intermediate result (minor region with large blur).
        blur_mask_major_one: numpy.array (unwrap_size, unwrap_size, 3). Intermediate result (set major region to one).
        blur_mask_final: numpy.array (unwrap_size, unwrap_size, 3). The final blurred mask.
    '''
    # Step 1. Use minor_valid_mask to restrict the tex_mask region
    minor_valid_mask = tex_mask * minor_valid_mask

    # Step 2. Blur the mask with a large blur kernel, then give a small weight (0.8)
    blur_mask_minor = cv2.blur(minor_valid_mask, (51, 51), 0) * 0.8

    # Step 3. Set the major_valid_mask region back to one;
    major_valid_mask = tex_mask * major_valid_mask
    one_map = np.ones_like(major_valid_mask)
    blur_mask_major_one = major_valid_mask * one_map + (1 - major_valid_mask) * blur_mask_minor

    # Step 4. Blur the mask again, but with a small blur kernel.
    blur_mask_final = cv2.blur(blur_mask_major_one, (31, 31), 0)

    return blur_mask_minor, blur_mask_major_one, blur_mask_final


def fill_facial_region(template_tex, input_tex, tex_mask, major_valid_mask, minor_valid_mask, mouth_mask):
    '''
    Fill the facial region, the invisible region is filled by template.

    Args:
        template_tex: numpy.array (unwrap_size, unwrap_size, 3). The template.
        input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input facial texture.
        tex_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input facial texture.
        major_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of major valid region.
        minor_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of minor valid region.
        mouth_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of mouth region.
    Returns:
        fill_tex: numpy.array (unwrap_size, unwrap_size, 3). The filled texture.
    '''

    # Step 1. template match color
    template_tex_match_color = match_color_in_yuv(src_tex=template_tex, dst_tex=input_tex, mask=tex_mask)

    # Step 2. process fill mask
    # erode the texture mask (blur + clip)
    tex_mask_erode = cv2.blur(tex_mask, (3, 3), 0)
    tex_mask_erode1 = cv2.blur(tex_mask_erode, (5, 5), 0)
    tex_mask_erode2 = cv2.blur(tex_mask_erode1, (9, 9), 0)
    tex_mask_erode1 = img2mask(tex_mask_erode1, thre=1., mode='greater-equal')
    tex_mask_erode2 = img2mask(tex_mask_erode2, thre=1., mode='greater-equal')

    # treat the mouth area specifically.
    tex_mask_erode = tex_mask_erode1 * mouth_mask + tex_mask_erode2 * (1 - mouth_mask)

    # blur the tex_mask_erode considering major_valid_mask and minor_valid_mask
    fill_mask_blur_minor, fill_mask_blur_major_one, fill_mask_blur = blur_tex_mask_with_major_minor_valid_mask(
        tex_mask=tex_mask_erode, major_valid_mask=major_valid_mask, minor_valid_mask=minor_valid_mask)

    # remove the pixels beyond tex_mask, avioding involve black pixel during blending
    zero_map = np.zeros_like(fill_mask_blur)
    fill_mask_blur = tex_mask_erode * fill_mask_blur + (1.0 - tex_mask_erode) * zero_map
    fill_mask_blur = cv2.blur(fill_mask_blur, (21, 21), 0)
    fill_mask_blur = tex_mask * fill_mask_blur + (1.0 - tex_mask) * zero_map

    # Step 3. blend with template UV map
    fill_tex = linear_blend(template_tex=template_tex_match_color, input_tex=input_tex, mask=fill_mask_blur)
    fill_mask = tex_mask_erode * minor_valid_mask

    return fill_tex, fill_mask


def blend_with_template(template_tex, input_tex, tex_mask, major_valid_mask, minor_valid_mask, hair_mask):
    '''
    Blend the input facial texture with the template.

    Args:
        template_tex: numpy.array (unwrap_size, unwrap_size, 3). The template.
        input_tex: numpy.array (unwrap_size, unwrap_size, 3). The input facial texture.
        tex_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of input facial texture.
        major_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of major valid region.
        minor_valid_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of minor valid region.
        hair_mask: numpy.array (unwrap_size, unwrap_size, 3). The mask of hair region.
    Returns:
        blend_result: numpy.array (unwrap_size, unwrap_size, 3). The final blend texture.
    '''

    # Step 1. template match color
    template_tex_match_color = match_color_in_yuv(src_tex=template_tex, dst_tex=input_tex, mask=tex_mask)

    # Step 2. process blend mask
    # blur the tex_mask considering major_valid_mask and minor_valid_mask
    blend_mask_blur_minor, blend_mask_blur_major_one, blend_mask_blur = blur_tex_mask_with_major_minor_valid_mask(
        tex_mask=tex_mask, major_valid_mask=major_valid_mask, minor_valid_mask=minor_valid_mask)

    # Step 3. blend with template UV map
    blend_tex = laplacian_pyramid_blend(template_tex=template_tex_match_color,
                                        input_tex=input_tex,
                                        mask=blend_mask_blur)

    # Step 4. cover hair region with template UV map (the mask is first eroded and then blurred)
    # erode mask
    hair_mask_erode = cv2.blur(hair_mask, (11, 11), 0)
    hair_mask_erode = img2mask(hair_mask_erode, thre=1., mode='greater-equal')
    # blur mask
    hair_mask_blur = cv2.blur(hair_mask_erode, (15, 15), 0)
    blend_result = linear_blend(template_tex=template_tex, input_tex=blend_tex, mask=(1 - hair_mask_blur))
    blend_result = np.clip(blend_result, 0, 255)

    return blend_result
