import numpy as np
import cv2
import scipy.sparse as sps
import scipy.sparse.linalg as linalg


def get_laplacian_matrices(img_H, img_W):
    num_pxls = img_H * img_W

    identity_mat = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')
    laplacian_mat = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')

    for i in range(num_pxls):
        identity_mat[i, i] = 1

        laplacian_mat[i, i] = 4
        if i > img_W:
            laplacian_mat[i, i - img_W] = -1
        if i % img_W != 0:
            laplacian_mat[i, i - 1] = -1
        if i + img_W < num_pxls:
            laplacian_mat[i, i + img_W] = -1
        if i % img_W != img_W - 1:
            laplacian_mat[i, i + 1] = -1

    return identity_mat, laplacian_mat


def get_poisson_linlsq_A_b(template_img, input_img, mask, laplacian_mat=None, identity_mat=None):
    H, W = mask.shape[:2]
    assert input_img.ndim == 2 and template_img.ndim == 2 and mask.ndim == 2
    assert input_img.shape[0] == H and input_img.shape[1] == W
    assert template_img.shape[0] == H and template_img.shape[1] == W

    if laplacian_mat is None or identity_mat is None:
        identity_mat, laplacian_mat = get_laplacian_matrices(H, W)

    lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)
    input_img = cv2.filter2D(input_img, -1, lap_kernel)

    input_img = input_img.flatten(order='C')
    template_img = template_img.flatten(order='C')
    mask = (mask > 0.5).astype(np.float32).flatten(order='C')

    # The sparse matrix is element-wise multiplied with the broadcast dense 1D array
    # refer to https://mlog.club/article/1966054
    mask_sps = sps.lil_matrix((H * W, H * W))
    mask_sps.setdiag(mask)
    mask_inv_sps = sps.lil_matrix((H * W, H * W))
    mask_inv_sps.setdiag(1 - mask)
    A = mask_sps * laplacian_mat + mask_inv_sps * identity_mat
    b = input_img * mask + template_img * (1 - mask)

    return A, b


def poisson_blend(template_img, input_img, mask, laplacian_mat=None, identity_mat=None):
    template_img = template_img / 255.
    input_img = input_img / 255.
    H, W, C = input_img.shape

    res_img_list = []
    for i in range(C):
        te_im = template_img[:, :, i]
        in_im = input_img[:, :, i]
        m = mask[:, :, i]

        A, b = get_poisson_linlsq_A_b(te_im, in_im, m, laplacian_mat, identity_mat)
        res = np.reshape(linalg.spsolve(A, b), (H, W))
        res_img_list.append(res)

    res_img = np.stack(res_img_list, axis=-1) * 255.
    return res_img
