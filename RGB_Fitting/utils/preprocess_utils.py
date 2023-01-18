import torch
import numpy as np
from PIL import Image
from skimage import transform as trans
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


# resize and crop images
def resize_n_crop_img(img, lm, t, s, target_size=224., skin_mask=None, parse_mask=None):
    w0, h0 = img.size
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if skin_mask is not None:
        skin_mask = skin_mask.resize((w, h), resample=Image.BICUBIC)
        skin_mask = skin_mask.crop((left, up, right, below))

    if parse_mask is not None:
        parse_mask = parse_mask.resize((w, h), resample=Image.BICUBIC)
        parse_mask = parse_mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - np.reshape(np.array([(w / 2 - target_size / 2), (h / 2 - target_size / 2)]), [1, 2])

    return img, lm, skin_mask, parse_mask


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([
        lm[lm_idx[0], :],
        np.mean(lm[lm_idx[[1, 2]], :], 0),
        np.mean(lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]
    ],
                    axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def align_img(img, lm, lm3D, skin_mask=None, parse_mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        skin_mask_new      --PIL.Image  (target_size, target_size, 3)
        parse_mask_new     --PIL.Image  (target_size, target_size, 3)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (68, 3)
        skin_mask          --PIL.Image  (raw_H, raw_W, 3)
        parse_mask         --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    lm5p = lm if lm.shape[0] == 5 else extract_5p(lm)
    lm3D = lm3D if lm3D.shape[0] == 5 else extract_5p(lm3D)

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor / s

    # processing the image
    img_new, lm_new, skin_mask_new, parse_mask_new = resize_n_crop_img(img,
                                                                       lm,
                                                                       t,
                                                                       s,
                                                                       target_size=target_size,
                                                                       skin_mask=skin_mask,
                                                                       parse_mask=parse_mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, skin_mask_new, parse_mask_new


# utils for face recognition model
def estimate_norm(lm_68p, H):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
                   dtype=np.float32)
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]


# utils for face recognition model
def estimate_norm_torch(lm_68p, H):
    lm_68p_ = lm_68p.detach().cpu().numpy()
    M = []
    for i in range(lm_68p_.shape[0]):
        M.append(estimate_norm(lm_68p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
    return M