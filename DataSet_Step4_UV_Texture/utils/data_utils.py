from PIL import Image
import numpy as np
import torch
import cv2


def np2pillow(img, src_range=255.):
    coef = 255. / src_range
    return Image.fromarray(np.squeeze(np.clip(np.round(img * coef), 0, 255).astype(np.uint8)))


def pillow2np(img, dst_range=255.):
    coef = dst_range / 255.
    return np.asarray(img, np.float32) * coef


def read_img(path, resize=None, dst_range=255.):
    img = Image.open(path)

    if resize is not None:
        img = img.resize(resize)

    img = pillow2np(img, dst_range)

    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def save_img(img, path, src_range=255.):
    np2pillow(img, src_range).save(path)


def save_mask_cover_img(mask, img, path, src_range=255., alpha=0.3):
    '''mask: float (0, 1)'''
    if mask.ndim == 2:
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
    if mask.shape[2] == 1:
        mask = np.tile(mask, (1, 1, 3))
    if mask.shape[2] > 3:
        mask = mask[:, :, :3]
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(img.shape[:2]), np.float32) / 255.0

    cover_img = np.ones_like(img)
    cover_img[:, :, 0] = 0.
    cover_img[:, :, 1] = 1.
    cover_img[:, :, 2] = 0.
    cover_img = cover_img * src_range

    mask = np.clip(mask, 0, 1)
    final_img = (1 - mask) * img + mask * (cover_img * alpha + img * (1 - alpha))
    save_img(final_img, path, src_range)


def save_coord_cover_img(coord, img, path, src_range=255., radius=5):
    '''coord: float (n, 2)'''
    assert coord.ndim == 2 and coord.shape[1] == 2
    n = coord.shape[0]
    h, w, c = img.shape

    final_img = img.copy()
    for i in range(n):
        x, y = round(coord[i, 0]), round(coord[i, 1])
        cv2.circle(final_img, (x, w - y), radius, (255, 0, 0), -1)

    save_img(final_img, path, src_range)


def np2tensor(img, src_range=255., dst_range=1., device='cuda'):
    coef = dst_range / src_range
    return torch.from_numpy(np.array(img, dtype=np.float32) * coef).float().permute(2, 0, 1).unsqueeze(0).to(device)


def tensor2np(img, src_range=1., dst_range=255.):
    coef = dst_range / src_range
    return (img * coef).squeeze(0).permute(1, 2, 0).cpu().numpy()


def img3channel(img):
    '''make the img to have 3 channels'''
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def img2mask(img, thre=128, mode='greater'):
    '''mode: greater/greater-equal/less/less-equal/equal'''
    if mode == 'greater':
        mask = (img > thre).astype(np.float32)
    elif mode == 'greater-equal':
        mask = (img >= thre).astype(np.float32)
    elif mode == 'less':
        mask = (img < thre).astype(np.float32)
    elif mode == 'less-equal':
        mask = (img <= thre).astype(np.float32)
    elif mode == 'equal':
        mask = (img == thre).astype(np.float32)
    else:
        raise NotImplementedError

    mask = img3channel(mask)

    return mask
