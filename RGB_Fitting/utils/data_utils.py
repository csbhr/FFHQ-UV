from PIL import Image
import numpy as np
import torch
import cv2
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def np2pillow(img, src_range=255.):
    coef = 255. / src_range
    return Image.fromarray(np.squeeze(np.clip(np.round(img * coef), 0, 255).astype(np.uint8)))


def pillow2np(img, dst_range=255.):
    coef = dst_range / 255.
    return np.asarray(img, np.float32) * coef


def read_img(path, resize=None, rescale=1.0, dst_range=255.):
    img = Image.open(path).convert('RGB')

    if resize is not None:
        img = img.resize(resize)
    
    if rescale != 1.0:
        w, h = img.size
        img = img.resize((int(w * rescale), int(h * rescale)))

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


def resize_img(img, resize=None, src_range=255.):
    img = np2pillow(img, src_range)
    if resize is not None:
        img = img.resize(resize)
    img = pillow2np(img, src_range)
    return img


def np2tensor(img, src_range=255., dst_range=1., device='cuda'):
    coef = dst_range / src_range
    return torch.from_numpy(np.array(img, dtype=np.float32) * coef).float().permute(2, 0, 1).unsqueeze(0).to(device)


def tensor2np(img, src_range=1., dst_range=255.):
    coef = dst_range / src_range
    return (img[0] * coef).permute(1, 2, 0).detach().cpu().numpy()


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


def draw_mask(img, mask, src_range=255., alpha=0.3):
    '''img (H, W, 3), mask (H, W, :) (0~1)'''
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
    return final_img


def draw_landmarks(img, landmarks, color='r', radius=2):
    '''img (H, W, 3), landmarks (:, 2)'''
    if color == 'r':
        c = np.array([255., 0, 0])
    elif color == 'g':
        c = np.array([0, 255., 0])
    elif color == 'b':
        c = np.array([0, 0, 255.])
    else:
        raise NotImplementedError

    H, W, _ = img.shape
    img, landmarks = img.copy(), landmarks.copy()
    landmarks = np.round(landmarks).astype(np.int32)
    for i in range(landmarks.shape[0]):
        x, y = landmarks[i, 0], landmarks[i, 1]
        for j in range(-radius, radius):
            for k in range(-radius, radius):
                u = np.clip(x + j, 0, W - 1)
                v = np.clip(y + k, 0, H - 1)
                img[v, u] = c
    return img


def combine_row_txt_images(img_list, txt_list=None, resize=None):
    '''
    img_list: List[numpy.array (h, w, 3)], a list of images with same size.
    txt_list: List[str], a list of txt label, if None, do not add lable.
    resize: Tuple(dh, dw), if None, do nothing.
    '''
    if resize is not None:
        img_list = [pillow2np(np2pillow(img).resize(resize)) for img in img_list]
    h, w = img_list[0].shape[:2]
    line_img = np.concatenate(img_list, axis=1)
    if txt_list is not None:
        blank = np.ones(shape=(25, w * len(img_list), 3), dtype='uint8') * 255
        for i, txt in enumerate(txt_list):
            cv2.putText(blank, txt, (5 + i * w, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        line_img = np.concatenate([line_img, blank], axis=0)
    return line_img


def combine_col_txt_images(img_list, txt_list=None, resize=None):
    '''
    img_list: List[numpy.array (h, w, 3)], a list of images with same size.
    txt_list: List[str], a list of txt label, if None, do not add lable.
    resize: Tuple(dh, dw), if None, do nothing.
    '''
    if resize is not None:
        img_list = [pillow2np(np2pillow(img).resize(resize)) for img in img_list]
    h, w = img_list[0].shape[:2]
    line_img = np.concatenate(img_list, axis=0)
    if txt_list is not None:
        blank = np.ones(shape=(25, h * len(img_list), 3), dtype='uint8') * 255
        for i, txt in enumerate(txt_list):
            cv2.putText(blank, txt, (5 + i * h, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        blank = np.fliplr(np.transpose(blank, axes=(1, 0, 2)))
        line_img = np.concatenate([line_img, blank], axis=1)
    return line_img
