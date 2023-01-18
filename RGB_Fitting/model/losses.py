import numpy as np
import torch
import torch.nn.functional as F


def perceptual_loss(id_featureA, id_featureB):
    '''
    Recognition id feature level loss.
    Args:
        id_featureA, id_featureB: torch.Tensor, (B, -1).
    '''
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]


def photo_loss(imageA, imageB, mask):
    return photo_loss_l1(imageA, imageB, mask)


def photo_loss_l1(imageA, imageB, mask):
    '''
    Image level loss with a mask.
    L1 norm.
    Args:
        imageA, imageB: torch.Tensor, (B, 3, H, W).
        mask: torch.Tensor, (B, 1, H, W).
    '''
    imageA = imageA * mask
    imageB = imageB * mask
    loss = F.l1_loss(imageA, imageB, reduction='sum') / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss


def photo_loss_l2(imageA, imageB, mask):
    '''
    Image level loss with a mask.
    L2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Args:
        imageA, imageB: torch.Tensor, (B, 3, H, W).
        mask: torch.Tensor, (B, 1, H, W).
    '''
    eps = 1e-6
    loss = torch.sqrt(eps + torch.sum((imageA - imageB)**2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss


def vgg_loss(image_A, image_B, vgg_model):
    '''
    Vgg Loss, L2 norm.
    Args:
        imageA: torch.Tensor, (B, 3, H, W).
        image_B_features: torch.Tensor, (B, -1), the feature of target image_B.
        vgg_model: the vgg model.
    '''
    if image_A.shape[2] > 256:
        image_A = F.interpolate(image_A, size=(256, 256), mode='area')
    if image_B.shape[2] > 256:
        image_B = F.interpolate(image_B, size=(256, 256), mode='area')
    image_A = image_A * 255.
    image_B = image_B * 255.
    image_A_features = vgg_model(image_A, resize_images=False, return_lpips=True)
    image_B_features = vgg_model(image_B, resize_images=False, return_lpips=True)
    dist = (image_A_features - image_B_features).square().sum()
    return dist


def landmark_loss(predict_lm, gt_lm, weight=None):
    '''
    Weighted mse loss on 68/86 landmarks.
    Args:
        predict_lm, gt_lm: torch.Tensor, (B, 68/86, 2).
        weight: torch.Tensor, (1, 68/86).
    '''
    n_lmk = predict_lm.shape[1]
    assert gt_lm.shape[1] == n_lmk
    if not weight:
        if n_lmk == 68:
            weight = np.ones([68])
            weight[28:31] = 20
            weight[-8:] = 20
        elif n_lmk == 86:
            weight = np.ones([86])
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


def reflectance_loss(texture, mask):
    """
    minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo.
    Args:
        texture: torch.Tensors, (B, N, 3).
        mask: torch.Tensors, (N), 1 or 0.
    """
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss


def latents_geocross_loss(latents):
    '''Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors'''
    if (latents.shape[1] == 1):
        return 0
    else:
        X = latents.view(-1, 1, 18, 512)
        Y = latents.view(-1, 18, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.).sum()
        return D


def latents_mean_reg_loss(latents, mean_latents):
    loss = torch.sum(torch.mean((latents - mean_latents)**2, dim=(1, 2)))
    return loss


def uvtex_symmetry_reg_loss(uvtex, mask):
    mask_flip = torch.fliplr(mask)
    mask = mask * mask_flip
    uxtex_flip = torch.fliplr(uvtex)
    loss = torch.sum(torch.mean(((uvtex - uxtex_flip) * mask)**2, dim=(1, 2, 3)))
    return loss


def coeffs_reg_loss(coeffs_dict):
    '''
    coeffs regulization
    l2 norm without the sqrt, from yu's implementation (mse)
    tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    Args:
        coeffs_dict: a dict of torch.Tensors, keys: id, exp, tex, gamma.
    '''
    # coefficient regularization to ensure plausible 3d faces
    bs = coeffs_dict['id'].shape[0]
    loss_reg_id = torch.sum(coeffs_dict['id']**2) / bs
    loss_reg_exp = torch.sum(coeffs_dict['exp']**2) / bs
    loss_reg_tex = torch.sum(coeffs_dict['tex']**2) / bs

    # gamma regularization to ensure a nearly-monochromatic light
    if coeffs_dict['gamma'].size()[1] == 27:
        gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
        gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
        loss_reg_gamma = torch.mean((gamma - gamma_mean) ** 2)
    else:
        loss_reg_gamma = 0.

    return loss_reg_id, loss_reg_exp, loss_reg_tex, loss_reg_gamma
