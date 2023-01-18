from operator import le
from tkinter import E
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from PIL import Image

from . import transforms
from .vgg import VGG


class Exp_Recog_API:

    def __init__(self, model_path, cut_size=44, device='cuda'):
        '''
        Args:
            model_path: str. The pretrained facial expression recognition model.
            cut_size: int. The size of input image.
            device: str. The device.
        '''

        self.transform = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.net = VGG('VGG19')
        self.net.load_state_dict(torch.load(model_path)['net'])
        self.net.to(device)
        self.net.eval()

        self.cut_size = int(cut_size)
        self.device = device

    def preprocess(self, img):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input RGB image.
        Returns:
            inputs: torch.Tensor, (10, 3, cut_size, cut_size). The crops form the input image after preprocess.
        '''
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        gray_img = resize(gray_img, (self.cut_size, self.cut_size), mode='symmetric').astype(np.uint8)
        img = gray_img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = self.transform(img)
        return inputs

    def __call__(self, img):
        '''
        Args:
            img: numpy.array, float, (h, w, 3). The input image.
        Returns:
            predicted_label: str. The predicted expression (Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral).
            score_dict: dict. The score of each expression.
        '''
        with torch.no_grad():
            inputs = self.preprocess(img).to(self.device)
            ncrops, c, h, w = inputs.size()

            outputs = self.net(inputs)
            outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)

            predicted_label = self.class_names[int(predicted.cpu().numpy())]

            score_dict = {}
            for i, exp in enumerate(self.class_names):
                score_dict[exp] = float(score[i].cpu().numpy())

            return predicted_label, score_dict
