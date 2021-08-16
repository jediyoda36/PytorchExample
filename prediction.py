import argparse
import os

import torch
import torchvision
from torchvision import transforms
import cv2
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Car:
    def __init__(self, model_path):

        if torch.cuda.is_available():
            self.model = torch.load(model_path).to(device)
        else:
            self.model = torch.load(model_path, map_location='cpu')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 将图像转化为128 * 128
            transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 归一化
        ])

    def detect(self, image):

        image = self.transform(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        outputs = self.model(image)
        prob = F.softmax(outputs, dim=1)
        pred = torch.argmax(prob, dim=1)
        pred = pred.cpu().numpy()
        return pred[0]


def main():

    car = Car('./model/car-50.pth')
    root = './data/train/blured'
    img_list = [f for f in os.listdir(root) if f.endswith('.bmp')]
    for img in img_list:
        image = cv2.imread(os.path.join(root, img))
        pred = car.detect(image)
        if pred == 0:
            print('Normal')
            # cv2.imshow('test', image)
            # cv2.waitKey(0)
        else:
            print('Blured')
        # cv2.imshow('test', image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
