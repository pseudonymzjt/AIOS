# vgg_cpu.py
import os

import numpy as np
import scipy.io
from layer_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer
from layer_2 import ConvolutionalLayer, FlattenLayer, MaxPoolingLayer
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 探测 VGG19 权重文件
possible_param_paths = [
    'imagenet-vgg-verydeep-19.mat',
    '../data/vgg19_data/imagenet-vgg-verydeep-19.mat',
    '../../data/vgg19_data/imagenet-vgg-verydeep-19.mat',
    os.path.join(CURRENT_DIR, 'imagenet-vgg-verydeep-19.mat'),
    os.path.join(CURRENT_DIR, '..', 'data', 'vgg19_data', 'imagenet-vgg-verydeep-19.mat'),
    '/opt/code_chap_2_3/data/vgg19_data/imagenet-vgg-verydeep-19.mat',
    '/workspace/dataset/private/datasets/vgg19/imagenet-vgg-verydeep-19.mat'
]

PARAM_PATH = None
for p in possible_param_paths:
    if os.path.exists(p):
        PARAM_PATH = p
        break

if PARAM_PATH is None:
    raise FileNotFoundError("未找到 imagenet-vgg-verydeep-19.mat，请检查路径或执行下载！")

# 2. 探测测试图片 cat1.jpg
possible_img_paths = [
    'cat1.jpg',
    '../cat1.jpg',
    os.path.join(CURRENT_DIR, 'cat1.jpg'),
    '/opt/code_chap_2_3/code_chap_2_3_student/exp_3_1_vgg/cat1.jpg',
    '/workspace/dataset/private/datasets/vgg19/cat1.jpg'
]

IMG_PATH = None
for p in possible_img_paths:
    if os.path.exists(p):
        IMG_PATH = p
        break

class VGG19(object):
    def __init__(self, param_path='imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'Softmax'
        )

    def build_model(self):
        self.layers = {}
        # Layer 1
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)
        # Layer 2
        self.layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(3, 128, 128, 1, 1)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2)
        # Layer 3
        self.layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_4'] = ReLULayer()       
        self.layers['pool3'] = MaxPoolingLayer(2, 2)
        # Layer 4
        self.layers['conv4_1'] = ConvolutionalLayer(3, 256, 512, 1, 1)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_4'] = ReLULayer()       
        self.layers['pool4'] = MaxPoolingLayer(2, 2)
        # Layer 5
        self.layers['conv5_1'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_3'] = ReLULayer()
        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_4'] = ReLULayer()       
        self.layers['pool5'] = MaxPoolingLayer(2, 2)
        # Flatten & FC
        self.layers['flatten'] = FlattenLayer([512, 7, 7], [512 * 7 * 7])
        self.layers['fc6'] = FullyConnectedLayer(512 * 7 * 7, 4096)
        self.layers['relu6'] = ReLULayer()
        self.layers['fc7'] = FullyConnectedLayer(4096, 4096)
        self.layers['relu7'] = ReLULayer()
        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)
        self.layers['Softmax'] = SoftmaxLossLayer()
        
        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        for idx in range(43):
            if 'conv' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = np.transpose(weight, [2, 0, 1, 3])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)
            if idx >= 37 and 'fc' in self.param_layer_name[idx]:
                weight, bias = params['layers'][0][idx - 1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                bias = bias.reshape(-1)
                self.layers[self.param_layer_name[idx]].load_param(weight, bias)

    def forward(self):
        current = self.input_image
        for idx in range(len(self.param_layer_name)):
            current = self.layers[self.param_layer_name[idx]].forward(current)
        return current

    def evaluate(self):
        prob = self.forward()
        top1 = np.argmax(prob[0])
        print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))

    def load_image(self, image_dir):
            print('Loading and preprocessing image from ' + image_dir)
            # 使用 PIL 打开图片并转为 RGB 格式（防止单通道或RGBA导致通道数对不上）
            img = Image.open(image_dir).convert('RGB')
            
            # 缩放到 224x224 尺寸（注意 PIL 的 resize 传参是 (width, height)）
            img = img.resize((224, 224), Image.BILINEAR)

            self.input_image = np.array(img, dtype=np.float32)
            self.input_image -= self.image_mean
            self.input_image = np.expand_dims(self.input_image, axis=0)
            self.input_image = np.transpose(self.input_image, [0, 3, 1, 2])

if __name__ == '__main__':
    vgg = VGG19(param_path=PARAM_PATH)
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image(IMG_PATH)
    vgg.evaluate()