# vgg19_demo.py

import os
import sys
import time
import numpy as np
import scipy.io
from PIL import Image

try:
    import pycnnl
except ImportError:
    print("[Warning] 未检测到 pycnnl 库，请确保在 DLP 实验环境中运行或已编译 cnnl_python 库。")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------- 自动探测文件路径 -----------------
def find_file(filename, search_paths):
    for path in search_paths:
        if os.path.exists(path):
            return path
    return filename

PARAM_PATH = find_file('imagenet-vgg-verydeep-19.mat', [
    'imagenet-vgg-verydeep-19.mat',
    '../data/vgg19_data/imagenet-vgg-verydeep-19.mat',
    '../../data/vgg19_data/imagenet-vgg-verydeep-19.mat',
    os.path.join(CURRENT_DIR, 'imagenet-vgg-verydeep-19.mat'),
    os.path.join(CURRENT_DIR, '..', 'data', 'vgg19_data', 'imagenet-vgg-verydeep-19.mat'),
    '/opt/code_chap_2_3/data/vgg19_data/imagenet-vgg-verydeep-19.mat'
])

LABEL_PATH = find_file('synset_words.txt', [
    'synset_words.txt',
    '../synset_words.txt',
    os.path.join(CURRENT_DIR, 'synset_words.txt'),
    '/opt/code_chap_2_3/code_chap_2_3_student/exp_3_1_vgg/synset_words.txt'
])

CAT_IMAGE_PATH = find_file('cat1.jpg', [
    'cat1.jpg',
    '../cat1.jpg',
    os.path.join(CURRENT_DIR, 'cat1.jpg'),
    '/opt/code_chap_2_3/code_chap_2_3_student/exp_3_1_vgg/cat1.jpg'
])


class VGG19(object):
    def __init__(self):
        # 创建 pycnnl.CnnlNet() 实例 net
        self.net = pycnnl.CnnlNet()

    def _make_shape4(self, d0, d1, d2, d3):
        """辅助函数：创建4维 IntVector"""
        v = pycnnl.IntVector(4)
        v[0] = d0
        v[1] = d1
        v[2] = d2
        v[3] = d3
        return v

    def _add_pool(self, name, kernel_size=2, stride=2):
        """自适应调用池化层接口"""
        if hasattr(self.net, 'createPoolingLayer'):
            self.net.createPoolingLayer(name, kernel_size, stride)
        elif hasattr(self.net, 'createPoolLayer'):
            self.net.createPoolLayer(name, kernel_size, stride)
        else:
            raise AttributeError("pycnnl.CnnlNet 未找到池化层创建接口！")

    def build_model(self, param_path=PARAM_PATH):
        self.param_path = param_path
        self.net.setInputShape(1, 3, 224, 224)

        # ---------------- Stage 1 ----------------
        self.net.createConvLayer('conv1_1', self._make_shape4(1, 3, 224, 224), 64, 3, 1, 1, 1)
        self.net.createReLuLayer('relu1_1')
        self.net.createConvLayer('conv1_2', self._make_shape4(1, 64, 224, 224), 64, 3, 1, 1, 1)
        self.net.createReLuLayer('relu1_2')
        self._add_pool('pool1', 2, 2)

        # ---------------- Stage 2 ----------------
        self.net.createConvLayer('conv2_1', self._make_shape4(1, 64, 112, 112), 128, 3, 1, 1, 1)
        self.net.createReLuLayer('relu2_1')
        self.net.createConvLayer('conv2_2', self._make_shape4(1, 128, 112, 112), 128, 3, 1, 1, 1)
        self.net.createReLuLayer('relu2_2')
        self._add_pool('pool2', 2, 2)

        # ---------------- Stage 3 ----------------
        self.net.createConvLayer('conv3_1', self._make_shape4(1, 128, 56, 56), 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_1')
        self.net.createConvLayer('conv3_2', self._make_shape4(1, 256, 56, 56), 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_2')
        self.net.createConvLayer('conv3_3', self._make_shape4(1, 256, 56, 56), 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_3')
        self.net.createConvLayer('conv3_4', self._make_shape4(1, 256, 56, 56), 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_4')
        self._add_pool('pool3', 2, 2)

        # ---------------- Stage 4 ----------------
        self.net.createConvLayer('conv4_1', self._make_shape4(1, 256, 28, 28), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_1')
        self.net.createConvLayer('conv4_2', self._make_shape4(1, 512, 28, 28), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_2')
        self.net.createConvLayer('conv4_3', self._make_shape4(1, 512, 28, 28), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_3')
        self.net.createConvLayer('conv4_4', self._make_shape4(1, 512, 28, 28), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_4')
        self._add_pool('pool4', 2, 2)

        # ---------------- Stage 5 ----------------
        self.net.createConvLayer('conv5_1', self._make_shape4(1, 512, 14, 14), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_1')
        self.net.createConvLayer('conv5_2', self._make_shape4(1, 512, 14, 14), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_2')
        self.net.createConvLayer('conv5_3', self._make_shape4(1, 512, 14, 14), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_3')
        self.net.createConvLayer('conv5_4', self._make_shape4(1, 512, 14, 14), 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_4')
        self._add_pool('pool5', 2, 2)

        # ---------------- FC Layers ----------------
        # fc6 (25088 -> 4096)
        in_shape_fc6 = self._make_shape4(1, 1, 1, 512 * 7 * 7)
        w_shape_fc6  = self._make_shape4(1, 1, 512 * 7 * 7, 4096)
        out_shape_fc6 = self._make_shape4(1, 1, 1, 4096)
        self.net.createMlpLayer('fc6', in_shape_fc6, w_shape_fc6, out_shape_fc6)
        self.net.createReLuLayer('relu6')

        # fc7 (4096 -> 4096)
        in_shape_fc7 = self._make_shape4(1, 1, 1, 4096)
        w_shape_fc7  = self._make_shape4(1, 1, 4096, 4096)
        out_shape_fc7 = self._make_shape4(1, 1, 1, 4096)
        self.net.createMlpLayer('fc7', in_shape_fc7, w_shape_fc7, out_shape_fc7)
        self.net.createReLuLayer('relu7')

        # fc8 (4096 -> 1000)
        in_shape_fc8 = self._make_shape4(1, 1, 1, 4096)
        w_shape_fc8  = self._make_shape4(1, 1, 4096, 1000)
        out_shape_fc8 = self._make_shape4(1, 1, 1, 1000)
        self.net.createMlpLayer('fc8', in_shape_fc8, w_shape_fc8, out_shape_fc8)

        # Softmax
        input_shapes = pycnnl.IntVector(3)
        input_shapes[0] = 1
        input_shapes[1] = 1
        input_shapes[2] = 1000
        self.net.createSoftmaxLayer('Softmax', input_shapes, 1)

    def load_image(self, image_dir):
        """读取图像并预处理（转为 float64 传给 DLP）"""
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
        print('Loading and preprocessing image from ' + image_dir)
        
        # 使用 Pillow 兼容读取
        img = Image.open(image_dir).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        input_image = np.array(img, dtype=np.float32)
        
        input_image -= image_mean
        input_image = np.reshape(input_image, [1] + list(input_image.shape))
        # NCHW 格式转换
        input_image = np.transpose(input_image, [0, 3, 1, 2])
        # pycnnl 接口要求 float64
        self.input_data = input_image.flatten().astype(np.float64)
        self.net.setInputData(self.input_data)

    def load_model(self):
        """加载神经网络参数"""
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))

        count = 0
        for idx in range(self.net.size()):
            layer_name = self.net.getLayerName(idx)
            if 'conv' in layer_name:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # MatConvNet: [height, width, in_channel, out_channel]
                # pycnnl: [out_channel, height, width, in_channel]
                weight = np.transpose(weight, [3, 0, 1, 2]).flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)
                count += 1
            if 'fc' in layer_name:
                # 官方模型不含 flatten 层，读取需偏移 -1
                weight, bias = params['layers'][0][idx - 1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                weight = weight.flatten().astype(np.float64)
                bias = bias.reshape(-1).astype(np.float64)
                self.net.loadParams(idx, weight, bias)
                count += 1
        print("Successfully loaded parameters for %d layers." % count)

    def forward(self):
        return self.net.forward()

    def get_top5(self, label=None):
        start = time.time()
        self.forward()
        end = time.time()
        print('inference time: %f' % (end - start))
        result = self.net.getOutputData()
        
        top1 = False
        top5 = False
        print('----- Top 5 of ' + str(self.image) + ' -----')
        prob = sorted(list(result), reverse=True)[:6]
        
        if label is not None and result.index(prob[0]) == label:
            top1 = True
            
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if label is not None and idx == label:
                top5 = True
            label_name = self.labels[idx].strip() if hasattr(self, 'labels') and len(self.labels) > idx else str(idx)
            print('%f : %s' % (top, label_name))
            
        return top1, top5

    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        self.labels = []
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r') as f:
                self.labels = f.readlines()

        start = time.time()
        with open(file_list, 'r') as f:
            lines = f.readlines()
            total_num = len(lines)
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                image = parts[0]
                label = int(parts[1])
                self.load_image(image)
                top1, top5 = self.get_top5(label)
                if top1:
                    top1_num += 1
                if top5:
                    top5_num += 1

        end = time.time()
        if total_num > 0:
            print('Global accuracy : ')
            print('accuracy1: %f (%d/%d)' % (float(top1_num) / float(total_num), top1_num, total_num))
            print('accuracy5: %f (%d/%d)' % (float(top5_num) / float(total_num), top5_num, total_num))
            print('Total execution time: %f' % (end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    
    # 评测模式判断：若提供了测试集列表文件则批量评测，否则对单张图片做前向推断
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        vgg.evaluate(sys.argv[1])
    elif os.path.exists('file_list'):
        vgg.evaluate('file_list')
    else:
        # 单张图片测试（默认 tabby cat，真实标签 281）
        print("Running single image inference on %s..." % CAT_IMAGE_PATH)
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH, 'r') as f:
                vgg.labels = f.readlines()
        vgg.load_image(CAT_IMAGE_PATH)
        vgg.get_top5(label=281)