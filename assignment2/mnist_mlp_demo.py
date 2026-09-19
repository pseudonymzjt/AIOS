# -*- coding: utf-8 -*-
import os
import sys
import time
import struct
import numpy as np

# 导入 pycnnl
import pycnnl

# 自动定位当前脚本目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# 🚨 必须与 weight.npy 保持绝对一致的全局常量 (调整为 256 * 384)
HIDDEN1 = 256
HIDDEN2 = 384
OUT = 10


def load_mnist_data(file_dir, is_images=True):
    """
    底层读取 IDX 格式文件
    """
    with open(file_dir, 'rb') as bin_file:
        bin_data = bin_file.read()

    if is_images:
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1

    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    return mat_data


def to_int_vector(*dims):
    vec = pycnnl.IntVector(len(dims))
    for i, d in enumerate(dims):
        vec[i] = int(d)
    return vec


class MNIST_MLP(object):
    def __init__(self):
        """
        初始化网络容器
        """
        self.net = pycnnl.CnnlNet()
        self.test_data = None
        self.batch_size = 10000
        self.input_size = 784
        # 🚨 修正：与全局保持一致
        self.hidden1 = HIDDEN1
        self.hidden2 = HIDDEN2
        self.out_classes = OUT

    def load_data(self, data_path, label_path):
        """
        读取测试集并进行与训练端严格一致的标准化预处理
        """
        test_images = load_mnist_data(data_path, is_images=True)
        test_labels = load_mnist_data(label_path, is_images=False)

        # 保持与 cpu 端训练一致的标准化
        test_images = (test_images.astype(np.float32) / 255.0 - 0.1307) / 0.3081

        self.test_data = np.append(test_images, test_labels, axis=1)

    def build_model(self, batch_size=10000, input_size=784, hidden1=HIDDEN1, hidden2=HIDDEN2, out_classes=OUT):
        """
        定义网络拓扑结构：
        设置网络输入形状，构建 FC1 -> ReLU1 -> FC2 -> ReLU2 -> FC3 -> Softmax
        """
        self.batch_size = int(batch_size)
        self.input_size = int(input_size)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)
        self.out_classes = int(out_classes)

        # 按照官方规范格式设定输入形状 (batch_size, input_size, 1, 1)
        self.net.setInputShape(self.batch_size, self.input_size, 1, 1)

        # FC1
        self.net.createMlpLayer(
            'fc1',
            to_int_vector(self.batch_size, 1, 1, self.input_size),
            to_int_vector(self.batch_size, 1, self.input_size, self.hidden1),
            to_int_vector(self.batch_size, 1, 1, self.hidden1)
        )
        self.net.createReLuLayer('relu1')

        # FC2
        self.net.createMlpLayer(
            'fc2',
            to_int_vector(self.batch_size, 1, 1, self.hidden1),
            to_int_vector(self.batch_size, 1, self.hidden1, self.hidden2),
            to_int_vector(self.batch_size, 1, 1, self.hidden2)
        )
        self.net.createReLuLayer('relu2')

        # FC3
        self.net.createMlpLayer(
            'fc3',
            to_int_vector(self.batch_size, 1, 1, self.hidden2),
            to_int_vector(self.batch_size, 1, self.hidden2, self.out_classes),
            to_int_vector(self.batch_size, 1, 1, self.out_classes)
        )

        # Softmax (注意是 3 维形状)
        self.net.createSoftmaxLayer('softmax', to_int_vector(self.batch_size, 1, self.out_classes), 1)

    def load_model(self, param_dir='weight.npy'):
        if not os.path.exists(param_dir):
            local_param = os.path.join(CURRENT_DIR, os.path.basename(param_dir))
            if os.path.exists(local_param):
                param_dir = local_param
            else:
                local_weight = os.path.join(CURRENT_DIR, 'weight.npy')
                if os.path.exists(local_weight):
                    param_dir = local_weight

        params = np.load(param_dir, allow_pickle=True, encoding="latin1").item()

        # 转为 float32 并 .tolist() 传入底层
        w1 = params['w1'].flatten().astype(np.float32).tolist()
        b1 = params['b1'].flatten().astype(np.float32).tolist()
        self.net.loadParams(0, w1, b1)

        w2 = params['w2'].flatten().astype(np.float32).tolist()
        b2 = params['b2'].flatten().astype(np.float32).tolist()
        self.net.loadParams(2, w2, b2)

        w3 = params['w3'].flatten().astype(np.float32).tolist()
        b3 = params['b3'].flatten().astype(np.float32).tolist()
        self.net.loadParams(4, w3, b3)

    def forward(self):
        return self.net.forward()

    def evaluate(self):
        pred_results = np.zeros(self.test_data.shape[0])
        num_batches = self.test_data.shape[0] // self.batch_size

        for idx in range(num_batches):
            batch_images = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size, :-1]
            data = batch_images.flatten().tolist()
            self.net.setInputData(data)

            start = time.time()
            self.forward()
            end = time.time()
            print('inferencing time: %f' % (end - start))

            prob = self.net.getOutputData()
            prob = np.array(prob).reshape((self.batch_size, self.out_classes))
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size:(idx + 1) * self.batch_size] = pred_labels

        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)
        return accuracy


def find_dataset_path(filename):
    search_dirs = [
        CURRENT_DIR,
        os.path.join(CURRENT_DIR, '..', 'mnist_data'),
        os.path.join(CURRENT_DIR, 'mnist_data'),
        os.path.join(CURRENT_DIR, '..', 'data', 'mnist_mlp_data', 'mnist_data'),
        '/workspace/dataset/private/datasets/mnist_data'
    ]
    for d in search_dirs:
        full_p = os.path.join(d, filename)
        if os.path.exists(full_p):
            return full_p
    return filename


if __name__ == '__main__':
    batch_size = 10000
    mlp = MNIST_MLP()
    mlp.build_model(batch_size=batch_size, hidden1=HIDDEN1, hidden2=HIDDEN2, out_classes=OUT)

    test_data_path = find_dataset_path('t10k-images-idx3-ubyte')
    test_label_path = find_dataset_path('t10k-labels-idx1-ubyte')
    mlp.load_data(test_data_path, test_label_path)

    weight_path = os.path.join(CURRENT_DIR, 'weight.npy')
    mlp.load_model(weight_path)

    for i in range(3):
        mlp.evaluate()