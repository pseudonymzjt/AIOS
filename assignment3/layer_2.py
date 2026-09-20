# layer_2.py
import time

import numpy as np


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, version='fast'):
        """
        :param version: 'fast' 为高效向量化版本 (img2col + GEMM)；'slow' 为传统多重循环版本
        """
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.version = version

    def init_param(self, std=0.01):
        self.weight = np.random.normal(
            loc=0.0, scale=std,
            size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
        )
        self.bias = np.zeros([self.channel_out])

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, input, version=None):
        ver = version if version is not None else self.version
        if ver == 'fast':
            return self.forward_fast(input)
        else:
            return self.forward_slow(input)

    def forward_slow(self, input):
        """传统四重循环版本（教材代码 3.2）"""
        self.input = input  # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2 
        width = self.input.shape[3] + self.padding * 2 
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding + self.input.shape[2], 
                             self.padding:self.padding + self.input.shape[3]] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])

        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        h_start = idxh * self.stride
                        w_start = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.input_pad[idxn, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                            * self.weight[:, :, :, idxc]
                        ) + self.bias[idxc]
        return self.output

    def forward_fast(self, input):
        """高效向量化矩阵乘法版本（img2col + BLAS 矩阵内积）"""
        self.input = input  # [N, C, H, W]
        N, C, H, W = self.input.shape
        height = H + self.padding * 2 
        width = W + self.padding * 2 

        # 边界扩充
        self.input_pad = np.zeros([N, C, height, width], dtype=self.input.dtype)
        self.input_pad[:, :, self.padding:self.padding + H, self.padding:self.padding + W] = self.input

        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1

        # 卷积核展开: (C_in * K * K, C_out)
        weight_reshape = self.weight.reshape([-1, self.channel_out])

        # 特征图向量化 (img2col): (N * H_out * W_out, C_in * K * K)
        img2col = np.zeros(
            [N * height_out * width_out, C * self.kernel_size * self.kernel_size],
            dtype=self.input.dtype
        )
        col_idx = 0
        for idxn in range(N):
            for idxh in range(height_out):
                h_start = idxh * self.stride
                h_end = h_start + self.kernel_size
                for idxw in range(width_out):
                    w_start = idxw * self.stride
                    w_end = w_start + self.kernel_size
                    # 取出对应感受野窗口并展平
                    img2col[col_idx, :] = self.input_pad[idxn, :, h_start:h_end, w_start:w_end].reshape(-1)
                    col_idx += 1

        output = np.dot(img2col, weight_reshape) + self.bias  # (N * H_out * W_out, C_out)

        # 恢复为标准 NCHW 形状: (N, H_out, W_out, C_out) -> (N, C_out, H_out, W_out)
        self.output = output.reshape([N, height_out, width_out, self.channel_out]).transpose([0, 3, 1, 2])
        return self.output


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, version='fast'):
        """
        :param version: 'fast' 为全切片并行取最大值；'slow' 为传统多重循环版本
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.version = version

    def forward(self, input, version=None):
        ver = version if version is not None else self.version
        if ver == 'fast':
            return self.forward_fast(input)
        else:
            return self.forward_slow(input)

    def forward_slow(self, input):
        """传统四重循环版本"""
        self.input = input  # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])

        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        h_start = idxh * self.stride
                        w_start = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[idxn, idxc, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                        )
        return self.output

    def forward_fast(self, input):
        """高效向量化版本（滑动窗口切片并行 Reduce）"""
        self.input = input  # [N, C, H, W]
        N, C, H, W = self.input.shape
        height_out = (H - self.kernel_size) // self.stride + 1
        width_out = (W - self.kernel_size) // self.stride + 1

        # 通过对窗口内的每个相对坐标 (kh, kw) 提取全局切片并同时取 max，消除四重 Python 循环
        patches = [
            self.input[:, :, kh:kh + height_out * self.stride:self.stride, kw:kw + width_out * self.stride:self.stride]
            for kh in range(self.kernel_size)
            for kw in range(self.kernel_size)
        ]
        self.output = np.maximum.reduce(patches)
        return self.output

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output