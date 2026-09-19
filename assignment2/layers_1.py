import numpy as np

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        # kaiming initialization
        self.weight = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / self.num_input), size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
        self.input = input
        # 全连接层前向传播公式 (2.3): Y = XW + b
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        # 全连接层反向传播公式 (2.4)
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0, keepdims=True) # 对应维度 1*p 的全1向量乘法
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input):
        self.input = input
        # ReLU前向传播公式 (2.5)
        output = np.maximum(0, self.input)
        return output

    def backward(self, top_diff):
        # ReLU反向传播公式 (2.6): 输入小于0的位置梯度为0
        bottom_diff = top_diff * (self.input > 0)
        return bottom_diff


class SoftmaxLossLayer(object):
    def forward(self, input):
        # 为了数值稳定性，先减去最大值 (公式 2.11)
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        # 将标签转为 one-hot 向量
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        # 交叉熵损失公式 (2.12)
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        # Softmax反向传播公式 (2.13)
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff