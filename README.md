# lesson1
1. ReLU的常见变体：
Leaky ReLU (LReLU):
公式：f(x) = max(αx, x)，其中 α 是一个很小的正数（例如 0.01）。
解决了ReLU在负数区域梯度为零（“神经元死亡”）的问题，为负数输入提供了一个小的、非零的梯度。
Parametric ReLU (PReLU):
公式与Leaky ReLU类似：f(x) = max(αx, x)。
关键区别：斜率参数 α 不是固定的，而是作为模型参数之一，在训练过程中与其他权重一起学习。这使得激活函数能自适应数据。
Exponential Linear Unit (ELU):
当 x >= 0 时，f(x) = x
当 x < 0 时，f(x) = α(exp(x) - 1)
优点：继承了Leaky ReLU避免神经元死亡的特性，同时其输出均值更接近零，有助于加速收敛，对噪声更有鲁棒性。
Scaled Exponential Linear Unit (SELU):
是ELU的一个经过特殊缩放的版本，可以在特定的初始条件下（如lecun_normal初始化）使网络具有自归一化的特性，使各层输出的均值和方差在训练中保持稳定。
Swish:
公式：f(x) = x * sigmoid(βx) （β可以是常数或可学习参数）。
由Google提出，在部分任务上表现优于ReLU，曲线平滑且非单调。
2.其他常见的激活函数：
Softmax： 常用于多分类神经网络的输出层。它将一个实数向量映射为一个概率分布向量，所有输出之和为1。
Linear / Identity： f(x) = x。主要用于回归任务的输出层。
3.反向传播代码
import numpy as np

# -------------------------- 1. 定义激活函数及导数 --------------------------
def sigmoid(x):
    """Sigmoid激活函数，防止exp溢出：x减去最大值"""
    x = np.clip(x, -500, 500)  # 避免e^-500下溢或e^500上溢
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    """Sigmoid导数：sigmoid(x) * (1 - sigmoid(x))，x为sigmoid的输入"""
    s = sigmoid(x)
    return s * (1 - s)

def binary_cross_entropy(y_true, y_pred):
    """二分类交叉熵损失，防止log(0)"""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# -------------------------- 2. 定义2层神经网络反向传播类 --------------------------
class TwoLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化权重和偏置（Xavier初始化，避免梯度消失/爆炸）
        :param input_dim: 输入维度
        :param hidden_dim: 隐藏层神经元数
        :param output_dim: 输出维度（二分类为1）
        """
        # 隐藏层权重：input_dim → hidden_dim，偏置：hidden_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        # 输出层权重：hidden_dim → output_dim，偏置：output_dim
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        """前向传播，保存中间值（用于反向传播）"""
        self.X = X  # 输入数据 (n_samples, input_dim)
        self.z1 = np.dot(X, self.W1) + self.b1  # 隐藏层线性部分 (n_samples, hidden_dim)
        self.a1 = sigmoid(self.z1)  # 隐藏层激活输出 (n_samples, hidden_dim)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 输出层线性部分 (n_samples, output_dim)
        self.y_pred = sigmoid(self.z2)  # 输出层预测值 (n_samples, output_dim)
        return self.y_pred

    def backward(self, X, y_true, learning_rate=0.01):
        """
        反向传播计算梯度，并更新参数
        :param X: 输入数据 (n_samples, input_dim)
        :param y_true: 真实标签 (n_samples, output_dim)
        :param learning_rate: 学习率
        :return: loss: 当前批次的交叉熵损失
        """
        n_samples = y_true.shape[0]
        # 前向传播得到预测值（传入X，确保self.X被初始化）
        y_pred = self.forward(X)
        # 计算损失
        loss = binary_cross_entropy(y_true, y_pred)

        # -------------------------- 反向传播核心：链式求导 --------------------------
        # 步骤1：计算输出层的误差项（dL/dz2）
        # 二分类交叉熵+Sigmoid的联合导数：y_pred - y_true（简化版，无需单独计算sigmoid导数）
        dz2 = self.y_pred - y_true  # (n_samples, output_dim)
        # 步骤2：计算输出层权重和偏置的梯度
        dW2 = (1 / n_samples) * np.dot(self.a1.T, dz2)  # (hidden_dim, output_dim)
        db2 = (1 / n_samples) * np.sum(dz2, axis=0, keepdims=True)  # (1, output_dim)

        # 步骤3：计算隐藏层的误差项（dL/dz1），链式法则：dz2 → W2 → a1 → z1
        da1 = np.dot(dz2, self.W2.T)  # (n_samples, hidden_dim)
        dz1 = da1 * sigmoid_deriv(self.z1)  # (n_samples, hidden_dim)
        # 步骤4：计算隐藏层权重和偏置的梯度
        dW1 = (1 / n_samples) * np.dot(self.X.T, dz1)  # (input_dim, hidden_dim)
        db1 = (1 / n_samples) * np.sum(dz1, axis=0, keepdims=True)  # (1, hidden_dim)

        # -------------------------- 梯度更新（SGD） --------------------------
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return loss

# -------------------------- 3. 测试反向传播 --------------------------
if __name__ == "__main__":
    # 生成模拟数据：输入维度2，样本数100，二分类标签（0/1）
    X = np.random.randn(100, 2)  # 输入 (100, 2)
    y_true = np.random.randint(0, 2, (100, 1))  # 真实标签 (100, 1)

    # 初始化网络：输入2，隐藏层8，输出1
    model = TwoLayerNN(input_dim=2, hidden_dim=8, output_dim=1)
    # 训练1000轮，观察损失下降（验证反向传播有效）
    epochs = 1000
    learning_rate = 0.5
    for epoch in range(epochs):
        # 调用backward时传入X（关键修复点）
        loss = model.backward(X, y_true, learning_rate)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # 测试最终预测
    y_pred = model.forward(X)
    y_pred_label = (y_pred > 0.5).astype(int)
    acc = np.mean(y_pred_label == y_true)
    print(f"\n训练后准确率：{acc:.2f}")
