import numpy as np
import scipy.special as sps


#Constructing a three-layer neural network model
class NeuralNetwork(object):
    #初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes  #输入节点数量
        self.hnodes = hiddennodes  #隐藏节点数量
        self.onodes = outputnodes  #输出节点数量
        self.lr = learningrate  #学习率
        #按照正态分布随机选取初始矩阵
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        #define sigmod function
        self.activation_function = lambda x: sps.expit(x)

    #训练神经网络，参数为：输入列表，目标列表
    def train(self, input_list, targer_list):
        inputs = np.array(input_list, ndmin=2).T  #输入值
        targers = np.array(targer_list, ndmin=2).T  #目标值

        hidden_inputs = np.dot(self.wih, inputs)  #计算隐藏层的输入层
        hidden_outputs = self.activation_function(hidden_inputs)  #计算隐藏层的输出层

        final_inputs = np.dot(self.who, hidden_outputs)  #计算输出层的输入数据
        final_outputs = self.activation_function(final_inputs)  #计算输出数据
        output_errors = targers - final_outputs  #计算输出误差矩阵

        hidden_errors = np.dot(self.who.T, output_errors)  #将误差按照权重反向传播到隐藏误差
        #迭代矩阵，方法为降梯度法
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    #使用训练好的模型，参数为输入列表
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outpus = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outpus)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784  #输入层节点数
output_nodes = 10  #输出层节点数
hidden_nodes = 200  #隐藏层节点数
learning_rate = 0.1  #学习率
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
