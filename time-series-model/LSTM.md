LSTM是由循环神经网络**RNN，Recurrent Neural Network**发展而来。
**（RNN，Recurrent Neural Network）**
#### 1单个cell计算

​	假设有一个标量$z$，三个门控由$z_i,z_f,z_o$操控，神经元细胞中原本储存有记忆$c$，$g(\cdot),f(\cdot)$为`sigmod`函数。

由`input`门控和`forget`门控决定有多少记忆$c'$需要储存，再有output门控决定输出$h$有多少。
$$
\begin{align}
c'&=g(z)f(z_i)+cf(z_f)\\
h&=h(c')f(z_o)
\end{align}
$$

<center><img src="..\image\15.jpg" width="60%"/></center>

#### 2 架构



​	对于输入数据$\mathbf{x}^t \in \mathbb{R}^n $，上一时间步的记忆状态$\mathbf{c}^{t-1} \in \mathbb{R}^m$，其中$m$是隐藏层神经元细胞数目。将输入的数据$\mathbf{x}^t$经过四次线性转换成$\mathbf{z},\mathbf{z}^f,\mathbf{z}^i,\mathbf{z}^o \in \mathbb{R}^m$，其中$\mathbf{z}^*$的每一维度作为控制每一个神经元细胞的输入，分别控制细胞的输入、forget、input、output。假如$\mathbf{x}^t=[1,2,3],\mathbf{x}^{t+1}=[4,5,6]$，隐藏层有5个细胞。那么计算如下：
$$
\begin{align}
	\mathbf{z} &= W_z \mathbf{x}^t \\
	\mathbf{z}^f &= W_{z^f} \mathbf{x}^t \\
	\mathbf{z}^i &= W_{z^i} \mathbf{x}^t  \\
	\mathbf{z}^o &= W_{z^o} \mathbf{x}^t  \\
	i_t &=\sigma(W_{ii}x_t+W_{hi}+h_{t−1}+b_i)\\
	o_t &= \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o)\\
	g_t &= \tanh(W_{ig} x_t + W_{hg}h_{t-1} + b_g)\\
	c_t &= f_t * c_{t-1} + i_t * g_t\\
	h_t &= o_t * \tanh(c_t)
\end{align}
$$

<img src='..\image\16.jpg'  width="60%"/>

​	类似于一个“过滤器”，LSTM按照序列依次计算。如下

<img src='..\image\17.jpg'  width="60%"/>

一般使用的LSTM是将上一期的输出$\mathbf{h}^{t-1}$加到下一时期的输入上。

<img src='..\image\18.jpg'  width="60%"/>

扩展版本可将上一期的记忆$\mathbf{c}^{t-1}$也加入下一期的输入中。

<img src='..\image\19.jpg'  width="60%"/>

LSTM也可以叠加许多层，每一层输出作为上一层的输入。



<img src='..\image\20.jpg' width="60%"/>

#### 3代码

```python
nn.LSTM(input_size, hidden_size, num_layers, bidirectional,proj_size=0)
```

当`num_layers`大于1、`bidirectional=True`时，其`forward`函数中的输入如下：
$$
D = 2\ if\ bidirectional=True\ else\ 1\\
H_{out}= hidden\_size\ if\ proj\_size=0\ else\ proj\_size\\
H_{cell} = hidden\_size
$$


**input, (h_0, c_0)**

+ **input：**可以时没有`batch`的（L，F）的tensor，也可以是有`batch`的（B，L，F）的tensor。
+ **h_0：**形状为（D$\times$num_layers，Batch，$H_{out}$）或没有batch的（D$\times$num_layers，$H_{out}$）。

+ **c_0：**形状为（D$\times$num_layers，Batch，$H_{cell}$）或没有batch的（D$\times$num_layers，$H_{cell}$）

**output, (h_n, c_n)**

+ **output**返回形状为（L，Batch，D$\times H_{out}$）或batch在前或没有batch的tensor。
+ **h_n** 返回形状为（D$\times$num_layers，$H_{out}$），或有batch的tensor。注意，例如，num_layers=2，bidirectional=True时，**h_0**的0，1行为第一层的数据。
+ **c_n** 返回形状为（D$\times$num_layers，$H_{out}$）或有batch的tensor。

<img src="..\image\lstm.png" width="60%"/>

上图假设原序列有4维特征，设置的hidden_size为6。可以理解为，上述神经网络对序列进行“扫描”，并储存每一步的记忆。





