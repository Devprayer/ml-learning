## 1.self-attention

​	自注意力机制的 **主要思想** 是序列中每一个位置都能关注到其他位置信息的方式，方法是 **为每个位置分配不同的权重，以整合各位置的信息。** 其运作方式是吃一整个信息，考察信息每个位置的权重。也可以理解为是找寻序列中不同位置的相关信息

<img src="..\image\5.jpg" alt="5" width='60%'/>

​	这个相关信息常使用`dot product`方法来计算，两个不同位置的输入向量经过 $W^q$ 和 $W^k$ 相乘再相加。也可以使用`additive`方式来计算，其计算方式为，计算两个输入向量的 $q$ 和 $k$ ，再将 $q$ 和 $k$ 串起来，经过一个`activation function`，再经过一个举证转换，得到最终的 $\alpha$ 。

<img src="..\image\6.jpg" alt="6" width='60%'/>

​	在数学角度，给定的**向量** 是列向量，其计算图为

<img src="..\image\9.jpg" alt="9" width='60%'/>

用矩阵来描述即

<img src="..\image\3.jpg" alt="2" width='60%'/>

给定数据：输入序列 $X=[x_1,x_2,...,x_n]$ ，其中 $x_i\in R^{d_{model}}$ ，具体的， $X$ 形如

$$
X=\begin{array}{lll}
	[[a_{11}&a_{12}&\cdots&a_{1n}]\\
	[a_{21}&a_{22}&\cdots&a_{2n}]]
	\end{array}
$$

所以，以下的相乘是以计算机中储存形式计算，转置即为数学公式形式。

1. 生成查询（Query）、键（Key）、值（Value）：

$$
Q = XW^Q\\
K = XW^K\\
V = XW^V
$$

其中， $W^Q,W^K,W^V \in R^{d_{model} \times d_k}$ 

2. 计算注意力分数

$$
Attention(Q,K,V)=softmax(\frac {Q K^T}{\sqrt {d_k}})V
$$

​	其中， $d_k$ 是 $K$ 矩阵维度。由于点积 $QK^T$ 的值会随着 $d_k$ 的增大而增大，从而导致`softmax`的值走向极端，梯度变得非常小或者不稳定，所以引入 $\frac{1}{\sqrt{d_k}}$ 来缩放，使得点积相对稳定。

通常设置为一个可以被总的 **embedding** 维度均分的数，例如，当多头为8时，embedding维度为512时， $d_k=d_v$ 的值为512/8。所以 $d_k$ 满足

$$
d_k=\frac{d_{model}}{head}
$$

## 2.multi-head attention

假设我们有输入 $X∈R^{n×d_{model}}$ ，多头注意力的流程如下：

1. 对于每个头 $i\in\{1,2...h\}$ ：


   $$
   Q_i=XW_i^Q,K_i=XW_i^K,V_i=XW_i^V
   $$


   其中， $W_i^Q,W_i^K,W_i^V\in R^{d_{model} \times d_k}$ 。

   <img src="..\image\13.jpg" alt="13" width='60%'/>

3. 每个头计算：

   
   $$
   head_i = Attention(Q,K,V)
   $$
   

5. 连接所有头

   
   $$
   MultiHead(X) = Concat(head_1,head_2,...,head_h)W^O
   $$

   
   其中， $W^O \in R^{hd_k \times d_{model}}$ 

   <img src="..\image\14.jpg" alt="14" width='60%'/>

## 3. 反向传播

$$
\mathcal{L} = ||Attention(X)-Y||^2
$$

以更新 $W^V$ 为例，先计算梯度

$$
\frac{\partial{\mathcal{L}}}{\part{W^Q}}=\frac{\partial{\mathcal{L}}}{\part{A}} \cdot \frac{\partial{A}}{\part{V}}\cdot \frac{\partial{V}}{\part{W^Q}}\\
\frac{\partial{\mathcal{L}}}{\part{A}} = A-Y\\
\frac{\partial{A}}{\part{V}} =softmax(\frac {Q K^T}{\sqrt {d_k}})\\
\frac{\partial{V}}{\part{W^Q}} = X \\
\therefore \frac{\partial{\mathcal{L}}}{\part{W^Q}}=X^T \cdot softmax(\frac {Q K^T}{\sqrt {d_k}})^T \cdot (A-Y)\\
W^V = W^V - \eta \cdot \frac{\partial{\mathcal{L}}}{\part{W^Q}}
$$

## 4.Positional Encoding

`transformer`不具备循环神经网络的按照时间步处理的方式，所以需要添加位置编码告知模型各向量的位置。具体数学公式如下：

$$
PE_{pos,2i}=sin(\frac{pos}{base^{2i/d_{model}}})\\
PE_{pos,2i+1}=cos(\frac{pos}{base^{2i/d_{model}}})
$$

偶数用`sin`，奇数用`cos`。其中，`pos`是位置编号，`i`是维度编码。具体流程如下：

1. 假设有 $d_{model}$ ，计算其对2整除 $n = d_{model} // 2-1$，$i \in \{0,2,4,...,n\}$ 。
2. 构造缩放因子`div_term`

| `i`  | 维度编号 | 对应维度   | 用的函数 | 频率因子（除以 $base^{2i / d_{model}}$） |
| ---- | -------- | ---------- | -------- | :--------------------------------------- |
| 0    | 0, 1     | dim0, dim1 | sin/cos  | base⁰ = 1                                |
| 1    | 2, 3     | dim2, dim3 | sin/cos  | base^(2/8) = base^0.25                   |
| 2    | 4, 5     | dim4, dim5 | sin/cos  | base^0.5                                 |
| 3    | 6, 7     | dim6, dim7 | sin/cos  | base^0.75                                |

3. 乘以位置，再用`sin`和`cos`运算

| pos  | i=0 (`sin`)       | i=1 (`cos`)        | i=2 (`sin`)            | i=3 (`cos`)          |
| ---- | ----------------- | ------------------ | ---------------------- | -------------------- |
| 0    | sin(0/1) = 0      | cos(0/1) = 1       | sin(0/100) ≈ 0         | cos(0/100) ≈ 1       |
| 1    | sin(1/1) ≈ 0.8415 | cos(1/1) ≈ 0.5403  | sin(1/100) ≈ 0.0099998 | cos(1/100) ≈ 0.99995 |
| 2    | sin(2/1) ≈ 0.9093 | cos(2/1) ≈ -0.4161 | sin(2/100) ≈ 0.0199986 | cos(2/100) ≈ 0.99980 |

4. 拼接成最终矩阵

   位置0: [ 0.0000, 1.0000, 0.0000, 1.0000 ]
   位置1: [ 0.8415, 0.5403, 0.0100, 0.9999 ]
   位置2: [ 0.9093, -0.4161, 0.0200, 0.9998 ]

5. 再与各位置向量相加

**base**的说明

| 基数           | 高频维度变化   | 低频维度变化 | 适合应用场景         |
| -------------- | -------------- | ------------ | -------------------- |
| 大（如 10000） | 慢（变化缓慢） | 很慢         | 长序列、稳定建模     |
| 中（如 1000）  | 中等           | 慢           | 中等长度序列、泛用   |
| 小（如 100）   | 快（剧烈变化） | 快           | 短序列、快速局部差异 |

## 5.掩码

### 5.1Padding

假设数据：

$$
s_1 = \left[
\begin{array}{lll}
1 & 1 & 1\\
2 & 3 & 5\\
\end{array}
\right]\\
s_2 = \left[
\begin{array}{lll}
1 & 1 & 1 & 8\\
2 & 3 & 5 & 10\\
\end{array}
\right]\\
$$

​	其中 $s_1,s_2$ 是两个序列，但是长度不一样（每一列是序列一个元素）。这时候需要padding将序列扩展到相同维度，使用batch进行训练。

分为两步

+ 先使用**0**对序列进行填充到相同长度

$$
s_1 = \left[
\begin{array}{lll}
1 & 1 & 1 & 0\\
2 & 3 & 5 & 0\\
\end{array}
\right]\\
s_2 = \left[
\begin{array}{lll}
1 & 1 & 1 & 8\\
2 & 3 & 5 & 10\\
\end{array}
\right]\\
$$

+ 再为每个序列生成对应用**0**填充的位置掩码，用来控制注意力的连接性。但是 **mask 的维度** 与 `embedding_dim` 无关，mask 只跟 **序列长度** 和 **batch 大小** 有关，形状为（batch，seq_len)。

$$
Padding Mask = \left[
\begin{array}{lll}
False & False & False & \color{red}{True}\\
False & False & False & False\\
\end{array}
\right]\\
$$

### 5.2 Causal mask

假设时序数据：

$$
s_1 = \left[
\begin{array}{lll}
1 & 1 & 1 & 0\\
2 & 3 & 5 & 0\\
\end{array}
\right]\\
$$

当不想第一个数据查看最后一个数据的信息（因为第一个数据发生时，最后一个没发生），这时候需要位置掩码。

其形状只与**序列长度**相关，（seq_len, seq_len)。例如上述序列长度为**4**，这时causal mask为

$$
\begin{array}{lll}
False & \color{red}{True} & \color{red}{True} & \color{red}{True}\\
False & False & \color{red}{True} & \color{red}{True}\\
False & False & False & \color{red}{True}\\
False & False & False & False\\
\end{array}
$$

`Look-Ahead Mask`的作用机制是

$$
Attention(Q,K,V)=softmax(\frac {Q K^T}{\sqrt {d_k}}+mask)V
$$

若是`mask`是`-inf`，经过`softmax`函数后位置权重为0，从而有效屏蔽该位置。

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

**例如**：假设序列长度=4，query长度为4：

$$
QK^T=\left[
\begin{array}{cc}
1 & 2 & 3 & 4\\
2 & 1 & 0 & 3\\
0 & 2 & 1 & 2\\
3 & 1 & 4 & 2
\end{array}
\right]
$$

Causal Mask为：

$$
Causal Mask = \left[
\begin{array}{cccc}
0 & \color{red}{-\infty} & \color{red}{-\infty} & \color{red}{-\infty}\\
0 & 0 & \color{red}{-\infty} & \color{red}{-\infty}\\
0 & 0 & 0 & \color{red}{-\infty}\\
0 & 0 & 0 & 0\\
\end{array}
\right]
$$

$$
QK^T + CausalMask=\left[
\begin{array}{cc}
1 & -\infty & -\infty & -\infty\\
2 & 1 & -\infty & -\infty\\
0 & 2 & 1 & -\infty\\
3 & 1 & 4 & 2
\end{array}
\right]
$$

### 5.3下一步处理

经过transformer的数据仍然具有padding位置，下一步的分类器也通常需要保留固定长度向量。但又不希望padding值影响到分类结果，所以需要做 **padding-aware 聚合 / 池化**，或者只取特定位置的向量。

+ （1）Masked Pooling

对有效**token**做池化，忽略padding token。

$$
\bar{y}=\frac{\sum_{i=1}^L y_i\cdot m_i}{\sum_{i=1}^Lm_i}
$$

其中， $m_i=1$ 表示有效位置， $m_i=0$ 表示padding

```python
import torch

output = torch.randn(2, 5, 128)  # Transformer 输出: [B, L, d_model]
padding_mask = torch.tensor([
    [False, False, False, True, True],
    [False, False, True, True, True]
])  # [B, L]

mask = (~padding_mask).unsqueeze(-1).float()  # [B, L, 1]，
#[
#[[1],[1],[0]],相当于对batch中每个序列的每个元素进行一个权重分配，有效位置为1，padding位置为0
#]
masked_output = output * mask

pooled_sum = masked_output.sum(dim=1)         # 对序列长度维度求和
mask_count = mask.sum(dim=1)                  # 有效 token 数量

pooled_output = pooled_sum / mask_count       # 平均

```



+ （2）**取特定位置的向量**，第一个 token 代表整个句子：`pooled_output = output[:, 0, :]`
+ （3）**Attention Pooling**

给定一个transformer的输出：

$$
H \in \mathbb{R}^{B\times L \times d_{model}}
$$

计算步骤：

1. **计算注意力权重**

$$
e_i = w^T \cdot tanh(WH_i+b)
$$

其中：

+  $H_i$ 为第 $i$ 个token的向量
+  $W,b,w$ 是可学习参数

2. 加padding mask

$$
\begin{equation}
e_i = \left\{
             \begin{array}{lr}
             e_i, &有效位置 \\
             -\infty, & padding位置\\
             \end{array}
\right.
\end{equation}
$$

2. **softmax**

$$
\alpha_i=\dfrac{exp(e_i)}{\sum_jexp(e_j)}加权就和
$$

3. **加权求和**

$$
v = \sum_{i=1}^L\alpha_iH_i
$$

最终 $v$ 就是池化后的句子向量，shape = `[B, d_model]`。

+ Transformer 输出：`output = [B, L, d_model]`

+ padding mask：`padding_mask = [B, L]`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_fc = nn.Linear(d_model, 1)  # 简单的线性层计算权重

    def forward(self, H, padding_mask=None):
        """
        H: [B, L, d_model]
        padding_mask: [B, L], True = padding
        """
        scores = self.attn_fc(H).squeeze(-1)  # [B, L]

        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))  # 屏蔽 padding

        attn_weights = F.softmax(scores, dim=1)  # [B, L]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, L, 1]

        pooled = torch.sum(attn_weights * H, dim=1)  # [B, d_model]
        return pooled, attn_weights

# 使用示例
B, L, d_model = 2, 5, 128
output = torch.randn(B, L, d_model)
padding_mask = torch.tensor([
    [False, False, True, True, True],
    [False, False, False, True, True]
])

attn_pool = AttentionPooling(d_model)
pooled_output, attn_weights = attn_pool(output, padding_mask)

print(pooled_output.shape)  # [B, d_model]
print(attn_weights.shape)   # [B, L, 1]

```



## 6.Transformer架构

<img src="..\image\transformers.png" width='60%'/>

