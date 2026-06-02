# Transformer 架构详解

## 引言

Transformer 是 Google 在 2017 年提出的革命性神经网络架构，论文《Attention Is All You Need》彻底改变了自然语言处理领域。Transformer 完全基于注意力机制，摒弃了传统的 RNN 和 CNN 结构，实现了更高的并行化训练效率。

## 核心组件

### 1. 自注意力机制（Self-Attention）

自注意力机制是 Transformer 的核心，它允许模型在处理序列时关注输入序列的不同位置。

**计算过程：**
1. 将输入向量转换为三个矩阵：Query（Q）、Key（K）、Value（V）
2. 计算注意力分数：Attention(Q, K, V) = softmax(QK^T / √d_k)V

**优势：**
- 能够捕捉长距离依赖关系
- 计算可以高度并行化
- 每个位置都可以直接关注其他所有位置

### 2. 多头注意力（Multi-Head Attention）

使用多组不同的 Q、K、V 投影矩阵，让模型同时关注不同子空间的信息：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. 位置编码（Positional Encoding）

由于 Transformer 没有循环或卷积结构，需要显式注入位置信息：

**正弦位置编码：**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 4. 前馈神经网络（Feed-Forward Network）

每个编码器和解码器层包含一个全连接前馈网络：
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 5. 层归一化（Layer Normalization）和残差连接

- **残差连接**：缓解梯度消失问题
- **层归一化**：稳定训练过程

## 编码器-解码器结构

### 编码器（Encoder）
- 由 N 个相同的层堆叠而成
- 每层包含两个子层：多头注意力 + 前馈网络
- 使用残差连接和层归一化

### 解码器（Decoder）
- 同样由 N 个相同的层堆叠而成
- 每层包含三个子层：
  1. 带掩码的多头自注意力（防止看到未来信息）
  2. 编码器-解码器注意力
  3. 前馈网络

## Transformer 的变体

### 1. BERT（Bidirectional Encoder Representations from Transformers）
- 仅使用 Transformer 的编码器部分
- 双向上下文理解
- 预训练任务：MLM（掩码语言模型）+ NSP（下一句预测）

### 2. GPT（Generative Pre-trained Transformer）
- 仅使用 Transformer 的解码器部分
- 自回归语言模型
- 适合文本生成任务

### 3. T5（Text-to-Text Transfer Transformer）
- 编码器-解码器完整结构
- 将所有 NLP 任务统一为文本到文本的格式

## 优势与局限

**优势：**
- 并行计算能力强
- 长距离依赖建模效果好
- 可扩展性强（模型规模可大幅增加）

**局限：**
- 计算复杂度高（O(n²) 的注意力计算）
- 内存需求大
- 对长序列处理效率下降
