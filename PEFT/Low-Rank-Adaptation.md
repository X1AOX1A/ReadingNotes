
# Low-Rank Adaptation (LoRA)

## Adaptive / Sparse LoRA

- LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
  - 2024.03.26, arXiv, [pdf](https://arxiv.org/abs/2403.17919)
  - Motivation：在LoRA中，底层 （embedding, e.g. LLaMA2-7B, GPT2）和/或顶层 (lm head, e.g. GPT2) 占据了大多数权重更新，而其他自注意力层只占很小一部分。这表明不同层次在更新时的重要性不同
  - Method: 在优化过程中随机冻结（均匀概率）大部分中间层，对不同层进行重要性采样

- ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models
  - 2024.03.24, NAACL2024, [pdf](https://arxiv.org/abs/2403.16187)
  - Method: 1. AB-LoRA: 计算每个 rank 在验证集上的得分，2. ALoRA: 对得分最少 rank (单位矩阵) 进行 prune (1->0)，将 rank 分配给未 prune 的权重 (增加 rank)【对 AdaLoRA, AutoLoRA 的改进】

- AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models
  - 2024.03.20, arXiv, [pdf](https://arxiv.org/abs/2403.13269)
  - Method: Normal LoRA + feature transformation vector + adaptive freezing of LoRA

- AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning
  - 2024.03.14, arXiv, [pdf](https://arxiv.org/abs/2403.09113), [code](https://github.com/ruz048/AutoLoRA)
  - Method: 使用迭代训练的方法，在 train set 上训练 loraA, loraB，在 val set 上训练 selection variables (summation=1)，迭代直到收敛；再通过 threshold 过滤 selection variables 进行 prune，获得每个 layer 的 optimal rank

- SoRA: Sparse Low-rank Adaptation of Pre-trained Language Models
  - 2023.11.20, arXiv, [pdf](https://arxiv.org/abs/2311.11696)

- InRank: Incremental Low-Rank Learning
  - 2023.06.20, arXiv, [pdf](https://arxiv.org/abs/2306.11250)

- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
  - 2023.03.18, ICLR2023, [pdf](https://arxiv.org/pdf/2303.10512.pdf), [code](https://arxiv.org/pdf/2303.10512.pdf)
  - Method: 基于梯度计算 rank 的 sensitivity&uncertainty score，动态分配 / mask SVD 对角矩阵；使用损失函数正则化逼近 SVD 分解
  - Results：DeBERTaV3-base (NLU); BART-large (NLG) 上超越 LoRA；FC2 以及更高层占据更多 rank

## SVD LoRA

- PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models
  - 2024.04.03, arXiv, [pdf](https://arxiv.org/abs/2404.02948)
  - Method: SVD init LoRA, large singular part freeze, small singular part update

- SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression
  - 2024.03.12, arXiv, [pdf](https://arxiv.org/abs/2403.07378)

- ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models
  - 2023.12.10, arXiv, [pdf](https://arxiv.org/abs/2312.05821)

## Weight Decomposed / Projection

- CTRLorALTer: Conditional LoRAdapter for Efficient 0-Shot Control & Altering of T2I Models
  - 2024.05.13, arXiv, [pdf](https://arxiv.org/abs/2405.07913), [code](https://github.com/CompVis/LoRAdapter), [home](https://compvis.github.io/LoRAdapter/)
  - Method: 对 LoRA_A 进行 conditional transformation, 控制 stabe diffusion structure&style

- FourierFT: Parameter-Efficient Fine-Tuning with Discrete Fourier Transform
  - 2024.05.05, ICML2024, [pdf](https://arxiv.org/abs/2405.03003), [code](https://github.com/Chaos96/fourierft)
  - <details>
        <summary>Method: 使用 2D inverse discrete fourier transform (IDFT) 对 delta weight 进行逼近</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405132255433.png"/>
        </details>
  - Results：RoBERTa, GPT-2, LLaMA1,2, 7B, 13B 相同 performance 条件下，大幅降低训练参数

- Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning
  - 2024.03.12, arXiv, [pdf](https://arxiv.org/abs/2403.07440)
  - 特定任务的参数矩阵进行线性变换，以动态改变参数矩阵的空间几何结构，并生成新的矩阵特征模式（特征向量）

- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
  - 2024.03.06, arXiv, [pdf](https://arxiv.org/abs/2403.03507)
  - Findings: 对梯度进行 low-rank projection，从而降低全量 finetune 参数量

- DoRA: Weight-Decomposed Low-Rank Adaptation
  - 2024.02.14, arXiv, [pdf](https://arxiv.org/abs/2402.09353)

- FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference
  - 2024.01.08, arXiv, [pdf](https://arxiv.org/abs/2401.04044)
  - Findings: 把FFN分解成两个FFN来降低运算量


## Activate Layer / Sensitive Neurons

- Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model
  - 2024.03.18, arXiv, [pdf](https://arxiv.org/abs/2403.11621)
  - Method: 在 FT 模型上的 Sensitive Neurons 进行  Neuron-Level Fine-Tuning (NeFT)

- Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding
  - 2024.03.25, arXiv, [pdf](https://arxiv.org/abs/2404.16710)
  - Method：层dropout的dropout率随层数增加而增加 (LLaMA1-7B)，使用指数增长函数来设置每个层的dropout率
  - 重要性度量：generated token corresponds to the earliest layer in the model that predicted it

- Neurons in Large Language Models: Dead, N-gram, Positional
  - 2023.09.09, arXiv, [pdf](https://arxiv.org/abs/2309.04827)
  - Findings: 70% are never activated in the first half of OPT-60B


## Others

- Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices
  - 2024.03.22, arXiv, [pdf](https://arxiv.org/abs/2403.14958)
  - Method: 采用随机低秩(自适应秩)矩阵近似来更有效和准确地近似Adam的二阶矩

- ResLoRA: Identity Residual Mapping in Low-Rank Adaption
  - 2024.02.28, arXiv, [pdf](https://arxiv.org/abs/2402.18039)

- LoRA Meets Dropout under a Unified Framework
  - 2024.02.25, arXiv, [pdf](https://arxiv.org/abs/2403.00812)

- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA
  - 2024.02.24, arXiv, [pdf](https://arxiv.org/abs/2402.16902)

- LoRA+: Efficient Low Rank Adaptation of Large Models
  - 2024.02.19, arXiv, [pdf](https://arxiv.org/abs/2402.12354)
  - Findings: 对 lora_b 使用更大的学习率，可以提升1～2%性能和2倍速度

- ReLoRA: High-Rank Training Through Low-Rank Updates
  - 2023.07.11, arXiv, [pdf](https://arxiv.org/abs/2307.05695)

