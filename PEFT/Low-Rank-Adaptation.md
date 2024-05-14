
# Low-Rank Adaptation (LoRA)

## Papers

- FourierFT: Parameter-Efficient Fine-Tuning with Discrete Fourier Transform
    - 2024.05.05, ICML2024, [pdf](https://arxiv.org/abs/2405.03003), [code](https://github.com/Chaos96/fourierft)
    - <details>
        <summary>Method: 使用 2D inverse discrete fourier transform (IDFT) 对 delta weight 进行逼近</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405132255433.png"/>
        </details>
    - Results：RoBERTa, GPT-2, LLaMA1,2, 7B, 13B 相同 performance 条件下，大幅降低训练参数

- HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning
    - 2024.04.30, arXiv, [pdf](https://arxiv.org/abs/2404.19245)
    - Motivation: 使用 t-sne 发现在不同子任务上 lora-A 偏向于学习共性知识，而 lora-B 偏向于学习特定知识
    - Method：使用 K-Means 自动决定 lora-B 数量，并共享 lora-A，加入类似 MoE 的可学习 router

- LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning
    - 2024.03.26, arXiv, [pdf](https://arxiv.org/abs/2403.17919)
    - Motivation：在LoRA中，底层 （embedding, e.g. LLaMA2-7B, GPT2）和/或顶层 (lm head, e.g. GPT2) 占据了大多数权重更新，而其他自注意力层只占很小一部分。这表明不同层次在更新时的重要性不同
    - Method: 在优化过程中随机冻结（均匀概率）大部分中间层，对不同层进行重要性采样

- ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models
    - 2024.03.24, NAACL2024, [pdf](https://arxiv.org/abs/2403.16187)
    - Method: 1. AB-LoRA: 计算每个 rank 在验证集上的得分，2. ALoRA: 对得分最少 rank (单位矩阵) 进行 prune (1->0)，将 rank 分配给未 prune 的权重 (增加 rank)【对 AdaLoRA, AutoLoRA 的改进】

- Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices
    - 2024.03.22, arXiv, [pdf](https://arxiv.org/abs/2403.14958)
    - Method: 采用随机低秩(自适应秩)矩阵近似来更有效和准确地近似Adam的二阶矩

- Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model
    - 2024.03.18, arXiv, [pdf](https://arxiv.org/abs/2403.11621)
  - Method: 在 FT 模型上的 Sensitive Neurons 进行  Neuron-Level Fine-Tuning (NeFT)

- AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning
    - 2024.03.14, arXiv, [pdf](https://arxiv.org/abs/2403.09113), [code](https://github.com/ruz048/AutoLoRA)
    - Method: 使用迭代训练的方法，在 train set 上训练 loraA, loraB，在 val set 上训练 selection variables (summation=1)，迭代直到收敛；再通过 threshold 过滤 selection variables 进行 prune，获得每个 layer 的 optimal rank

- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
    - 2023.03.18, ICLR2023, [pdf](https://arxiv.org/pdf/2303.10512.pdf), [code](https://arxiv.org/pdf/2303.10512.pdf)
    - Method: 基于梯度计算 rank 的 sensitivity&uncertainty score，动态分配 / mask SVD 对角矩阵；使用损失函数正则化逼近 SVD 分解
    - Results：DeBERTaV3-base (NLU); BART-large (NLG) 上超越 LoRA；FC2 以及更高层占据更多 rank