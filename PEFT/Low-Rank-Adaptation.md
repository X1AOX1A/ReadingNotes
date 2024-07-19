
# Low-Rank Adaptation (LoRA)

## Survey

- A Survey on LoRA of Large Language Models
  - 2024.07.08, arXiv, [pdf](https://arxiv.org/abs/2407.11046)

## Adaptive LoRA

- DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution
  - 2024.05.27, arXiv, [pdf](https://arxiv.org/abs/2405.17357), [code](https://github.com/mikumikumi0116/dora)
  - <details>
      <summary>Method: 将高秩LoRA分解为多个单秩LoRA组合，允许更细粒度的参数管理</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071026812.png"/>
    </details>

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

- InRank: Incremental Low-Rank Learning
  - 2023.06.20, arXiv, [pdf](https://arxiv.org/abs/2306.11250)

- AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
  - 2023.03.18, ICLR2023, [pdf](https://arxiv.org/pdf/2303.10512.pdf), [code](https://arxiv.org/pdf/2303.10512.pdf)
  - Method: 基于梯度计算 rank 的 sensitivity&uncertainty score，动态分配 / mask SVD 对角矩阵；使用损失函数正则化逼近 SVD 分解
  - Results：DeBERTaV3-base (NLU); BART-large (NLG) 上超越 LoRA；FC2 以及更高层占据更多 rank

## Sparse LoRA

- Sparse Matrix in Large Language Model Fine-tuning
  - 2024.05.24, arXiv, [pdf](https://arxiv.org/abs/2405.15525)
  - <details>
      <summary>Method: 直接对原参数矩阵中少量参数进行微调；V 比 QK更重要</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071019833.png"/>
    </details>

- FourierFT: Parameter-Efficient Fine-Tuning with Discrete Fourier Transform
  - 2024.05.05, ICML2024, [pdf](https://arxiv.org/abs/2405.03003), [code](https://github.com/Chaos96/fourierft)
  - <details>
        <summary>Method: 使用 2D inverse discrete fourier transform (IDFT) 对 delta weight 进行逼近</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405132255433.png"/>
        </details>
  - Results：RoBERTa, GPT-2, LLaMA1,2, 7B, 13B 相同 performance 条件下，大幅降低训练参数

- PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA
  - 2024.02.24, arXiv, [pdf](https://arxiv.org/abs/2402.16902)
  - <details>
      <summary>Method: 对 lora A,B 参数进行部分旋转广播共享，保留部分非共享 rank</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202405240040215.png"/>
    </details>

- SoRA: Sparse Low-rank Adaptation of Pre-trained Language Models
  - 2023.11.20, arXiv, [pdf](https://arxiv.org/abs/2311.11696)

- VeRA: Vector-based Random Matrix Adaptation
  - 2023.10.17, ICLR24, [pdf](https://arxiv.org/abs/2310.11454v1)
  - <details>
      <summary>Method: 固定共享随机矩阵 loraA, loraB，训练向量 b, d，大幅减少训练参数量</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202405232131931.png"/>
    </details>

## Weight Decomposed (SVD) / Projection / Transformation

- LoRA-GA: Low-Rank Adaptation with Gradient Approximation
  - 2024.07.12, arXiv, [pdf](https://arxiv.org/abs/2407.05000), [code](https://github.com/Outsider565/LoRA-GA)
  - <details>
      <summary>Method: 基于首次梯度的 SVD 分解对 loraA, loraB 进行初始化</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202407121358871.png"/>
    </details>

- FLoRA: Low-Rank Core Space for N-dimension
  - 2024.05.23, arXiv, [pdf](https://arxiv.org/abs/2405.14739), [code](https://github.com/SJTU-DeepVisionLab/FLoRA)
  - <details>
      <summary>Method: Tucker 分解</summary>
      <img src="https://picx.zhimg.com/70/v2-4122388a7a0d9d410843c8ad83cfe8b0_1440w.awebp?source=172ae18b&biz_tag=Post"/>
    </details>

- LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation
  - 2024.06.18, arXiv, [pdf](https://arxiv.org/abs/2406.12832), [code](https://github.com/arminazizi98/lamda)
  - <details>
      <summary>Method: SVD init + 只微调单位矩阵</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406191450549.png"/>
    </details>

- MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning
  - 2024.06.13， arXiv, [pdf](https://arxiv.org/abs/2406.09044)
  - <details>
      <summary>Method: 使用 SVD 初始化，只微调小部分 singular components</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406191454925.png"/>
    </details>
- SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors
  - 2024.05.30, arXiv, [pdf](https://arxiv.org/abs/2405.19597), [code](https://github.com/vijaylingam95/svft)
  - <details>
      <summary>Method: 只微调 SVD 对角矩阵</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071028939.png"/>
    </details>

- LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters
  - 2024.05.27, arXiv, [pdf](https://arxiv.org/abs/2405.17604), [code](https://github.com/MohammadrezaBanaei/LoRA-XS)
  - <details>
      <summary>Method: 只微调 SVD 对角矩阵</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071027436.png"/>
    </details>

- VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks
  - 2024.05.24, arXiv, [pdf](https://arxiv.org/abs/2405.15179)
  - <details>
      <summary>Method: 使用 vector bank 替代 loraA, loraB 进行参数选择与共享</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071023133.png"/>
    </details>

- MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning
  - 2024.05.20, arXiv, [pdf](https://arxiv.org/abs/2405.12130), [code](https://github.com/kongds/mora)
  - <details>
      <summary>Method: 1. 使用 high-rank 方阵替代 low-rank; 2. 使用非参数压缩&解压缩函数（linear time），包括截断维度、共享行列、重塑输入、旋转算子</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202405241157170.png"/>
    </details>

- CTRLorALTer: Conditional LoRAdapter for Efficient 0-Shot Control & Altering of T2I Models
  - 2024.05.13, arXiv, [pdf](https://arxiv.org/abs/2405.07913), [code](https://github.com/CompVis/LoRAdapter), [home](https://compvis.github.io/LoRAdapter/)
  - Method: 对 LoRA_A 进行 conditional transformation, 控制 stabe diffusion structure&style

- PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models
  - 2024.04.03, arXiv, [pdf](https://arxiv.org/abs/2404.02948)
  - Method: SVD init LoRA, large singular part freeze, small singular part update

- SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression
  - 2024.03.12, arXiv, [pdf](https://arxiv.org/abs/2403.07378)

- Matrix-Transformation Based Low-Rank Adaptation (MTLoRA): A Brain-Inspired Method for Parameter-Efficient Fine-Tuning
  - 2024.03.12, arXiv, [pdf](https://arxiv.org/abs/2403.07440)
  - <details>
      <summary>Method: 对特定任务的参数矩阵进行线性变换，以动态改变参数矩阵的空间几何结构，并生成新的矩阵特征模式（特征向量）</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202405241159207.png"/>
    </details>

- DoRA: Weight-Decomposed Low-Rank Adaptation
  - 2024.02.14, arXiv, [pdf](https://arxiv.org/abs/2402.09353)

- FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference
  - 2024.01.08, arXiv, [pdf](https://arxiv.org/abs/2401.04044)
  - Findings: 把FFN分解成两个FFN来降低运算量

- ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models
  - 2023.12.10, arXiv, [pdf](https://arxiv.org/abs/2312.05821)

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

## LoRA-MoE

- AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts
  - 2024.05.01, arXiv, [pdf](https://arxiv.org/abs/2405.00361)
  - <details>
        <summary>Method: 使用 gating function 和 threshold function 动态选取 lora expert topK 个数；LLaMA-2-7B 超越 MoLE</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405151325246.png"/>
        </details>

- HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning
  - 2024.04.30, arXiv, [pdf](https://arxiv.org/abs/2404.19245)
  - Motivation: 使用 t-sne 发现在不同子任务上 lora-A 偏向于学习共性知识，而 lora-B 偏向于学习特定知识
  - Method：使用 K-Means 自动决定 lora-B 数量，并共享 lora-A，加入类似 MoE 的可学习 router

- Snowflake Arctic: The Best LLM for Enterprise AI — Efficiently Intelligent, Truly Open
  - 2024.04.24, [Blog](https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/)
  - <details>
        <summary>Method: Dense+MoE</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405151327490.png"/>
        </details>

- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
  - 2024.04.22, arXiv, [pdf](https://arxiv.org/abs/2404.15159)
  - <details>
        <summary>Method: Add LoRA to MoE FFN</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405151328719.png"/>
        </details>

- Mixture-of-LoRAs: An Efficient Multitask Tuning for Large Language Models
  - 2024.03.06, arXiv, [pdf](https://arxiv.org/abs/2403.03432)

- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models
  - 2024.02.22, arXiv, [pdf](https://arxiv.org/abs/2402.14800)

- MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models
  - 2024.02.20, arXiv, [pdf](https://arxiv.org/abs/2402.12851)

- MoLA: Higher Layers Need More LoRA Experts
  - 2024.02.13, arXiv, [pdf](https://arxiv.org/abs/2402.08562), [code](https://github.com/gcyzsl/mola)
  - Findings: LLaMA2-7B (QKVUpDownGate) 上六个数据集均是深层次增加更多 LoRA Experts 更好

- MoLE: Mixture of LoRA Experts
  - 2024.01.16, arXiv, [pdf](https://openreview.net/forum?id=uWvKBCYh4S)
  - <details>
        <summary>Method: Add MoE-LoRA to Dense Model；代码参考 AdaMoLE</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405151330923.png"/>
        </details>

- LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin
  - 2023.12.15, arXiv, [pdf](https://arxiv.org/abs/2312.09979)

## LoRA Composition / Fusion

- Towards Modular LLMs by Building and Reusing a Library of LoRAs
  - 2024.05.18, arXiv, [pdf](https://arxiv.org/abs/2405.11157)
  - <details>
      <summary>Method: 使用 SVD 集成 LoRA</summary>
      <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/imgs/202406071017867.png"/>
    </details>

- LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report
  - 2024.04.29, arXiv, [pdf](https://arxiv.org/abs/2405.00732), [code](https://github.com/predibase/lora_bakeoff?tab=readme-ov-file)
  - TL;DR: 25 fine-tuned Mistral-7b models that consistently outperform base models by 70% and GPT-4 by 4-15%

- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
  - 2024.01.18, arXiv, [pdf](https://arxiv.org/abs/2307.13269)


## Understanding

- LoRA Learns Less and Forgets Less
  - 2024.05.15, arXiv, [pdf](https://arxiv.org/abs/2405.09673)
  - Findings: 1. LoRA 相比 FT 学的少也忘的少；2. FT 并没有学到很低的 low rank（但浅层与深层rank更低）；3. LoRA 学习率更大更敏感

- The Impact of Initialization on LoRA Finetuning Dynamics
  - 2024.06.12, arXiv, [pdf](https://arxiv.org/abs/2406.08447)
  - Findings: Initializing B to zero and A to random on average yields better performance compared to the counterpart.


## Others

- Adapprox: Adaptive Approximation in Adam Optimization via Randomized Low-Rank Matrices
  - 2024.03.22, arXiv, [pdf](https://arxiv.org/abs/2403.14958)
  - Method: 采用随机低秩(自适应秩)矩阵近似来更有效和准确地近似Adam的二阶矩

- GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
  - 2024.03.06, arXiv, [pdf](https://arxiv.org/abs/2403.03507)
  - Findings: 对梯度进行 low-rank projection，从而降低全量 finetune 参数量

- ResLoRA: Identity Residual Mapping in Low-Rank Adaption
  - 2024.02.28, arXiv, [pdf](https://arxiv.org/abs/2402.18039)

- LoRA Meets Dropout under a Unified Framework
  - 2024.02.25, arXiv, [pdf](https://arxiv.org/abs/2403.00812)

- LoRA+: Efficient Low Rank Adaptation of Large Models
  - 2024.02.19, arXiv, [pdf](https://arxiv.org/abs/2402.12354)
  - Findings: 对 lora_b 使用更大的学习率，可以提升1～2%性能和2倍速度

- ReLoRA: High-Rank Training Through Low-Rank Updates
  - 2023.07.11, arXiv, [pdf](https://arxiv.org/abs/2307.05695)
  - Method: 在 LoRA 基础上，固定周期将 loraA, loraB 吸收回 W，并重新初始化 loraA, loraB；以增加最终 deltaW 的 rank

