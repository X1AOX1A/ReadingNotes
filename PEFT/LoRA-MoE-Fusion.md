# LoRA-MoE / LoRA Fusion

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

- LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
  - 2024.01.18, arXiv, [pdf](https://arxiv.org/abs/2307.13269)

- MoLE: Mixture of LoRA Experts
  - 2024.01.16, arXiv, [pdf](https://openreview.net/forum?id=uWvKBCYh4S)
  - <details>
        <summary>Method: Add MoE-LoRA to Dense Model；代码参考 AdaMoLE</summary>
        <img src="https://x1a-alioss.oss-cn-shenzhen.aliyuncs.com/SnippetsLab/202405151330923.png"/>
        </details>

- LoRAMoE: Alleviate World Knowledge Forgetting in Large Language Models via MoE-Style Plugin
  - 2023.12.15, arXiv, [pdf](https://arxiv.org/abs/2312.09979)