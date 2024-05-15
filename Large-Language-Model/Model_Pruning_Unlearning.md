# Model Pruning / Unlearning

## Papers

- Pruning as a Domain-specific LLM Extractor
  - 2024.05.10, NAACL2024, [pdf](https://arxiv.org/abs/2405.06275), [code](https://github.com/psunlpgroup/D-Pruner)
  - Method: 基于loss，保持通用能力的同时 prune 模型参数提升 domain specific 能力
  - Results: D-PRUNER + LoRA 效果超越其他 prune+LoRA baseline