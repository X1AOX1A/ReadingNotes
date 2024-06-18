# Implementation

## Training

- [Accelerate Large Model Training using DeepSpeed](https://huggingface.co/blog/accelerate-deepspeed) (2022.06.28)
    - Stage 1 : Shards optimizer states across data parallel workers/GPUs
    - Stage 2 : Shards optimizer states + gradients across data parallel workers/GPUs
    - Stage 3: Shards optimizer states + gradients + model parameters across data parallel workers/GPUs
    - Optimizer Offload: Offloads the gradients + optimizer states to CPU/Disk building on top of ZERO Stage 2
    - Param Offload: Offloads the model parameters to CPU/Disk building on top of ZERO Stage 3



## From Scratch

- [从零实现一个MOE（专家混合模型）](https://zhuanlan.zhihu.com/p/701777558) (2024.06.05)

## Tricks

- [超大模型加载转换Trick](https://zhuanlan.zhihu.com/p/698950172) (2024.05.21)