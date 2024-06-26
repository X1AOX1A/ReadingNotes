# Implementation

## Training

- [Accelerate Large Model Training using DeepSpeed](https://huggingface.co/blog/accelerate-deepspeed) (2022.06.28)
    - <details>
        <summary>Summary of zero stages</summary>
        1. Stage 1: Shards optimizer states across data parallel workers/GPUs<br>
        2. Stage 2: Shards optimizer states + gradients across data parallel workers/GPUs<br>
        3. Stage 3: Shards optimizer states + gradients + model parameters across data parallel workers/GPUs<br>
        4. Optimizer Offload: Offloads the gradients + optimizer states to CPU/Disk building on top of ZERO Stage 2<br>
        5. Param Offload: Offloads the model parameters to CPU/Disk building on top of ZERO Stage 3
        </details>
- [LLM.int8 - bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) (2022.08.17)
    - <details>
        <summary>Summary of dtypes (FP32, TF32, FP16, BF16, Int8)</summary>
        1. 混合精度(FP32&FP16) 降低显存，提升速度<br>
        2. bitsandbytes 量化(Int8)<br>
        3. LLM.int8 (效果与FP16相同，降低显存，但速度变慢 FP16&ltFP32&ltInt8)<br>
        4. Usage for inference
        </details>
- [How 🤗 Accelerate runs very large models thanks to PyTorch](https://huggingface.co/blog/accelerate-large-models)
    - Introduction of HF's 'meta_device' 'init_empty_weights', 'device map', 'offload_folder', 'offload_state_dict' ([Notion note](https://www.notion.so/x1a/How-Accelerate-runs-very-large-models-thanks-to-PyTorch-7e0aa33d81eb4d1ea1f175590f0c0960?pvs=4))
- [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft) (2023.02.10): HF peft-lora usage
- [Accelerate Large Model Training using PyTorch Fully Sharded Data Parallel](https://huggingface.co/blog/pytorch-fsdp) (2023.05.02)
    - Introduction of ZeRO stages and 3D parallelisms ([Notion note](https://www.notion.so/x1a/Accelerate-Large-Model-Training-using-PyTorch-Fully-Sharded-Data-Parallel-fdaaa6e7e9174f77805942dfdafbed1d?pvs=4))
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) (2023.05.24):
    - <details>
        <summary>Summary: NF4, BF16</summary>
        1. 4bit inference & fine-tune (QLoRA) usage<br>
        2. QLoRA Introduction (NF4 storage & BF16 compute)<br>
        3. 效果与fp16相同，降低显存
        </details>
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) (2023.07.18)
- [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl) (2023.08.08): HF DPO Usage
- [Making LLMs lighter with AutoGPTQ and transformers](https://huggingface.co/blog/gptq-integration) (2023.09.18)
    - <details>
        <summary>Summary of GPTQ (int4 storage, & fp16 activation (W4A16))</summary>
        1. 效果与fp16相同，降低显存，但速度变慢(fp16&ltGPTQ(x1.5)&ltbitsandbytes(x2)) <br>
        2. Usage for GPTQ model<br>
        3. 使用Optimum 量化模型<br>
        4. Fine-tune quantized models with PEFT<br>
      </details>
- [Fine-tuning Llama 2 70B using PyTorch FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp) (2023.09.15):
    - <details>
        <summary>Summary of SHARDED_STATE_DICT and FlashAttention</summary>
        1. 使用 SHARDED_STATE_DICT 解决多卡模型保存缓慢<br>
        2. 使用 FlashAttention 提高训练速度&降低 RAM 使用<br>
        3. [Notion Note](https://www.notion.so/x1a/Fine-tuning-Llama-2-70B-using-PyTorch-FSDP-0c433dbfd933484db10616c35d19d1fe?pvs=4)
      </details>
- [Overview of natively supported quantization schemes in 🤗 Transformers](https://huggingface.co/blog/overview-quantization-transformers) (2023.12.09)
    - Comparison of bitsandbytes-QLoRA and auto-gptq (GPTQ-4bit)
- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065) (2024.01.01)
- [浅谈后向传递的计算量大约是前向传递的两倍](https://zhuanlan.zhihu.com/p/675517271)(2024.01.01)

## Evaluation

- [LLM Benchmarks: MMLU, HellaSwag, BBH, and Beyond](https://www.confident-ai.com/blog/llm-benchmarks-mmlu-hellaswag-and-beyond) (2024.03.16)
    - <details>
        <summary>A comparision of different LLM benchmarks</summary>
        <img src="https://cdn.prod.website-files.com/64bd90bdba579d6cce245aec/65f9717426dad046975c2dba_benchmarks.png" align="middle" />
        </details>


## Model Explanation

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) by Harvard NLP (2018.04.03) ([Code](https://github.com/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb)) ([Updated version](http://nlp.seas.harvard.edu/annotated-transformer/))
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) (2023.12.11)
- [从零实现一个MOE（专家混合模型）](https://zhuanlan.zhihu.com/p/701777558) (2024.06.05)

## Tricks

- [llm-action](https://github.com/liguodongiot/llm-action): 大模型相关技术原理以及实战经验
- [llm-resource](https://github.com/liguodongiot/llm-resource): LLM全栈优质资源汇总
- [超大模型加载转换Trick](https://zhuanlan.zhihu.com/p/698950172) (2024.05.21)

## Links

Personal Blogs:
- [Yao Fu's Blog](https://www.notion.so/yaofu/Yao-Fu-s-Blog-b536c3d6912149a395931f1e871370db?pvs=4)
- [Lilian Weng's Blog](https://lilianweng.github.io)
- [Sebastian Ruder's Blog](https://www.ruder.io)
- [Jay Alammar's Blog](https://jalammar.github.io): Illustration of Models
- [Yann LeCun's Blog](http://yann.lecun.com/ex/index.html)
- [Andrej Karpathy's Blog](http://karpathy.github.io)
- [Eugene Yan's Blog](https://eugeneyan.com/writing/)
- [Chip Huyen's Blog](https://huyenchip.com/blog/)
- [pacman100's Blog](https://github.com/pacman100#%EF%B8%8F-blog-posts-)

Company Blogs:
- [HuggingFace Blog](https://huggingface.co/blog?p=1)
- [OpenAI Blog](https://openai.com/blog)
- [DeepMind Blog](https://deepmind.com/blog)
- [Google AI Blog](https://ai.googleblog.com)
- [Facebook AI Blog](https://ai.facebook.com/blog)
- [Microsoft AI Blog](https://blogs.microsoft.com/ai)
- [Apple Machine Learning Blog](https://machinelearning.apple.com)
