# RWKV-LM-RLHF
<p align='center'>
<image src="kotori.webp" width=15%/>
</p>

<div align="center"> 
A repository exploring the possibilities for deeper fine-tuning of RWKV.
</div>
<div align="center"> 
Finally, I started writing the Wiki🙂
</div>
<div align="center">
(https://github.com/OpenMOSE/RWKV-LM-RLHF/wiki)
</div>

## Support model architecture
   - RWKV v6 "Finch" (full feature support)
   - RWKV v7 "Goose" (full feature support)
   - RWKV v7 "Goose" + Mixture of LoRA Experts (SFT Support)
   - ARWKV (GQA + SwiGLU -> RWKV v7 TimeMix + SwiGLU) (SFT support)
   - PRWKV (GQA + SwiGLU -> RWKV cxa076 TimeMix + SwiGLU) (SFT, Infctx, DPO, ORPO)

## Key Features

- **RLHF Training with ORPO(Odds Ratio Preference Optimization) v6,v7**: 

  ORPO is that it allows for simultaneous SFT and aligning. By adjusting the orpo_alpha value, you can control the ratio between SFT and aligning.
   - Uses odds ratios instead of probability ratios to measure policy changes

   - Can often achieve better performance with fewer training steps

   - Designed to mitigate common issues in RLHF like reward hacking and overoptimization

- **RLHF Training with DPO(Direct Preference Optimization) v6,v7**: 

  Direct Preference Optimization (DPO) is an advanced training method for Large Language Models that aims to align model outputs with human preferences more effectively than traditional methods.
   - Directly optimizes for human preferences without the need for a separate reward model

   - Simplifies the training pipeline compared to other preference learning methods

   - More stable training process compared to RL-based methods

- **Infinite Context Compression Distillation Training v6,v7**:
  - This approach involves simultaneously training a student model using compressed logits data pre-acquired from a larger parameter model (for example, 14B) as soft labels, and the dataset as hard labels.

  - Soft label learning can significantly reduce the risk of overfitting.

  - It allows for efficient transfer of specific tasks to smaller models.
- **Infinite Context Masked SFT with Smoothing Loss v6,v7,ARWKV**:
  - By incorporating smoothing into the loss calculation, the transfer probability can be increased.

  - Dynamic masking allows for efficient loss reduction during multi-batch processing.

  - With an RTX4090, a 14B parameter model can be trained with 65k contexts.

## Peft Training Backend

- **Bone (Block Affine Transformation)**: 
   - New training method proposed by @Jl-er
   - Achieve faster convergence and better fit to data.
   - No complex initialization is required, and fast convergence and better fit to data can be achieved.
- **DoRA (Weight-Decomposed Low-Rank Adaptation) NEW!**: 
   - DoRA decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters.
   - Same training speed compared LoRA
- **LoRA**: 
   - You all already know this, right? :)
- **Quantization**:
   - FP8: Fastest. Native matmul. 
   - FP6: by TorchAO Matmul slightly faster than bitsandbytes
   - Int8: 8-bit quantization with Bitsandbytes. 16-bit Matmul
   - NF4: 4-bit quantization with Bitsandbytes. 16-bit Matmul
   - What is next?

> Rank can be set variably for each layer. see layer_profile




## System Requirements
   - CPU RAM >= 32GB (128GB Recommend for stress free)
   - Cuda or Rocm GPU.(NVIDIA RTX3090,4090, AMD MI100)
   - CUDA 12.4+, Rocm 6.1.2+
   - Python 3.12+
   - Pytorch 2.5+
   - Bitsandbytes (in MI100 with Rocm6.1.3, need build)
   - some case need (conda install libstdcxx -c conda-forge --override-channels) for building cuda kernel

## How to Use
   - [Odds Ratio Preference Optimization document](https://github.com/OpenMOSE/RWKV-LM-RLHF/wiki/Odds-Ratio-Preference-Optimization)
   - [Direct Preference Optimization document](https://github.com/OpenMOSE/RWKV-LM-RLHF/wiki/Direct-Preference-Optimization)
   - [Distillation document](main/example/Distillation/readme.md)
   - [SFT document](https://github.com/OpenMOSE/RWKV-LM-RLHF/wiki/Layer-Profile)
   - [Layer Profile](main/layerprofile/readme.md)



## Todo(RLHF)
   - 1. Iterative DPO, ORPO -> Suspend. 
   - 2. Reinforcement++ -> working :)

## Todo(Quantization)
   - 1. bnb 8bit optimizer. done
   - 2. ToachAO's FP6 Matmul testing

## Dream
   - 1. Realtime Training(with Reverse State Distillation)



# And Thanks to:
   - RWKV-LM @BlinkDL
   - RWKV-LM-RLHF-DPO @Triang-jyed-driung
   - RWKV-PEFT @Jl-er
   - Flash-Linear-Attention @fla-org
   - SmerkyG
   - Orpo @xfactlab
   - All RWKV Community Members :)




# License
same with RWKV-LM

Apache 2.0


@ 2025 OpenMOSE
