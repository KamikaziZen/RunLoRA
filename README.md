# 🏃 RunLoRA:  Framework for optimized LoRA implementations
## Based on paper: [Run LoRA Run: Faster and Lighter LoRA Implementations](https://arxiv.org/abs/2312.03415)

> LoRA is a technique that reduces the number of trainable parameters in a neural network by introducing low-rank adapters to linear layers. This technique is used both for fine-tuning (LoRA, QLoRA) and full train (ReLoRA). This paper presents the RunLoRA framework for efficient implementations of LoRA that significantly improves the speed of neural network training and fine-tuning using low-rank adapters. The proposed implementation optimizes the computation of LoRA operations based on dimensions of corresponding linear layer, layer input dimensions and lora rank by choosing best forward and backward computation graph based on FLOPs and time estimations, resulting in faster training without sacrificing accuracy. The experimental results show up to 17% speedup on Llama family of models. 

## Run Experiments

```
python3 experiments/model_exp.py -m configs/llama_60m.json \
        --target_modules q_proj v_proj k_proj o_proj up_proj down_proj gate_proj \
        --criterions flops --n_batch 10 --min_run_time 20 -r 8 --dtype 'bf16'
```

```
usage: model_exp.py [-h] -m MODEL_NAME_OR_CONFIG [--n_batch N_BATCH]
                    [-r LORA_R] [-a LORA_ALPHA] [-d LORA_DROPOUT]
                    [--dtype DTYPE]
                    [--target_modules TARGET_MODULES [TARGET_MODULES ...]]
                    [--criterions CRITERIONS [CRITERIONS ...]]
                    [--min_run_time MIN_RUN_TIME] [-o OUT]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_NAME_OR_CONFIG, --model_name_or_config MODEL_NAME_OR_CONFIG
                        path to config file or name of model available in
                        transformers hub
  --n_batch N_BATCH     batch size
  -r LORA_R, --lora_r LORA_R
                        rank of LoRA adapter
  -a LORA_ALPHA, --lora_alpha LORA_ALPHA
                        LoRA scaling factor
  -d LORA_DROPOUT, --lora_dropout LORA_DROPOUT
                        dropout applied to LoRA adapter input
  --dtype DTYPE         dtype of parameters and activations
  --target_modules TARGET_MODULES [TARGET_MODULES ...]
                        list of modules eligible for LoRA adapters
  --criterions CRITERIONS [CRITERIONS ...]
                        criterions for best forward-backwardpair estimation
  --min_run_time MIN_RUN_TIME
                        min time in seconds for running consecutiveexperiments
                        in mean runtime estimation
  -o OUT, --out OUT     prefix of output file
```

## Results

| Implementation  | Mean F-B loop, ms | Memory for F-B loop, MB | Speedup, \% | Memory Saved, MB |
|  :---:  |  :---:  |  :---:  |  :---:  |  :---:  |
|||  Llama 60m b=64 r=8 |
| RunLoRA | 244.08 |25774.76 | 13.62 | 55.97 |
| PEFT | 282.56 | 25830.73 | - | - |
|||  Llama 130m b=64 r=8 |
RunLoRA | 486.23 | 42853.7 | 14.86 | 84.0
PEFT | 571.09 | 42937.7 | - | -
|||  Llama 250m b=60 r=32 |
RunLoRA | 938.92 | 74835.38 | 15.9 | 629.85
PEFT | 1116.39 | 75465.23 | - | -
|||  Llama 250m b=58 r=64 |
RunLoRA | 909.98 | 72435.24 | 16.54 | 1198.5
PEFT | 1090.27 | 73633.74 | - | -
|||  Llama 250m b=58 r=128 |
RunLoRA | 926.51 | 72433.74 | 16.93 | 2434.25
PEFT | 1115.37 | 74867.99 | - | -
|||  Llama 350m b=48 r=128 |
RunLoRA | 946.64 | 70667.72 | 16.44 | 2020.15
PEFT | 1132.82 | 72687.87 | - | -
|||  Llama 350m b=46 r=256 |
RunLoRA | 947.28 | 67742.67 | 16.1 | 3861.98
PEFT | 1129.06 | 71604.64 | - | -
|||  Llama 1.3b b=24 r=128 |
RunLoRA | 1883.23 | 66133.06 | 10.05 | 1015.98
PEFT | 2093.66 | 67149.05 | - | -
|||  Llama 1.3b b=24 r=256 |
RunLoRA | 1955.65 | 66132.01 | 10.69 | 2027.72
PEFT | 2189.84 | 68159.73 | - | -
|||  Llama 1.3b b=24 r=512 |
RunLoRA | 2139.12 | 66132.68 | 12.67 | 4034.06
PEFT | 2449.47 | 70166.73 | - | -
