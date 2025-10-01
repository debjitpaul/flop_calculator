# 🧮 FLOP Calculator for Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive toolkit for calculating FLOPs (Floating Point Operations) and MFU (Model FLOPs Utilization) for both **dense Transformer** and **Mixture-of-Experts (MoE)** architectures. This tool helps researchers and practitioners understand computational requirements, optimize training efficiency, and make informed decisions about model architecture and infrastructure.

## 📖 Overview

This calculator implements multiple FLOP counting methodologies:
- **OpenAI Method**: Simple 6N approximation for quick estimates
- **DeepMind Method**: Detailed component-wise calculations including embeddings and attention
- **MoE Extension**: Specialized calculations for sparse Mixture-of-Experts models
- **MFU Analysis**: Hardware utilization metrics across different GPU configurations

## 🚀 Features

- ✅ **Dense Transformer Calculations**: Support for standard Transformer architectures (GPT, LLaMA, etc.)
- ✅ **MoE Architecture Support**: Calculate FLOPs for models with Mixture-of-Experts layers
- ✅ **Multiple Methodologies**: Compare OpenAI vs DeepMind FLOP counting approaches
- ✅ **MFU Measurement**: Calculate Model FLOPs Utilization across different hardware
- ✅ **Hardware Profiles**: Pre-configured profiles for A100, H100, V100, and custom GPUs
- ✅ **Interactive CLI**: Easy-to-use command-line interface
- ✅ **Visualization**: Generate charts comparing efficiency across configurations
- ✅ **Batch Processing**: Calculate FLOPs for multiple model configurations at once

## 📦 Installation

### From PyPI (coming soon)
```bash
pip install flop-calculator
```

### From Source
```bash
git clone https://github.com/debjitpaul/flop_calculator.git
cd flop_calculator
pip install -e .
```

### Requirements
```
python >= 3.8
numpy >= 1.20.0
matplotlib >= 3.3.0
argparse
```

## 🔧 Quick Start

### Command Line Interface

#### Calculate FLOPs for a Dense Model
```bash
python flop_calculator.py \
  --model-type dense \
  --params 7e9 \
  --layers 32 \
  --d-model 4096 \
  --n-heads 32 \
  --seq-len 2048 \
  --vocab-size 32000
```

#### Calculate FLOPs for a MoE Model
```bash
python flop_calculator.py \
  --model-type moe \
  --total-params 120e9 \
  --active-params 5.1e9 \
  --layers 32 \
  --d-model 4096 \
  --num-experts 8 \
  --experts-per-token 2
```

#### Calculate MFU
```bash
python flop_calculator.py \
  --model-type dense \
  --params 7e9 \
  --hardware a100 \
  --num-gpus 8 \
  --throughput 8000 \
  --calculate-mfu
```

### Python API

```python
from flop_calculator import DenseTransformer, MoETransformer, calculate_mfu

# Dense Transformer
model = DenseTransformer(
    n_layers=32,
    d_model=4096,
    n_heads=32,
    seq_len=2048,
    vocab_size=32000
)

# Calculate FLOPs
flops_openai = model.flops_per_token_openai()
flops_deepmind = model.flops_per_token_deepmind()
print(f"OpenAI Method: {flops_openai:.2e} FLOPs/token")
print(f"DeepMind Method: {flops_deepmind:.2e} FLOPs/token")

# MoE Transformer
moe_model = MoETransformer(
    n_layers=32,
    d_model=4096,
    n_heads=32,
    num_experts=8,
    experts_per_token=2,
    total_params=120e9,
    active_params=5.1e9
)

moe_flops = moe_model.flops_per_token()
print(f"MoE FLOPs: {moe_flops:.2e} FLOPs/token")

# Calculate MFU
mfu = calculate_mfu(
    flops_per_token=flops_openai,
    throughput=8000,  # tokens/second
    peak_flops=312e12 * 8,  # 8x A100
)
print(f"MFU: {mfu*100:.1f}%")
```

## 📊 Examples

### Example 1: GPT-3 Scale Model

```python
from flop_calculator import DenseTransformer

gpt3 = DenseTransformer(
    n_layers=96,
    d_model=12288,
    n_heads=96,
    seq_len=2048,
    vocab_size=50257,
    ff_ratio=4
)

# Training on 1024 A100 GPUs
batch_size = 1536
tokens_per_step = batch_size * gpt3.seq_len
flops_per_token = gpt3.flops_per_token_openai()
total_flops = flops_per_token * tokens_per_step

print(f"FLOPs per training step: {total_flops:.2e}")
print(f"Parameters: {gpt3.total_params()/1e9:.1f}B")
```

**Output:**
```
FLOPs per training step: 3.29e+18
Parameters: 175.0B
```

### Example 2: GPT-OSS-120B MoE Model

```python
from flop_calculator import MoETransformer

gpt_oss = MoETransformer(
