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
    n_layers=32,
    d_model=4096,
    n_heads=32,
    num_experts=64,
    experts_per_token=2,
    total_params=117e9,
    active_params=5.1e9,
    seq_len=2048
)

# Compare with dense equivalent
dense_flops = 6 * gpt_oss.total_params
moe_flops = gpt_oss.flops_per_token()
efficiency_gain = dense_flops / moe_flops

print(f"Dense equivalent: {dense_flops:.2e} FLOPs/token")
print(f"MoE actual: {moe_flops:.2e} FLOPs/token")
print(f"Efficiency gain: {efficiency_gain:.1f}x")
```

**Output:**
```
Dense equivalent: 7.02e+11 FLOPs/token
MoE actual: 3.06e+10 FLOPs/token
Efficiency gain: 22.9x
```

### Example 3: MFU Comparison Across Hardware

```python
from flop_calculator import calculate_mfu, HARDWARE_PROFILES

model_flops = 6 * 7e9  # 7B parameter model
throughput = 8000  # tokens/second

for hardware, specs in HARDWARE_PROFILES.items():
    mfu = calculate_mfu(
        flops_per_token=model_flops,
        throughput=throughput,
        peak_flops=specs['peak_flops'] * 8  # 8 GPUs
    )
    print(f"{hardware}: MFU = {mfu*100:.1f}%")
```

**Output:**
```
A100-40GB: MFU = 13.5%
A100-80GB: MFU = 13.5%
H100-80GB: MFU = 8.5%
V100-32GB: MFU = 33.8%
```

## 🎯 Use Cases

### 1. Planning Training Runs

Estimate computational requirements before starting expensive training:

```bash
python flop_calculator.py \
  --model-type dense \
  --params 70e9 \
  --seq-len 4096 \
  --batch-size 256 \
  --training-tokens 1e12 \
  --estimate-cost
```

### 2. Comparing Architectures

Compare dense vs MoE efficiency:

```bash
python flop_calculator.py \
  --compare \
  --dense-params 120e9 \
  --moe-total-params 120e9 \
  --moe-active-params 5e9
```

### 3. Optimizing Hardware Utilization

Find optimal batch size for target MFU:

```bash
python flop_calculator.py \
  --optimize-batch-size \
  --params 7e9 \
  --hardware a100 \
  --num-gpus 8 \
  --target-mfu 0.5
```

### 4. Benchmarking Performance

Measure actual vs theoretical performance:

```bash
python flop_calculator.py \
  --benchmark \
  --params 7e9 \
  --hardware a100 \
  --num-gpus 8 \
  --duration 300  # 5 minutes
```

## 📐 Calculation Methodologies

### OpenAI Method (Simple Approximation)

**Formula:** `FLOPs per token ≈ 6N`

Where N is the number of non-embedding parameters. The factor of 6 accounts for:
- 2× forward pass
- 2× backward pass (gradients w.r.t. inputs)
- 2× backward pass (gradients w.r.t. parameters)

**Best for:** Quick estimates, comparing models at high level

### DeepMind Method (Detailed)

Calculates FLOPs component-by-component:
- Embeddings: `2 × seq_len × vocab_size × d_model`
- Attention QKV: `2 × seq_len × 3 × d_model × d_attn × n_heads`
- Attention logits: `2 × seq_len² × d_attn × n_heads`
- Attention reduce: `2 × seq_len² × d_attn × n_heads`
- Attention project: `2 × seq_len × d_attn × n_heads × d_model`
- Feedforward: `2 × seq_len × 2 × d_model × d_ff`
- Output logits: `2 × seq_len × d_model × vocab_size`

**Best for:** Detailed analysis, understanding bottlenecks

### MoE Extension

For Mixture-of-Experts models:

```python
# Router FLOPs
router_flops = 2 × d_model × num_experts

# Expert FLOPs (only active experts)
expert_flops = experts_per_token × (2 × d_model × d_ff + 2 × d_ff × d_model)

# Total with load balancing overhead
total_moe_flops = router_flops + expert_flops × load_balance_factor
```

**Best for:** Sparse models, MoE architectures

## 🔬 Advanced Features

### Custom Hardware Profiles

```python
from flop_calculator import add_hardware_profile

add_hardware_profile(
    name="custom_gpu",
    peak_flops_fp16=500e12,  # 500 TFLOPs
    peak_flops_fp32=250e12,  # 250 TFLOPs
    memory_bandwidth=2000e9,  # 2 TB/s
    memory_capacity=80e9      # 80 GB
)
```

### Activation Checkpointing

```python
from flop_calculator import calculate_checkpointing_overhead

base_flops = 6 * 7e9
checkpoint_ratio = 0.5  # Checkpoint 50% of activations
total_flops = calculate_checkpointing_overhead(base_flops, checkpoint_ratio)

print(f"FLOPs overhead: {(total_flops/base_flops - 1)*100:.1f}%")
```

### Scaling Laws

```python
from flop_calculator import compute_optimal_scaling

compute_budget = 1e23  # 100 zettaFLOPs
optimal_params, optimal_tokens = compute_optimal_scaling(compute_budget)

print(f"Optimal model size: {optimal_params/1e9:.0f}B parameters")
print(f"Optimal dataset size: {optimal_tokens/1e9:.0f}B tokens")
```

### Export Results

```bash
# Export to JSON
python flop_calculator.py --config models.yaml --export results.json

# Export to CSV
python flop_calculator.py --config models.yaml --export results.csv

# Generate comparison plots
python flop_calculator.py --config models.yaml --plot comparison.png
```

## 📚 Documentation

Comprehensive documentation available at: [https://debjitpaul.github.io/flop_calculator/](https://debjitpaul.github.io/flop_calculator/)

Topics covered:
- Detailed API reference
- FLOP counting theory and background
- MFU optimization strategies
- Hardware-specific considerations
- Case studies and examples
- Troubleshooting guide

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we'd love help:
- Additional hardware profiles
- Support for new architectures (Mamba, RWKV, etc.)
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📝 Citation

If you use this tool in your research, please cite:

```bibtex
@software{paul2024flop_calculator,
  author = {Debjit Paul},
  title = {FLOP Calculator: A Tool for Estimating Computational Requirements of Large Language Models},
  year = {2024},
  url = {https://github.com/debjitpaul/flop_calculator}
}
```

## 📖 Related Resources

**Blog Post:**
- [Understanding FLOPs, MFU, and Computational Efficiency in LLM Training](https://debjitpaul.github.io/blog/)

**Foundational Papers:**
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (OpenAI)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (DeepMind Chinchilla)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (Google)

**Other Tools:**
- [Adam Casson's Transformer FLOPs Calculator](https://www.adamcasson.com/posts/transformer-flops)
- [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/)

## 🙏 Acknowledgments

This tool builds upon excellent prior work:
- **Adam Casson** for comprehensive Transformer FLOP analysis
- **Pratish Raj** for practical guides to FLOP counting
- **OpenAI, DeepMind, and Google** for foundational scaling research

Special thanks to the open-source community for feedback and contributions.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🐛 Issues and Support

- **Bug reports:** [GitHub Issues](https://github.com/debjitpaul/flop_calculator/issues)
- **Feature requests:** [GitHub Discussions](https://github.com/debjitpaul/flop_calculator/discussions)
- **Questions:** [Stack Overflow](https://stackoverflow.com/questions/tagged/flop-calculator) with tag `flop-calculator`

## 🗺️ Roadmap

**v0.2.0** (Q1 2025)
- [ ] Support for inference FLOP calculations
- [ ] Multi-modal model support
- [ ] Web-based interactive calculator
- [ ] Integration with popular training frameworks

**v0.3.0** (Q2 2025)
- [ ] Support for sparse attention patterns
- [ ] Memory bandwidth analysis
- [ ] Cost estimation per cloud provider
- [ ] Automated benchmark suite

**v1.0.0** (Q3 2025)
- [ ] Production-ready API
- [ ] Comprehensive test coverage
- [ ] Full documentation
- [ ] Community plugins support

## 💬 Community

Join the discussion:
- Twitter: [@debjitpaul](https://twitter.com/debjitpaul)
- Discord: [LLM Efficiency Community](https://discord.gg/llm-efficiency)

---

**Made with ❤️ for the ML community**

*Last updated: October 2025*
