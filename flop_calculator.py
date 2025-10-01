from typing import Tuple, Dict
import gradio as gr


def dense_transformer_flops(
    n_layer: int,
    d_model: int,
    d_ff: int,
    d_attn: int,
    n_ctx: int,
    n_vocab: int,
    n_heads: int,
) -> Tuple[tuple, tuple]:
    """Calculate FLOPs for standard dense transformer (DeepMind method)"""
    embeddings = 2 * n_ctx * n_vocab * d_model
    attn_qkv = 2 * n_ctx * 3 * d_model * (d_attn * n_heads)
    attn_logits = 2 * n_ctx * n_ctx * (d_attn * n_heads)
    attn_softmax = 3 * n_heads * n_ctx * n_ctx
    attn_reduce = 2 * n_ctx * n_ctx * (d_attn * n_heads)
    attn_project = 2 * n_ctx * (d_attn * n_heads) * d_model
    ff = 2 * n_ctx * (d_model * d_ff + d_model * d_ff)
    logits = 2 * n_ctx * d_model * n_vocab

    params = (
        embeddings / n_ctx / 2,
        (n_layer * (attn_qkv + attn_project + ff)) / n_ctx / 2,
        logits / n_ctx / 2,
    )

    return (
        embeddings,
        attn_qkv * n_layer,
        attn_logits * n_layer,
        attn_softmax * n_layer,
        attn_reduce * n_layer,
        attn_project * n_layer,
        ff * n_layer,
        logits,
    ), params


def moe_layer_flops(
    n_ctx: int,
    d_model: int,
    d_ff: int,
    num_experts: int,
    experts_per_token: int,
    load_balance_factor: float = 1.0,
    router_type: str = "top-k"
) -> Tuple[int, int, Dict[str, int]]:
    """
    Calculate FLOPs for a single MoE layer
    
    Args:
        n_ctx: Sequence length
        d_model: Hidden dimension
        d_ff: Expert FFN dimension
        num_experts: Total number of experts
        experts_per_token: Number of experts activated per token (k in top-k)
        load_balance_factor: Overhead factor for load balancing (typically 1.0-1.2)
        router_type: Type of router ("top-k", "switch", "expert-choice")
    
    Returns:
        total_flops, active_params, breakdown_dict
    """
    
    # Router FLOPs: Linear projection from d_model to num_experts
    router_flops = 2 * n_ctx * d_model * num_experts
    
    # Expert FLOPs (only for active experts)
    # Each expert: two linear layers (d_model -> d_ff -> d_model)
    expert_flops_per_token = experts_per_token * (
        2 * d_model * d_ff +  # First linear layer
        2 * d_ff * d_model     # Second linear layer
    )
    expert_flops = n_ctx * expert_flops_per_token
    
    # Apply load balancing overhead
    effective_expert_flops = expert_flops * load_balance_factor
    
    # Additional routing overhead based on router type
    if router_type == "expert-choice":
        # Expert-choice routing has additional selection logic
        routing_overhead = 2 * n_ctx * num_experts
    else:
        routing_overhead = 0
    
    total_flops = router_flops + effective_expert_flops + routing_overhead
    
    # Active parameters (only experts used per token)
    active_params_per_token = experts_per_token * (d_model * d_ff + d_ff * d_model)
    active_params = active_params_per_token
    
    breakdown = {
        "router": router_flops,
        "experts": effective_expert_flops,
        "routing_overhead": routing_overhead,
        "total": total_flops
    }
    
    return total_flops, active_params, breakdown


def moe_transformer_flops(
    n_layer: int,
    d_model: int,
    d_ff: int,
    d_attn: int,
    n_ctx: int,
    n_vocab: int,
    n_heads: int,
    num_experts: int,
    experts_per_token: int,
    moe_layers_start: int = 0,
    moe_layers_end: int = None,
    load_balance_factor: float = 1.0,
    router_type: str = "top-k",
    quantization_bits: int = 16,
) -> Tuple[tuple, tuple, Dict]:
    """
    Calculate FLOPs for MoE transformer
    
    Args:
        moe_layers_start: Layer index where MoE starts (0-indexed)
        moe_layers_end: Layer index where MoE ends (None = all remaining layers)
        quantization_bits: Bit precision (16 for bf16, 8 for int8, 4 for int4/MXFP4)
    """
    
    if moe_layers_end is None:
        moe_layers_end = n_layer
    
    # Components that remain the same
    embeddings = 2 * n_ctx * n_vocab * d_model
    logits = 2 * n_ctx * d_model * n_vocab
    
    # Attention components (same for all layers)
    attn_qkv = 2 * n_ctx * 3 * d_model * (d_attn * n_heads)
    attn_logits = 2 * n_ctx * n_ctx * (d_attn * n_heads)
    attn_softmax = 3 * n_heads * n_ctx * n_ctx
    attn_reduce = 2 * n_ctx * n_ctx * (d_attn * n_heads)
    attn_project = 2 * n_ctx * (d_attn * n_heads) * d_model
    
    total_attn_flops = n_layer * (attn_qkv + attn_logits + attn_softmax + 
                                   attn_reduce + attn_project)
    
    # Dense FFN layers (before and after MoE section)
    num_dense_layers = n_layer - (moe_layers_end - moe_layers_start)
    dense_ff_flops = num_dense_layers * 2 * n_ctx * (d_model * d_ff + d_model * d_ff)
    
    # MoE layers
    num_moe_layers = moe_layers_end - moe_layers_start
    moe_total_flops = 0
    moe_active_params = 0
    moe_breakdown = {}
    
    if num_moe_layers > 0:
        layer_flops, layer_params, layer_breakdown = moe_layer_flops(
            n_ctx, d_model, d_ff, num_experts, experts_per_token,
            load_balance_factor, router_type
        )
        moe_total_flops = num_moe_layers * layer_flops
        moe_active_params = layer_params
        moe_breakdown = {k: v * num_moe_layers for k, v in layer_breakdown.items()}
    
    # Quantization overhead factor (approximate computational overhead)
    quant_factor = 1.0
    if quantization_bits == 4:
        quant_factor = 1.05  # MXFP4 has slight overhead for dequantization
    elif quantization_bits == 8:
        quant_factor = 1.02  # INT8 overhead
    
    # Total FLOPs
    total_flops = (embeddings + total_attn_flops + dense_ff_flops + 
                   moe_total_flops * quant_factor + logits)
    
    # Parameter calculation
    total_params = (
        n_vocab * d_model +  # Embeddings
        n_layer * (4 * d_model * d_model) +  # Attention params
        num_dense_layers * (2 * d_model * d_ff) +  # Dense FFN params
        num_moe_layers * num_experts * (2 * d_model * d_ff) +  # All MoE params
        d_model * n_vocab  # Output projection
    )
    
    # Active parameters (parameters actually used per forward pass)
    active_params = (
        n_vocab * d_model +  # Embeddings
        n_layer * (4 * d_model * d_model) +  # Attention params (all active)
        num_dense_layers * (2 * d_model * d_ff) +  # Dense FFN params
        num_moe_layers * moe_active_params +  # Active MoE params only
        d_model * n_vocab  # Output projection
    )
    
    # Detailed breakdown
    breakdown = {
        "embeddings": embeddings,
        "attention": total_attn_flops,
        "dense_ffn": dense_ff_flops,
        "moe_router": moe_breakdown.get("router", 0),
        "moe_experts": moe_breakdown.get("experts", 0),
        "moe_routing_overhead": moe_breakdown.get("routing_overhead", 0),
        "logits": logits,
        "total": total_flops,
        "quantization_factor": quant_factor,
        "total_params": total_params,
        "active_params": active_params,
        "sparsity_ratio": active_params / total_params if total_params > 0 else 0
    }
    
    return (
        embeddings,
        total_attn_flops,
        dense_ff_flops,
        moe_total_flops,
        logits,
        total_flops
    ), (total_params, active_params), breakdown


def calculator(
    model_type: str,
    n_layer: int,
    d_model: int,
    n_heads: int,
    n_vocab: int,
    ff_ratio: int,
    n_ctx: int,
    n_tokens: int,
    # MoE specific parameters
    num_experts: int,
    experts_per_token: int,
    moe_layers_start: int,
    moe_layers_end: int,
    load_balance_factor: float,
    router_type: str,
    quantization_bits: int,
    # Standard parameters
    incl_embed: bool,
    fwd_only: bool,
) -> Tuple:
    """Main calculator function supporting both dense and MoE transformers"""
    
    d_attn = d_model // n_heads
    if d_model % n_heads != 0:
        raise gr.Error("d_model must be divisible by n_heads")
    
    d_ff = d_model * ff_ratio
    
    if model_type == "Dense Transformer":
        flops_terms, params_tuple = dense_transformer_flops(
            n_layer, d_model, d_ff, d_attn, n_ctx, n_vocab, n_heads
        )
        
        if incl_embed:
            flops_per_sequence = sum(flops_terms)
            total_params = sum(params_tuple)
        else:
            flops_per_sequence = sum(flops_terms[1:])
            total_params = sum(params_tuple[1:])
        
        active_params = total_params
        sparsity_ratio = 1.0
        breakdown_text = "Dense model - no sparsity"
        
    else:  # MoE Transformer
        if moe_layers_end == 0:
            moe_layers_end = n_layer
            
        flops_terms, params_tuple, breakdown = moe_transformer_flops(
            n_layer, d_model, d_ff, d_attn, n_ctx, n_vocab, n_heads,
            num_experts, experts_per_token, moe_layers_start, moe_layers_end,
            load_balance_factor, router_type, quantization_bits
        )
        
        total_params, active_params = params_tuple
        
        if not incl_embed:
            flops_per_sequence = breakdown["total"] - breakdown["embeddings"] - breakdown["logits"]
        else:
            flops_per_sequence = breakdown["total"]
        
        sparsity_ratio = breakdown["sparsity_ratio"]
        
        # Format breakdown text
        breakdown_text = f"""
**MoE Breakdown:**
- Total Parameters: {total_params:,.0f}
- Active Parameters: {active_params:,.0f}
- Sparsity Ratio: {sparsity_ratio:.2%}
- Attention FLOPs: {breakdown['attention']:,.0f}
- Dense FFN FLOPs: {breakdown['dense_ffn']:,.0f}
- MoE Router FLOPs: {breakdown['moe_router']:,.0f}
- MoE Experts FLOPs: {breakdown['moe_experts']:,.0f}
- Quantization Overhead: {breakdown['quantization_factor']:.2f}x
"""
    
    flops_per_token = flops_per_sequence / n_ctx
    n_tokens_flops = flops_per_token * n_tokens if n_tokens > 0 else 0
    
    # Apply forward/backward multiplier
    if not fwd_only:
        flops_per_sequence *= 3
        flops_per_token *= 3
        n_tokens_flops *= 3
    
    efficiency_gain = total_params / active_params if active_params > 0 else 1.0
    
    return (
        total_params,
        active_params,
        sparsity_ratio,
        efficiency_gain,
        flops_per_sequence,
        flops_per_token,
        n_tokens_flops,
        breakdown_text
    )


# Gradio Interface
with gr.Blocks(title="MoE & Dense Transformer FLOPs Calculator") as iface:
    gr.Markdown("""
    # 🧮 MoE & Dense Transformer FLOPs Calculator
    
    Calculate FLOPs for both **Dense Transformers** and **Mixture-of-Experts (MoE)** models.
    
    Supports models like GPT-OSS with features including:
    - Top-k expert routing
    - Native quantization (MXFP4, INT8, BF16)
    - Custom MoE layer ranges
    - Load balancing overhead
    
    Based on [DeepMind's Chinchilla paper](https://arxiv.org/abs/2203.15556) extended for MoE architectures.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏗️ Model Configuration")
            
            model_type = gr.Radio(
                ["Dense Transformer", "MoE Transformer"],
                label="Model Type",
                value="MoE Transformer"
            )
            
            gr.Markdown("#### Basic Architecture")
            n_layer = gr.Number(label="Number of layers", value=32)
            d_model = gr.Number(label="Model dimension (d_model)", value=4096)
            n_heads = gr.Number(label="Number of attention heads", value=32)
            n_vocab = gr.Number(label="Vocabulary size", value=128256)
            ff_ratio = gr.Number(value=4, label="FFN expansion ratio")
            
            gr.Markdown("#### MoE Configuration")
            num_experts = gr.Number(
                label="Number of experts per MoE layer",
                value=8,
                visible=True
            )
            experts_per_token = gr.Number(
                label="Experts activated per token (top-k)",
                value=2,
                visible=True
            )
            
            with gr.Row():
                moe_layers_start = gr.Number(
                    label="MoE start layer (0-indexed)",
                    value=0,
                    visible=True
                )
                moe_layers_end = gr.Number(
                    label="MoE end layer (0=all)",
                    value=0,
                    visible=True
                )
            
            load_balance_factor = gr.Slider(
                minimum=1.0,
                maximum=1.5,
                value=1.1,
                step=0.05,
                label="Load balance overhead factor",
                visible=True
            )
            
            router_type = gr.Radio(
                ["top-k", "switch", "expert-choice"],
                label="Router type",
                value="top-k",
                visible=True
            )
            
            quantization_bits = gr.Radio(
                [16, 8, 4],
                label="Quantization (bits)",
                value=4,
                visible=True
            )
            
            gr.Markdown("#### Data Configuration")
            n_ctx = gr.Number(label="Sequence length (context)", value=8192)
            n_tokens = gr.Number(
                value=0,
                label="Total training tokens (optional)",
            )
            
            gr.Markdown("#### Calculation Settings")
            incl_embed = gr.Checkbox(value=True, label="Include embeddings & logits")
            fwd_only = gr.Checkbox(
                value=False,
                label="Forward pass only (uncheck for training: 3x multiplier)"
            )
            
            btn = gr.Button(value="Calculate FLOPs", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### 📊 Results")
            
            with gr.Row():
                total_params = gr.Number(label="Total Parameters")
                active_params = gr.Number(label="Active Parameters")
            
            with gr.Row():
                sparsity_ratio = gr.Number(label="Sparsity Ratio (active/total)")
                efficiency_gain = gr.Number(label="Parameter Efficiency Gain")
            
            gr.Markdown("#### FLOPs Metrics")
            flops_per_sequence = gr.Number(label="FLOPs per sequence")
            flops_per_token = gr.Number(label="FLOPs per token")
            n_tokens_flops = gr.Number(label="Total FLOPs for training")
            
            breakdown_text = gr.Markdown(label="Detailed Breakdown")
    
    # Event handlers
    def update_moe_visibility(model_type):
        visible = model_type == "MoE Transformer"
        return [
            gr.update(visible=visible),  # num_experts
            gr.update(visible=visible),  # experts_per_token
            gr.update(visible=visible),  # moe_layers_start
            gr.update(visible=visible),  # moe_layers_end
            gr.update(visible=visible),  # load_balance_factor
            gr.update(visible=visible),  # router_type
            gr.update(visible=visible),  # quantization_bits
        ]
    
    model_type.change(
        update_moe_visibility,
        inputs=[model_type],
        outputs=[
            num_experts,
            experts_per_token,
            moe_layers_start,
            moe_layers_end,
            load_balance_factor,
            router_type,
            quantization_bits,
        ]
    )
    
    btn.click(
        calculator,
        inputs=[
            model_type,
            n_layer,
            d_model,
            n_heads,
            n_vocab,
            ff_ratio,
            n_ctx,
            n_tokens,
            num_experts,
            experts_per_token,
            moe_layers_start,
            moe_layers_end,
            load_balance_factor,
            router_type,
            quantization_bits,
            incl_embed,
            fwd_only,
        ],
        outputs=[
            total_params,
            active_params,
            sparsity_ratio,
            efficiency_gain,
            flops_per_sequence,
            flops_per_token,
            n_tokens_flops,
            breakdown_text,
        ],
    )
    
    gr.Markdown("### 📝 Pre-configured Examples")
    
    with gr.Tab("GPT-OSS Models"):
        gr.Markdown("""
        **GPT-OSS-120B**: 117B total params, 5.1B active params
        **GPT-OSS-20B**: 21B total params, 3.6B active params
        """)
        gr.Examples(
            [
                # GPT-OSS-120B (estimated configuration)
                ["MoE Transformer", 64, 4096, 32, 128256, 4, 8192, 0, 
                 64, 2, 0, 0, 1.1, "top-k", 4, True, False],
                # GPT-OSS-20B (estimated configuration)
                ["MoE Transformer", 32, 3072, 24, 128256, 4, 8192, 0,
                 16, 2, 0, 0, 1.1, "top-k", 4, True, False],
            ],
            [
                model_type, n_layer, d_model, n_heads, n_vocab, ff_ratio,
                n_ctx, n_tokens, num_experts, experts_per_token,
                moe_layers_start, moe_layers_end, load_balance_factor,
                router_type, quantization_bits, incl_embed, fwd_only,
            ],
            [total_params, active_params, sparsity_ratio, efficiency_gain,
             flops_per_sequence, flops_per_token, n_tokens_flops, breakdown_text],
            calculator,
            cache_examples=False,
        )
    
    with gr.Tab("Dense Models (GPT-3 Family)"):
        gr.Examples(
            [
                ["Dense Transformer", 12, 768, 12, 50257, 4, 4096, 0,
                 1, 1, 0, 0, 1.0, "top-k", 16, True, False],  # 125M
                ["Dense Transformer", 96, 12288, 96, 50257, 4, 4096, 0,
                 1, 1, 0, 0, 1.0, "top-k", 16, True, False],  # 175B
            ],
            [
                model_type, n_layer, d_model, n_heads, n_vocab, ff_ratio,
                n_ctx, n_tokens, num_experts, experts_per_token,
                moe_layers_start, moe_layers_end, load_balance_factor,
                router_type, quantization_bits, incl_embed, fwd_only,
            ],
            [total_params, active_params, sparsity_ratio, efficiency_gain,
             flops_per_sequence, flops_per_token, n_tokens_flops, breakdown_text],
            calculator,
            cache_examples=False,
        )

if __name__ == "__main__":
    iface.launch()
