# Jamba (AI21 Labs)

Jamba is AI21 Labs' hybrid architecture that combines **Mixture of Experts** with **Mamba** (a state-space model), representing one of the most architecturally novel MoE models. It replaces the standard Transformer attention with a mix of attention and Mamba layers, combined with MoE for the FFN.

## Architecture

> **Released**: March 2024 (Jamba), August 2024 (Jamba 1.5)
> **Paper**: "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21 Labs, 2024, arXiv: 2403.19887)
> **License**: Jamba Open License (Jamba 1.5: Apache 2.0)

### Jamba (Original)

| Parameter | Value |
|-----------|-------|
| Total parameters | 52B |
| Active parameters per token | 12B |
| Experts per MoE layer | 16 |
| Active experts per token | 2 (top-2) |
| Context length | 256K |
| Architecture | Hybrid Mamba + Attention + MoE |

### Jamba 1.5 Mini

| Parameter | Value |
|-----------|-------|
| Total parameters | 52B |
| Active parameters per token | 12B |
| Context length | 256K |

### Jamba 1.5 Large

| Parameter | Value |
|-----------|-------|
| Total parameters | 398B |
| Active parameters per token | 94B |
| Context length | 256K |

## The Hybrid Architecture

Jamba's key innovation is combining **three** architectural ideas:

### 1. Mamba Layers (State Space Model)

[Mamba](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) is a state-space model (SSM) that processes sequences in **linear time** (O(n)) rather than quadratic time (O(n²)) like attention:
- No KV cache needed — state is compressed into a fixed-size hidden state
- Dramatically lower memory usage for long sequences
- Faster inference for long contexts

### 2. Attention Layers

Standard Transformer attention layers are interspersed with Mamba layers:
- Capture long-range dependencies that SSMs may miss
- Provide "global" information flow
- Used sparingly (e.g., 1 attention layer per 4 Mamba layers)

### 3. MoE for FFN

The feed-forward networks use Mixture of Experts:
- 16 experts, top-2 routing
- Applied in a subset of layers (not every layer)
- Provides conditional computation benefits

### Block Structure

```
┌─────────────────────────┐
│ Mamba Block (×3)        │  ← Linear-time sequence processing
├─────────────────────────┤
│ Attention Block (×1)    │  ← Global information flow
├─────────────────────────┤
│ MoE FFN                 │  ← Conditional computation
└─────────────────────────┘
     ↑ Repeat ↑
```

## Why Mamba + MoE?

The combination addresses complementary scaling challenges:

| Challenge | Solution |
|-----------|----------|
| Quadratic attention cost | Mamba (linear time) |
| Large memory per token | MoE (sparse computation) |
| Long context support | Mamba (fixed-size state) |
| Model capacity | MoE (more total parameters) |

### Long Context Advantage

The Mamba+MoE combination is especially powerful for **long contexts**:
- Mamba provides O(n) sequence processing (vs. O(n²) for attention)
- MoE provides conditional computation (only subset of params active)
- Together: long sequences with large model capacity at reasonable cost
- 256K context length with manageable memory

## Performance

Jamba demonstrated competitive quality with:
- **LLaMA 2 70B** on general benchmarks (with far less active compute)
- **Mixtral 8x7B** on most tasks
- **Superior long-context** performance due to Mamba's linear scaling
- Particularly strong on tasks requiring long-range information retrieval

## Significance

Jamba is architecturally important because:
1. **First Mamba+MoE hybrid**: Showed these can be combined effectively
2. **256K context**: One of the longest context lengths at release
3. **Efficient long-context inference**: Mamba's linear scaling + MoE's sparse compute
4. **Alternative to pure Transformers**: Demonstrated that MoE isn't tied to Transformer attention

## See Also

- [MoE Overview](../concepts/overview.md)
- [Mixtral](mixtral.md) — comparable scale, pure Transformer MoE
- [Inference Optimization](../concepts/inference.md)
- [Scaling Laws](../concepts/scaling-laws.md)
