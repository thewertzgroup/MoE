# DeepSeek MoE Models

DeepSeek (a Chinese AI lab) has developed one of the most innovative MoE architectures, combining **fine-grained expert segmentation**, **shared experts**, and **Multi-head Latent Attention (MLA)** to achieve state-of-the-art performance with exceptional efficiency.

## Model Family

### DeepSeek-V2 (May 2024)

> **Paper**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
> **arXiv**: 2405.04434

| Parameter | Value |
|-----------|-------|
| Total parameters | 236B |
| Active parameters per token | 21B |
| Shared experts | 2 per layer |
| Routed experts | 160 per layer |
| Active routed experts per token | 6 |
| Transformer layers | 60 |
| Context length | 128K |
| Attention | Multi-head Latent Attention (MLA) |

**Key innovations**:
- **Multi-head Latent Attention (MLA)**: Compresses KV cache via low-rank projection, reducing KV cache by ~93% vs. standard MHA. This makes long-context inference dramatically cheaper.
- **Fine-grained MoE**: 160 routed experts (small) + 2 shared experts (always active), based on [DeepSeekMoE](../papers/deepseek-moe-2024.md)
- **Massive efficiency**: API priced at 1/10th the cost of comparable models due to compute efficiency

**Performance**: Competitive with LLaMA 3 70B and Mixtral 8x22B.

### DeepSeek-V3 (December 2024)

> **Paper**: "DeepSeek-V3 Technical Report"
> **arXiv**: 2412.19437

| Parameter | Value |
|-----------|-------|
| Total parameters | 671B |
| Active parameters per token | 37B |
| Shared experts | 1 per layer |
| Routed experts | 256 per layer |
| Active routed experts per token | 8 |
| Transformer layers | 61 |
| Context length | 128K |
| Attention | Multi-head Latent Attention (MLA) |
| Training tokens | 14.8T |

**Key innovations**:
- **Auxiliary-loss-free load balancing**: Instead of auxiliary losses, uses a bias term added to router logits that is adjusted dynamically to balance load. This avoids the quality degradation from auxiliary losses.
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens simultaneously as a training objective, improving data efficiency.
- **FP8 mixed precision training**: First large-scale model trained with FP8 for expert computations, reducing memory and increasing throughput.
- **Training cost**: Reported only **$5.576 million** for pre-training (2.788M H800 GPU hours) — remarkably cheap for a frontier model.

**Performance**: Competitive with Claude 3.5 Sonnet, GPT-4o, and LLaMA 3.1 405B on most benchmarks, while being dramatically cheaper to train and serve.

### DeepSeek-R1 (January 2025)

Built on DeepSeek-V3 architecture with reinforcement learning for reasoning:
- Chain-of-thought reasoning capabilities
- Strong mathematical and coding performance
- Uses the same MoE backbone as V3

## Architecture Deep Dive

### Fine-Grained MoE

The defining feature of DeepSeek's MoE, introduced in [DeepSeekMoE](../papers/deepseek-moe-2024.md):

```
Standard MoE:     8 large experts, top-2   → C(8,2) = 28 combinations
DeepSeek MoE:   256 small experts, top-8   → C(256,8) ≈ 4.4×10¹³ combinations
```

More combinations = more flexible token-to-computation matching.

### Shared + Routed Experts

```
Token → ┌─ Shared Expert(s)  [always active, ~15% of expert params]
        └─ Router → top-k of N routed experts [sparse, ~85% of expert params]
Output = shared_output + Σ gate_i × routed_expert_i(token)
```

Shared experts handle common patterns; routed experts specialize.

### Multi-head Latent Attention (MLA)

Not MoE-specific but critical to DeepSeek's efficiency:

```
Standard MHA:  KV cache = 2 × n_layers × n_heads × d_head × seq_len
DeepSeek MLA:  KV cache = 2 × n_layers × d_compressed × seq_len
```

Where `d_compressed << n_heads × d_head`. Compresses KV projections into a low-rank latent space.

### Auxiliary-Loss-Free Load Balancing (V3)

DeepSeek-V3 replaced auxiliary losses with a simpler mechanism:
- Each expert has a **bias term** added to its router logit
- Bias is **not learned** — it's adjusted by an online algorithm during training
- If an expert is underutilized, its bias increases; if overutilized, it decreases
- Avoids the quality-balance trade-off inherent in auxiliary losses

## Training Efficiency

DeepSeek's models are notable for their **cost efficiency**:

| Model | Total Params | Training Cost | Comparable Dense |
|-------|-------------|---------------|-----------------|
| DeepSeek-V2 | 236B | ~$2M (estimated) | LLaMA 3 70B |
| DeepSeek-V3 | 671B | $5.6M | LLaMA 3.1 405B |

This efficiency comes from:
1. Fine-grained MoE → lower active params per FLOP
2. MLA → cheaper attention computation and memory
3. FP8 training → higher hardware utilization
4. Efficient training infrastructure on H800 GPUs

## Impact

DeepSeek's MoE innovations have been highly influential:
- **Auxiliary-loss-free balancing** is being adopted by other projects
- **Fine-grained experts** demonstrated that more, smaller experts is better
- **Cost efficiency** challenged assumptions about frontier model training costs
- **Open weights** (V2 and V3) made these innovations accessible to the community

## See Also

- [DeepSeekMoE Paper](../papers/deepseek-moe-2024.md) — foundational research
- [MoE Overview](../concepts/overview.md)
- [Routing Mechanisms](../concepts/routing.md)
- [Load Balancing](../concepts/load-balancing.md)
- [Inference Optimization](../concepts/inference.md)
- [Mixtral](mixtral.md) — alternative MoE approach
