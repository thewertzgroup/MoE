# Grok (xAI)

Grok is xAI's (Elon Musk's AI company) flagship language model family. **Grok-1**, released with open weights in March 2024, was notable for being the **largest open-weight MoE model** at the time of release.

## Models

### Grok-1

> **Released**: March 2024 (weights), November 2023 (product)
> **License**: Apache 2.0
> **GitHub**: xai-org/grok-1

| Parameter | Value |
|-----------|-------|
| Total parameters | 314B |
| Active parameters per token | ~86B (estimated) |
| Experts per MoE layer | 8 |
| Active experts per token | 2 |
| Transformer layers | 64 |
| Hidden dimension | 6,144 |
| Attention heads | 48 |
| Context length | 8,192 |

**Architecture notes**:
- Standard top-2 of 8 experts routing
- Rotary Position Embeddings (RoPE)
- Similar overall design to [Mixtral](mixtral.md) but much larger
- Each expert significantly larger than Mixtral's experts

**Performance**: At launch, competitive with LLaMA 2 70B and GPT-3.5 on many benchmarks. The model was trained with relatively limited data and compute by frontier standards.

### Grok-2

> **Released**: August 2024 (API/product)
> **Architecture details**: Not publicly disclosed in full

- Significant quality improvement over Grok-1
- Competitive with Claude 3.5 Sonnet and GPT-4o on many benchmarks
- Believed to use an MoE architecture (not confirmed publicly)
- Available via xAI API and the X (Twitter) platform

### Grok-3

> **Released**: February 2025
> **Architecture details**: Limited public information

- Trained on xAI's Colossus supercomputer (100,000 H100 GPUs)
- Claimed frontier-level performance
- MoE architecture details not fully disclosed

## Significance

### Open Weight Release

Grok-1's open-weight release was significant because:
- **Largest open MoE**: 314B parameters, larger than Mixtral 8x7B (46.7B) or 8x22B (141B)
- **Full weights**: Released under Apache 2.0, including all expert weights
- **Community access**: Enabled research and fine-tuning of a large MoE model

### Limitations of the Release

- **No training code or data details**: Weights only, limited reproduction ability
- **Large memory requirements**: 314B parameters requires significant hardware (>600GB at fp16)
- **Limited documentation**: Sparse technical details about architecture choices
- **No instruct version**: Only base model weights released
- **Community adoption**: Limited due to size and lack of ecosystem support compared to Mixtral

## Architectural Comparison

| Model | Total | Active | Experts | Architecture |
|-------|-------|--------|---------|-------------|
| Grok-1 | 314B | ~86B | 8×top-2 | Standard MoE |
| [Mixtral 8x7B](mixtral.md) | 46.7B | 12.9B | 8×top-2 | Standard MoE |
| [Mixtral 8x22B](mixtral.md) | 141B | 39B | 8×top-2 | Standard MoE |
| [DeepSeek-V2](deepseek.md) | 236B | 21B | 160×top-6 + shared | Fine-grained MoE |
| [DBRX](dbrx.md) | 132B | 36B | 16×top-4 | Fine-grained MoE |

## See Also

- [MoE Overview](../concepts/overview.md)
- [Mixtral](mixtral.md) — comparable open MoE model
- [Inference Optimization](../concepts/inference.md)
