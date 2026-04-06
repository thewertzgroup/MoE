# Llama 4 (Meta)

Llama 4 is Meta's first model family built with a Mixture of Experts architecture. Released in April 2025, it is notable for being **natively multimodal** (trained on interleaved text and images from the start) and for pushing MoE to extreme context lengths.

## Models

### Llama 4 Scout

> **Released**: April 2025
> **License**: Llama 4 Community License

| Parameter | Value |
|-----------|-------|
| Total parameters | 109B |
| Active parameters per token | 17B |
| Experts per MoE layer | 16 |
| Active experts per token | 1 (top-1) + shared |
| Context length | 10,000,000 tokens |
| Modalities | Text + Image (native) |

**Key features**:
- **10M token context window** — by far the longest at release
- Fits on a **single H100 GPU** with Int4 quantization
- Alternating dense and MoE layers
- Natively multimodal — trained on interleaved text and images from the start

### Llama 4 Maverick

> **Released**: April 2025
> **License**: Llama 4 Community License

| Parameter | Value |
|-----------|-------|
| Total parameters | 400B |
| Active parameters per token | 17B |
| Experts per MoE layer | 128 routed + 1 shared |
| Active experts per token | 1 (top-1) + 1 shared |
| Context length | 512,000 tokens |
| Modalities | Text + Image (native) |

**Key features**:
- 128 routed experts + 1 always-active shared expert (similar to [DeepSeek](deepseek.md) approach)
- Top-1 routing keeps active parameters at 17B despite 400B total
- 512K context length
- Competitive with Claude 3.5 Sonnet, GPT-4o, and Gemini 2.0 Pro

### Llama 4 Behemoth (Preview)

- Teacher model used to distill Scout and Maverick
- Reported to outperform GPT-4.5, Claude 3.7 Sonnet, and Gemini 2.0 Pro on STEM benchmarks
- Full details not yet released

## Architecture Highlights

### Native Multimodality

Unlike most MoE models that are text-only with vision added later, Llama 4 was trained from scratch on **interleaved text and image data**. The MoE architecture serves as a natural mechanism for allocating compute across modalities — different experts can specialize in visual vs. textual processing.

### Shared Expert Design

Maverick uses 1 shared expert (always active) + 128 routed experts with top-1 routing. This mirrors [DeepSeek's](deepseek.md) shared expert approach:
- Shared expert handles common patterns across all tokens
- Routed experts specialize in specific patterns
- Combined with top-1 routing, keeps active compute very low (17B) despite massive total capacity (400B)

### Alternating Dense and MoE Layers

Not every layer is MoE — Llama 4 alternates between dense Transformer layers and MoE layers, balancing model capacity with implementation complexity.

## Significance

1. **Meta adopts MoE**: Meta's adoption of MoE for their flagship model family validates the architecture as mainstream
2. **Extreme context lengths**: 10M tokens (Scout) pushes the boundaries of what MoE models can handle
3. **Native multimodality + MoE**: Shows MoE as a natural fit for multimodal compute allocation
4. **Efficient deployment**: Scout fits on a single GPU with quantization, demonstrating practical MoE deployment

## See Also

- [MoE Overview](../concepts/overview.md)
- [DeepSeek](deepseek.md) — shared expert architecture comparison
- [Mixtral](mixtral.md) — earlier open MoE family
- [Routing Mechanisms](../concepts/routing.md)
- [Inference Optimization](../concepts/inference.md)
