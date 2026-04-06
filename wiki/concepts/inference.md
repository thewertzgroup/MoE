# Inference Optimization for MoE Models

MoE models present a unique inference challenge: **all parameters must be accessible** (in memory or on fast storage), but only a fraction are used per token. This creates a memory-compute imbalance that requires specialized optimization.

## The Core Challenge

Consider Mixtral 8x7B:
- **Total parameters**: ~46.7B (all 8 experts + shared layers)
- **Active parameters per token**: ~12.9B (2 of 8 experts + shared layers)
- **Compute**: Similar to a 13B dense model
- **Memory**: Similar to a 47B dense model

This means MoE models need **3-4× more memory** than a dense model with equivalent compute, making deployment on consumer hardware much harder.

## Expert Parallelism

The primary strategy for distributed MoE inference. Each device holds a subset of experts.

**How it works**:
1. All devices process the attention layers (replicated)
2. At MoE layers, tokens are routed via all-to-all communication to the device holding their assigned expert
3. Experts compute in parallel
4. Results are sent back via all-to-all communication

**Trade-offs**:
- Reduces per-device memory by spreading experts across devices
- Introduces all-to-all communication latency at every MoE layer
- Works well for batch inference; less efficient for single-sequence generation
- Can be combined with tensor/pipeline parallelism for attention layers

## Expert Offloading

For deployment on devices with limited GPU memory (e.g., consumer GPUs), experts can be stored in CPU memory or on disk and loaded on-demand.

**Strategies**:
1. **Naive offloading**: Load each expert from CPU→GPU when needed. Very slow due to PCIe bandwidth limits.
2. **Predictive offloading**: Use the router output to prefetch the next layer's experts while the current layer computes. Overlaps compute with data transfer.
3. **LRU caching**: Keep recently-used experts on GPU, evict least-recently-used. Works well when routing is "sticky" (same experts used across consecutive tokens).
4. **Speculative expert loading**: Predict which experts will be needed based on token context.

**Key insight from practice**: In autoregressive generation, expert selection tends to be **temporally correlated** — the same experts are often used for consecutive tokens, making caching effective.

## Quantization

MoE models benefit especially from quantization because memory is the bottleneck, not compute:

- **GPTQ / AWQ**: Post-training quantization to 4-bit or 3-bit. Mixtral 8x7B at 4-bit fits in ~26GB (vs ~93GB at fp16).
- **Per-expert quantization**: Different experts may tolerate different quantization levels. Less-used experts can be more aggressively quantized.
- **Mixed precision**: Keep router and attention in higher precision, quantize expert FFNs more aggressively.

## Expert Pruning and Distillation

Reducing the number of experts at inference time:

1. **Expert pruning**: Remove low-utilization experts entirely. If an expert handles <1% of tokens, removing it may have minimal quality impact.
2. **Expert merging**: Average the weights of similar experts to reduce count. Works when experts have high cosine similarity.
3. **Distillation to dense**: Train a smaller dense model to match the MoE model's outputs. Loses the MoE structure but simplifies deployment. [Mixtral](../models/mixtral.md) → dense distillation has been explored by the community.

## Batching Strategies

MoE inference is more efficient with larger batches:

- Larger batches amortize the cost of loading experts (more tokens per expert load)
- Dynamic batching groups requests to maximize expert reuse
- In top-1 routing with 8 experts, a batch of 8 tokens will on average activate ~5.7 unique experts — high overhead. A batch of 64 tokens will activate all 8 — much better amortization.

## Hardware Considerations

| Hardware | Strategy | Notes |
|----------|----------|-------|
| Multi-GPU server | Expert parallelism | Best throughput and latency |
| Single high-VRAM GPU (80GB+) | Full model in memory | Works for smaller MoE models |
| Single consumer GPU (24GB) | 4-bit quantization + offloading | Viable for Mixtral-class models |
| CPU-only | Full offloading | Very slow but functional |
| Edge devices | Distilled dense model | MoE structure too memory-heavy |

## Frameworks and Tools

- **vLLM**: Supports MoE models with expert parallelism and PagedAttention
- **TensorRT-LLM**: NVIDIA's optimized inference for MoE with custom CUDA kernels
- **llama.cpp**: CPU/GPU inference with quantized MoE models (GGUF format)
- **DeepSpeed-MoE**: Microsoft's inference optimization for MoE
- **Megablocks**: Efficient GPU kernels for MoE using block-sparse operations
- **SGLang**: Supports efficient MoE serving with expert parallelism

## See Also

- [Overview](overview.md)
- [Routing Mechanisms](routing.md)
- [Scaling Laws](scaling-laws.md)
- [Mixtral](../models/mixtral.md)
- [DeepSeek](../models/deepseek.md)
