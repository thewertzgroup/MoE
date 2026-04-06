# Mixtral

Mixtral is Mistral AI's family of open-weight Mixture of Experts language models. [Mixtral 8x7B](https://arxiv.org/abs/2401.04088), released in December 2023, was the **first open-source MoE model to match or exceed GPT-3.5-level performance**, proving that MoE could be both high-quality and openly accessible.

## Models

### Mixtral 8x7B

> **Released**: December 2023
> **Paper**: "Mixtral of Experts" (Jiang et al., 2024, arXiv: 2401.04088)
> **License**: Apache 2.0

| Parameter | Value |
|-----------|-------|
| Total parameters | 46.7B |
| Active parameters per token | 12.9B |
| Experts per MoE layer | 8 |
| Active experts per token | 2 (top-2) |
| Transformer layers | 32 |
| Hidden dimension | 4,096 |
| Attention heads | 32 |
| Context length | 32,768 |
| MoE placement | Every FFN layer |
| Vocabulary size | 32,000 |

**Performance**:
- Matched or exceeded **LLaMA 2 70B** on most benchmarks while using ~2.5× less compute per token
- Competitive with **GPT-3.5-Turbo** on many tasks
- Strong at math, code, and multilingual tasks (especially French)
- MMLU: ~70.6% (vs. LLaMA 2 70B: ~68.9%)

### Mixtral 8x22B

> **Released**: April 2024
> **License**: Apache 2.0

| Parameter | Value |
|-----------|-------|
| Total parameters | 141B |
| Active parameters per token | 39B |
| Experts per MoE layer | 8 |
| Active experts per token | 2 (top-2) |
| Context length | 65,536 |

**Performance**:
- Significant jump over 8x7B across all benchmarks
- Competitive with LLaMA 3 70B and GPT-4-class on many tasks
- MMLU: ~77.8%

### Instruct Variants

Both models have instruction-tuned versions (Mixtral-8x7B-Instruct, Mixtral-8x22B-Instruct) fine-tuned with:
- Supervised fine-tuning (SFT) on instruction data
- Direct Preference Optimization (DPO)
- Strong chat/instruction-following capabilities

## Architecture Details

### What Makes Mixtral's MoE Work

1. **Simple top-2 routing**: Standard softmax router, no exotic routing mechanism
2. **Every-layer MoE**: Unlike many prior works that use MoE every other layer, Mixtral uses MoE in **every** Transformer block's FFN
3. **Moderate expert count**: 8 experts is conservative by research standards but practical for deployment
4. **SwiGLU activation**: Uses SwiGLU (gated linear unit) for expert FFNs, following LLaMA's architecture
5. **Grouped Query Attention (GQA)**: 8 KV heads, shared across the 32 query heads
6. **Sliding Window Attention**: For efficient long-context handling
7. **Byte-fallback BPE tokenizer**: Same as Mistral 7B

### Expert Structure

Each expert is a SwiGLU FFN:
```
Expert(x) = SwiGLU(x · W₁, x · W₃) · W₂
SwiGLU(a, b) = (a ⊙ σ(b)) · W₂
```

Expert FFN intermediate dimension: 14,336

### Routing Analysis

Mistral AI's analysis showed:
- Expert assignment is **not strongly correlated with topic or domain**
- Different experts handle tokens across all domains
- Some evidence of syntactic specialization (function words vs. content words)
- Routing patterns are relatively stable across layers

## Deployment

### Memory Requirements

| Precision | Memory (8x7B) | Memory (8x22B) |
|-----------|---------------|-----------------|
| fp16 | ~93 GB | ~282 GB |
| 8-bit | ~47 GB | ~141 GB |
| 4-bit (GPTQ/AWQ) | ~26 GB | ~75 GB |
| 2-bit (GGUF) | ~16 GB | ~45 GB |

### Inference Characteristics

- **Throughput**: Similar to a 13B dense model (only 2 of 8 experts active)
- **Latency**: Comparable to 13B dense model for single-sequence generation
- **Memory**: Comparable to a 47B dense model (all experts must be loadable)
- Widely supported by vLLM, llama.cpp, TGI, TensorRT-LLM, and other frameworks

## Impact

Mixtral 8x7B was a watershed moment for open-source MoE:

1. **Proved open MoE is viable**: First model to show MoE can be both open-weight and competitive with proprietary models
2. **Made MoE practical**: The 8-expert design is deployable on consumer hardware with quantization
3. **Catalyzed ecosystem**: Spawned a wave of MoE fine-tunes, merges, and derivative models
4. **Influenced competitors**: Prompted other labs to release MoE models ([DBRX](dbrx.md), [Arctic](arctic.md))
5. **Community adoption**: Became one of the most popular open models for fine-tuning and deployment

## See Also

- [MoE Overview](../concepts/overview.md)
- [Routing Mechanisms](../concepts/routing.md)
- [Fine-Tuning MoE](../concepts/fine-tuning.md)
- [Inference Optimization](../concepts/inference.md)
- [DBRX](dbrx.md) — competitor MoE model
- [DeepSeek](deepseek.md) — alternative MoE architecture
