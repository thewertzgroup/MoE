# DBRX (Databricks)

DBRX is Databricks' open-weight Mixture of Experts language model, released in March 2024. It introduced a **fine-grained expert architecture** (16 experts, top-4) that offered a middle ground between Mixtral's coarse design and DeepSeek's extreme granularity.

## Architecture

> **Released**: March 2024
> **Paper**: "DBRX: An Open State-of-the-Art Mixture-of-Experts LLM" (Databricks, 2024)
> **License**: Databricks Open Model License

| Parameter | Value |
|-----------|-------|
| Total parameters | 132B |
| Active parameters per token | 36B |
| Experts per MoE layer | 16 |
| Active experts per token | 4 (top-4) |
| Transformer layers | 40 |
| Hidden dimension | 6,144 |
| Attention heads | 48 |
| KV heads | 8 (GQA) |
| Context length | 32,768 |
| MoE placement | Every FFN layer |
| Vocabulary size | 100,352 |
| Training data | 12T tokens |

## Key Design Choices

### Fine-Grained MoE (16 experts, top-4)

DBRX chose a **middle granularity** between coarse and fine:

| Model | Experts | Active | Combinations |
|-------|---------|--------|-------------|
| [Mixtral](mixtral.md) | 8 | 2 | C(8,2) = 28 |
| **DBRX** | **16** | **4** | **C(16,4) = 1,820** |
| [DeepSeek-V2](deepseek.md) | 160 | 6 | C(160,6) ≈ 2.1×10¹⁰ |

Databricks reported that 16×4 outperformed 8×2 in their ablations — more combinations enable better specialization.

### SwiGLU Expert FFNs

Each expert uses SwiGLU activation:
```
Expert(x) = (xW₁ ⊙ σ(xW₃)) · W₂
```

### Rotary Position Embeddings (RoPE)

Uses RoPE for position encoding, following the LLaMA family convention.

### Grouped Query Attention

8 KV heads shared among 48 query heads, reducing KV cache memory.

## Performance

At release, DBRX claimed state-of-the-art among open models:

| Benchmark | DBRX | Mixtral 8x7B | LLaMA 2 70B | Grok-1 |
|-----------|------|-------------|-------------|--------|
| MMLU | 73.7% | 70.6% | 68.9% | 73.0% |
| HumanEval | 70.1% | 40.2% | 32.3% | 63.2% |
| GSM8K | 66.9% | 58.4% | 54.1% | 62.9% |

Strong performance on code (HumanEval) and math (GSM8K) benchmarks.

## Training

- **Data**: 12 trillion tokens of carefully curated data
- **Curriculum learning**: Data mixture changed during training for optimal quality
- **Hardware**: Trained on Databricks' Mosaic AI platform
- **Cost**: Not publicly disclosed, but positioned as efficient

## Deployment

### Inference Characteristics

- **Compute**: Similar to a 36B dense model (4 of 16 experts active)
- **Memory**: 132B parameters (all must be accessible)
- Supported by vLLM, TGI, and Databricks' own serving infrastructure

### Availability

- Open weights via Hugging Face (databricks/dbrx-base, databricks/dbrx-instruct)
- Optimized serving on Databricks platform
- Available through Databricks Foundation Model APIs

## Impact

- Validated **fine-grained MoE** (more experts, higher top-k) as an effective strategy
- Demonstrated that the **enterprise AI** market could produce competitive open models
- Showed that 16×4 is a practical configuration, providing a data point between Mixtral and DeepSeek granularity
- Strong code and math performance highlighted MoE's potential for reasoning-heavy tasks

## See Also

- [Mixtral](mixtral.md) — comparable open MoE model
- [DeepSeek](deepseek.md) — more fine-grained approach
- [MoE Overview](../concepts/overview.md)
- [Routing Mechanisms](../concepts/routing.md)
- [Arctic](arctic.md) — another enterprise MoE model
