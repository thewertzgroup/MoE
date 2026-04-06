# Snowflake Arctic

Snowflake Arctic is an enterprise-focused MoE language model released by Snowflake in April 2024. It is notable for its **Dense-MoE hybrid architecture** that combines a large dense transformer with a smaller MoE residual layer.

## Architecture

> **Released**: April 2024
> **License**: Apache 2.0
> **Paper**: "Snowflake Arctic: The Best MoE for Enterprise AI" (Snowflake, 2024)

| Parameter | Value |
|-----------|-------|
| Total parameters | 480B |
| Active parameters per token | 17B |
| Dense backbone parameters | 10B |
| MoE parameters | 470B |
| Experts per MoE layer | 128 |
| Active experts per token | 2 (top-2) |
| Context length | 4,096 |

## Unique Architecture: Dense-MoE Hybrid

Arctic's architecture is unusual — it combines a **large dense transformer** with a **residual MoE layer**:

```
Input → Dense Transformer (10B) → + MoE Residual (top-2 of 128 experts) → Output
                                  ↑
                              Additive combination
```

### Design Philosophy

- The **dense backbone** (10B params) handles the core language modeling task
- The **MoE layer** (128 experts, top-2 active) provides additional capacity via residual connection
- This is conceptually similar to [DeepSeek's shared experts](deepseek.md) but taken to an extreme — the "shared expert" is the entire dense backbone

### Why This Design?

1. **Enterprise focus**: Optimized for enterprise tasks (SQL, code, reasoning) rather than general chat
2. **Inference efficiency**: The dense backbone handles most of the work; MoE adds refinement
3. **Training efficiency**: Dense backbone can be pre-trained independently, then MoE added
4. **Cost**: Reported **$2 million** training cost (very efficient for 480B total params)

## Performance

Positioned as an "enterprise AI" model, strong on:
- **SQL generation**: Key enterprise use case
- **Code generation**: Competitive coding performance
- **Instruction following**: Good at structured enterprise tasks

Benchmarks showed competitive performance with [Mixtral 8x7B](mixtral.md) and [DBRX](dbrx.md) on enterprise-relevant tasks, though not frontier-level on general benchmarks.

## Training

- **Data**: Not fully disclosed; enterprise-oriented data mix
- **Cost**: ~$2 million (competitive with DeepSeek-V2 for cost efficiency)
- **Hardware**: Trained on NVIDIA GPUs
- **Curriculum**: Enterprise-focused training mixture

## Impact

- Demonstrated an **alternative MoE architecture** (dense + MoE residual)
- Showed that MoE can be tailored for **enterprise use cases** rather than general-purpose chat
- The $2M training cost reinforced MoE's cost-efficiency narrative
- 128 experts with top-2 represents a high expert count with low activation ratio

## Limitations

- **Short context** (4,096 tokens) limits applicability compared to longer-context competitors
- **Niche positioning**: Enterprise focus means less community adoption than general models
- **Limited ecosystem**: Less fine-tuning and tooling support compared to Mixtral

## See Also

- [DBRX](dbrx.md) — another enterprise MoE
- [DeepSeek](deepseek.md) — shared expert comparison
- [MoE Overview](../concepts/overview.md)
- [Inference Optimization](../concepts/inference.md)
