# Qwen MoE (Alibaba)

Alibaba's Qwen team has released several MoE models as part of their Qwen language model family, demonstrating MoE's effectiveness at both small and large scales.

## Models

### Qwen1.5-MoE-A2.7B (February 2024)

> **Released**: February 2024
> **License**: Open

| Parameter | Value |
|-----------|-------|
| Total parameters | 14.3B |
| Active parameters per token | 2.7B |
| Performance reference | Comparable to Qwen1.5-7B |
| Training cost | ~25% of Qwen1.5-7B |

**Key result**: Matched the performance of the 7B dense Qwen1.5 model while using only 2.7B active parameters — achieving equivalent quality at roughly **25% of the training cost**.

### Qwen3-235B-A22B (2025)

> **Released**: 2025
> **License**: Open

| Parameter | Value |
|-----------|-------|
| Total parameters | 235B |
| Active parameters per token | ~22B |
| Context length | 32,768 (native), 131,072 (with YaRN) |

**Key features**:
- Strong performance on reasoning, code, and multilingual tasks
- Extended context via YaRN (Yet another RoPE extensioN) to 131K tokens
- Part of the Qwen3 family which spans multiple scales
- Competitive with other frontier MoE models

## Significance

Qwen's MoE models demonstrate:
1. **MoE at small scale**: Qwen1.5-MoE-A2.7B showed MoE benefits even at the ~3B active parameter level
2. **Cost efficiency**: 25% training cost for equivalent quality is a compelling argument for MoE
3. **Chinese AI ecosystem**: Along with [DeepSeek](deepseek.md), shows strong MoE development from Chinese labs
4. **Scaling range**: Models from 14.3B to 235B total parameters show MoE working across scales

## See Also

- [MoE Overview](../concepts/overview.md)
- [DeepSeek](deepseek.md) — another Chinese MoE model family
- [Mixtral](mixtral.md) — comparable open MoE models
- [Scaling Laws](../concepts/scaling-laws.md)
- [OLMoE](olmoe.md) — another small-scale MoE model
