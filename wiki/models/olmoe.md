# OLMoE (AI2)

OLMoE (Open Language Model with Mixture of Experts) is the Allen Institute for AI's (AI2) fully open MoE language model. It is notable for being **completely open** — open weights, open data, open training code, and open evaluation — making it the most transparent large MoE model available.

## Architecture

> **Released**: September 2024
> **Paper**: "OLMoE: Open Mixture-of-Experts Language Models" (Muennighoff et al., 2024, arXiv: 2409.02060)
> **License**: Apache 2.0 (model, code, and data)

| Parameter | Value |
|-----------|-------|
| Total parameters | 6.9B |
| Active parameters per token | 1.3B |
| Experts per MoE layer | 64 |
| Active experts per token | 8 (top-8) |
| Transformer layers | 16 |
| Hidden dimension | 2,048 |
| Attention heads | 16 |
| Context length | 4,096 |
| MoE placement | Every FFN layer |
| Training tokens | 5T |
| Routing | Token-choice top-8 |

## Key Features

### Fully Open

OLMoE's defining feature is **complete openness**:
- **Weights**: Apache 2.0
- **Training data**: Fully documented and available (DCLM + Dolma)
- **Training code**: Open-source (based on OLMo framework)
- **Evaluation code**: Open-source
- **Training logs**: W&B logs publicly available
- **Intermediate checkpoints**: Released for research

This makes it the gold standard for **reproducible MoE research**.

### Fine-Grained Routing

64 experts with top-8 routing gives C(64,8) ≈ 4.4 billion possible expert combinations per token — highly flexible specialization.

### Training Details

- Trained on **5 trillion tokens** from DCLM and Dolma datasets
- Uses SwiGLU expert FFNs
- Standard auxiliary load-balancing loss
- Router z-loss for stability
- Trained with expert parallelism

## Variants

### OLMoE-1B-7B

The base model: 1.3B active parameters, 6.9B total.

### OLMoE-1B-7B-Instruct

Instruction-tuned variant:
- SFT on Tulu 3 dataset
- DPO alignment
- Strong instruction-following performance relative to size

## Performance

For its active parameter budget (~1.3B), OLMoE performs competitively:

- Outperforms dense models of similar active size (e.g., OLMo 1B, Pythia 1.4B)
- Competitive with Mixtral-based models at smaller scale
- Strong on knowledge-intensive benchmarks relative to compute

Compared to other MoE models:
- Much smaller than Mixtral, DeepSeek, DBRX — targeting a different scale point
- Focus is on **research utility** rather than frontier performance
- Demonstrates MoE benefits even at smaller scales

## Research Contributions

The OLMoE paper provided several research insights:

### Expert Specialization Analysis
- Detailed analysis of what experts learn at different layers
- Early layers: less specialization, more general
- Later layers: more specialized, domain-specific patterns
- Routing patterns become more distinct with training

### Scaling Behavior
- Showed MoE scaling benefits hold at smaller scales (1-7B range)
- 64 experts with top-8 is effective even for relatively small models
- Diminishing returns with more than 64 experts at this scale

### Load Balancing Study
- Compared different balancing strategies
- Confirmed router z-loss is essential
- Auxiliary loss coefficient of 0.01 optimal

## Significance

OLMoE fills an important gap in the MoE landscape:
1. **Research-grade openness**: The only fully open MoE model (data + code + weights)
2. **Accessible scale**: Small enough to study and fine-tune on academic hardware
3. **Reproducibility**: Everything needed to reproduce the results is available
4. **Baseline for MoE research**: Provides a clean, well-documented baseline for MoE experiments

## See Also

- [MoE Overview](../concepts/overview.md)
- [Mixtral](mixtral.md) — larger open MoE
- [DeepSeek](deepseek.md) — different MoE architecture
- [Training Challenges](../concepts/training.md)
- [Routing Mechanisms](../concepts/routing.md)
