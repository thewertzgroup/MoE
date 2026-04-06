# Scaling Laws: MoE vs. Dense

One of the strongest arguments for Mixture of Experts is their favorable **scaling behavior** relative to dense models. This page summarizes what we know about how MoE models scale.

## The Core Result

Across multiple studies, MoE models consistently achieve **equivalent quality to dense models at 2-4× less training compute**:

- [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md): 1000× more parameters for same compute, significant quality gains
- [Switch Transformer](../papers/switch-transformer-2021.md): 7× speedup over T5-Base at equivalent quality
- [GLaM](../papers/glam-2022.md): Matched GPT-3 quality with 1/3 the training energy
- [ST-MoE](../papers/st-moe-2022.md): MoE-32B matched quality of dense models 5-10× its active size

## Scaling Laws Formulation

### Dense Scaling Laws (Kaplan et al., 2020; Hoffmann et al., 2022)

For dense models, loss scales as a power law with parameters N, data D, and compute C:

```
L(N) ∝ N^(-α_N)    where α_N ≈ 0.076
L(D) ∝ D^(-α_D)    where α_D ≈ 0.095
L(C) ∝ C^(-α_C)    where α_C ≈ 0.050
```

### MoE Scaling Behavior

MoE models introduce two new dimensions: **total parameters** (P_total) and **active parameters** (P_active), where P_active << P_total.

Key findings:

1. **Quality scales with total parameters**, not just active parameters — more experts (even inactive ones) improve model quality
2. **Compute scales with active parameters** — FLOPs per token proportional to P_active
3. **Diminishing returns**: Adding more experts shows log-linear improvement; doubling experts from 8→16 helps more than 128→256
4. **Expert count sweet spot**: Empirically, 8-64 experts per layer offers the best trade-off between quality gains and engineering complexity

### Granularity Effects

[DeepSeek-MoE](../papers/deepseek-moe-2024.md) showed that **expert granularity** matters:
- Using more, smaller experts (e.g., 64 small experts with top-6) outperforms fewer, larger experts (8 large experts with top-2)
- Fine-grained experts allow more precise specialization
- There's a limit — extremely fine-grained experts (>256) show diminishing returns and increased routing overhead

## Compute-Quality Frontier

```
Quality
  │
  │         ╱ MoE (total params)
  │       ╱
  │     ╱
  │   ╱  ╱ Dense
  │  ╱ ╱
  │╱╱
  └──────────────── Compute (FLOPs)
```

At any given compute budget, an MoE model can achieve better quality by:
1. Using a smaller active model (fewer FLOPs per token)
2. Compensating with more experts (more total parameters)
3. Training on more data with the saved compute

## Training Compute Efficiency

| Model | Total Params | Active Params | Training FLOPs | Quality Reference |
|-------|-------------|---------------|----------------|-------------------|
| GPT-3 (dense) | 175B | 175B | 3.14×10²³ | Baseline |
| [GLaM](../papers/glam-2022.md) | 1.2T | 96.6B | ~1.0×10²³ | Matches GPT-3 |
| [Switch-C](../papers/switch-transformer-2021.md) | 1.6T | ~1.6B | ~1.0×10²³ | Matches T5-XXL |

## Inference Scaling

MoE scaling advantages partially reverse at inference:
- **Throughput**: MoE matches dense compute per token, so throughput is excellent
- **Latency**: Similar to active-parameter equivalent dense model
- **Memory**: Scales with total parameters, not active — this is the main cost
- **Cost per token**: Depends on deployment scenario (see [Inference](inference.md))

### Total Cost of Ownership

When accounting for both training and inference:
- **Training-dominated workloads** (research, one-off tasks): MoE wins decisively
- **Inference-dominated workloads** (high-traffic APIs): Dense may win due to memory efficiency
- **Balanced workloads**: MoE often still wins, especially with expert parallelism

## Open Questions

1. **Do MoE scaling laws follow the same power-law form as dense?** Early evidence suggests yes, but with different exponents.
2. **What is the optimal expert count as a function of total compute?** No definitive answer yet.
3. **Do Chinchilla-optimal ratios apply to MoE?** The optimal data-to-parameter ratio may differ for MoE.
4. **Does expert specialization improve or degrade with scale?** Mixed evidence.

## See Also

- [Overview](overview.md)
- [GLaM](../papers/glam-2022.md)
- [Switch Transformer](../papers/switch-transformer-2021.md)
- [Inference Optimization](inference.md)
