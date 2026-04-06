# From Sparse to Soft Mixtures of Experts (2023)

> **Title**: From Sparse to Soft Mixtures of Experts
> **Authors**: Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Neil Houlsby
> **Year**: 2023
> **Venue**: ICLR 2024
> **arXiv**: 2308.00951
> **Significance**: Fully differentiable MoE routing; eliminates token dropping and load balancing issues

## Summary

Soft MoE replaces discrete, sparse routing with **fully differentiable** routing. Instead of assigning individual tokens to experts, each expert processes **weighted combinations of all tokens** (soft slots). This eliminates token dropping, load imbalance, and training instability from discrete routing — at the cost of not being usable for autoregressive generation.

## How It Works

### Standard (Sparse) MoE
```
Token_i → Router → Expert_k (one specific expert)
```

### Soft MoE
```
All tokens → Dispatch weights → Soft combinations → Each expert processes a "slot"
Expert outputs → Combine weights → Reconstruct per-token outputs
```

### Detailed Mechanism

1. **Compute dispatch weights**: `D = softmax(X · Φ)` where `Φ` is a learned parameter matrix of shape [d_model × (n_experts × slots_per_expert)]
2. **Create soft slots**: Each slot is a weighted combination of all input tokens: `S_j = Σᵢ D_ij · X_i`
3. **Expert processing**: Each expert processes its assigned slots
4. **Combine outputs**: `Y_i = Σ_j C_ij · E(S_j)` where `C = softmax(X · Φ)` (combine weights)

The key insight: **no discrete decisions** — everything is a soft weighted combination.

## Advantages

### 1. No Token Dropping
Every token contributes to every expert's input (via soft combinations). No token is ever ignored.

### 2. No Load Balancing Issues
Every expert processes exactly the same number of slots. Balance is **structural**, not enforced via auxiliary losses.

### 3. Fully Differentiable
No discrete routing decisions → smooth gradients → easier optimization. No Straight-Through Estimator tricks needed.

### 4. No Training Instability
Eliminates the sharp routing dynamics that cause loss spikes in sparse MoE.

### 5. Better Quality
On vision benchmarks, Soft MoE consistently outperformed both sparse MoE and dense models:
- **ImageNet**: Soft MoE-B/16 outperformed ViT-B/16 and V-MoE-B/16
- More compute-efficient than both sparse MoE and dense baselines

## Limitations

### Cannot Do Autoregressive Generation
The critical limitation: in Soft MoE, each token's representation depends on **all other tokens in the sequence** (through the soft dispatch weights). This means:
- During autoregressive generation, you'd need to recompute all slots when adding each new token
- KV caching doesn't work (slots change with new tokens)
- Makes it unsuitable for decoder-only LLMs in their standard form

### Primarily Validated on Vision
- Most experiments on image classification (ImageNet, JFT)
- Limited exploration in NLP, especially generative settings
- Works well for encoder models and classification

## Results

On image classification:

| Model | Params (active) | ImageNet Top-1 |
|-------|-----------------|----------------|
| ViT-B/16 | 86M | 85.5% |
| V-MoE-B/16 (sparse) | 86M active | 86.0% |
| Soft MoE-B/16 | 86M active | 86.7% |

Soft MoE consistently outperformed both dense and sparse MoE baselines across scales.

## Relation to Other Work

- **Built on [V-MoE](vision-moe-2021.md)**: Same research group, natural evolution
- **Alternative to [Expert Choice](expert-choice-2022.md)**: Both avoid token-to-expert routing, but Soft MoE goes further with fully differentiable combinations
- **Inspired by attention**: The dispatch/combine mechanism resembles cross-attention between tokens and expert slots

## Open Questions

1. Can Soft MoE be adapted for autoregressive models? (Some work on "causal Soft MoE" variants)
2. Does the advantage hold at much larger scales?
3. How does Soft MoE interact with other architectural innovations (e.g., FlashAttention)?

## See Also

- [V-MoE (2021)](vision-moe-2021.md) — predecessor
- [Expert Choice (2022)](expert-choice-2022.md) — parallel idea
- [Routing Mechanisms](../concepts/routing.md)
- [Training Challenges](../concepts/training.md)
