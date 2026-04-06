# Routing Mechanisms

The **router** (or gating network) is the defining component of a Mixture of Experts architecture. It decides which expert(s) process each input token. The design of routing mechanisms has been one of the most active areas of MoE research.

## How Routing Works

Given an input token representation **x**, the router produces a distribution over **N** experts:

```
g(x) = softmax(W_g · x)
```

where `W_g` is a learned linear projection from the token dimension to N (the number of experts). The top-k experts by gate value are selected, and their outputs are combined as a weighted sum.

## Routing Strategies

### Top-k Routing

The most common approach. Select the k experts with highest gate values.

**Top-2 Routing** — Used in the original [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md) and [GShard](../papers/gshard-2020.md):
- Each token is sent to its top-2 experts
- Output = `g₁·E₁(x) + g₂·E₂(x)` where g₁, g₂ are the gate values
- Provides redundancy and smoother gradients
- More compute per token than top-1

**Top-1 Routing** — Introduced by [Switch Transformer](../papers/switch-transformer-2021.md):
- Each token is sent to exactly one expert
- Simpler, faster, and empirically works as well or better than top-2
- Output = `g₁·E₁(x)` (gate value still used as a scaling factor)
- Reduces communication cost in distributed settings

### Expert Choice Routing

Introduced in [Expert Choice](../papers/expert-choice-2022.md) [Zhou et al., 2022]:
- **Inverts the routing**: instead of tokens choosing experts, experts choose tokens
- Each expert selects its top-k tokens from the batch
- Guarantees perfect load balancing by construction
- Eliminates the need for auxiliary load-balancing losses
- Tokens may be processed by variable numbers of experts (0, 1, 2, or more)
- Risk: some tokens may not be selected by any expert ("token dropping")

### Hash Routing

Introduced in [Hash Layers](https://arxiv.org/abs/2106.04426) [Roller et al., 2021]:
- Deterministic, non-learned routing based on token hashing
- No router parameters, no load-balancing loss needed
- Surprisingly competitive with learned routing in some settings
- Eliminates routing instability entirely

### BASE Layers

[BASE: Balanced Assignment of Sparse Experts](https://arxiv.org/abs/2103.16716) [Lewis et al., 2021]:
- Formulates routing as a linear assignment problem
- Uses auction algorithm to find optimal balanced assignment
- Guarantees each expert processes exactly the same number of tokens
- Higher quality than random/hash routing but more complex

### Soft MoE

Introduced in [Soft MoE](../papers/soft-moe-2023.md) [Puigcerver et al., 2023]:
- **Fully differentiable** — no discrete routing decisions
- Computes weighted combinations of all input tokens for each expert "slot"
- Each expert processes a soft combination of tokens, not individual tokens
- Eliminates token dropping, load balancing issues, and routing instability
- Trade-off: cannot be used for autoregressive generation (tokens see each other)
- Primarily used in vision and encoder models

### Shared Expert + Routed Expert

Used in [DeepSeek-MoE](../papers/deepseek-moe-2024.md) and [DeepSeek-V2](../models/deepseek.md):
- Some experts are **shared** (always active for every token)
- Remaining experts are **routed** (sparsely activated)
- Shared experts capture common knowledge; routed experts specialize
- Reduces redundancy across routed experts

## Routing with Auxiliary Mechanisms

### Noisy Top-k Gating

From [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md):
```
H(x) = W_g · x + StandardNormal() · softplus(W_noise · x)
g(x) = softmax(KeepTopK(H(x), k))
```
- Adds learned, input-dependent noise before the top-k selection
- Noise encourages exploration of different experts during training
- Amplitude of noise is itself learned

### Router z-loss

From [ST-MoE](../papers/st-moe-2022.md):
- Penalizes large router logits: `L_z = (1/B) Σ (log Σ exp(x_i))²`
- Prevents router from becoming too confident
- Significantly improves training stability
- Now standard practice in most MoE implementations

## Comparison Table

| Method | Learned? | Load Balanced? | Differentiable? | Key Paper |
|--------|----------|---------------|-----------------|-----------|
| Top-k | Yes | Needs aux loss | Partially | [Shazeer et al., 2017](../papers/sparsely-gated-moe-2017.md) |
| Top-1 | Yes | Needs aux loss | Partially | [Fedus et al., 2021](../papers/switch-transformer-2021.md) |
| Expert Choice | Yes | By construction | Partially | [Zhou et al., 2022](../papers/expert-choice-2022.md) |
| Hash | No | By construction | N/A | Roller et al., 2021 |
| BASE | Partially | By construction | No | Lewis et al., 2021 |
| Soft MoE | Yes | By construction | Fully | [Puigcerver et al., 2023](../papers/soft-moe-2023.md) |

## What Do Experts Learn?

Research has found varied results on expert specialization:

- In NLP, experts tend to specialize by **token frequency, domain, or syntax** rather than clean semantic categories
- [ST-MoE](../papers/st-moe-2022.md) found that expert specialization is often subtle and hard to interpret
- Some experts become "generalists" handling common tokens; others specialize in rare patterns
- In [V-MoE](../papers/vision-moe-2021.md), experts showed clearer spatial/semantic specialization in images

## See Also

- [Overview](overview.md)
- [Load Balancing](load-balancing.md)
- [Training Challenges](training.md)
- [Expert Choice Routing](../papers/expert-choice-2022.md)
- [Soft MoE](../papers/soft-moe-2023.md)
