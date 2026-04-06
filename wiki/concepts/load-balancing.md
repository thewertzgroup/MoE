# Load Balancing in MoE

Load balancing is one of the most critical challenges in training Mixture of Experts models. Without explicit mechanisms to encourage balanced expert usage, routers tend to **collapse** — routing most tokens to a small subset of experts while the rest go unused.

## The Problem

Left unconstrained, MoE routing exhibits a **rich-get-richer** dynamic:

1. Early in training, some experts get slightly more tokens by chance
2. These experts receive more gradient updates and improve faster
3. The router learns to send even more tokens to these improving experts
4. Other experts stagnate, receiving fewer tokens and less training signal
5. Eventually, most tokens route to just 1-2 experts — **expert collapse**

This wastes the capacity of unused experts and negates the benefits of MoE.

## Auxiliary Load-Balancing Loss

The standard solution, introduced in [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md) and refined in subsequent work.

### Importance Loss (Shazeer et al., 2017)

Penalizes the coefficient of variation of expert importance scores:

```
Importance(e) = Σ_x g(x)_e    (sum of gate values for expert e across batch)
L_importance = CV(Importance)² = (std(Importance) / mean(Importance))²
```

### Switch Balancing Loss (Fedus et al., 2021)

The [Switch Transformer](../papers/switch-transformer-2021.md) introduced a simpler formulation:

```
L_balance = α · N · Σᵢ fᵢ · pᵢ
```

Where:
- `N` = number of experts
- `fᵢ` = fraction of tokens routed to expert i
- `pᵢ` = average router probability for expert i
- `α` = load balancing coefficient (typically 0.01)

This loss encourages both **uniform routing** (equal fᵢ) and **uniform probabilities** (equal pᵢ).

### Router z-loss (Zoph et al., 2022)

From [ST-MoE](../papers/st-moe-2022.md), an additional loss that penalizes large router logits:

```
L_z = (1/B) Σ_b (log Σ_e exp(z_be))²
```

This prevents the router from becoming overconfident and improves training stability. It's complementary to the balancing loss and is now widely adopted.

## Capacity Factor

Introduced in [GShard](../papers/gshard-2020.md), the **capacity factor (CF)** limits how many tokens each expert can process:

```
Expert buffer size = CF × (total_tokens / num_experts)
```

- **CF = 1.0**: Each expert processes exactly its "fair share" of tokens
- **CF > 1.0** (typical: 1.25): Allows some imbalance; experts can take more than their share
- **CF < 1.0**: More aggressive throttling

Tokens that exceed an expert's capacity are **dropped** — they skip the MoE layer and pass through a residual connection only. This is the **token dropping** problem.

### Trade-offs

| CF Value | Load Balance | Token Dropping | Compute Waste |
|----------|-------------|----------------|---------------|
| 1.0 | Strict | High | None |
| 1.25 | Moderate | Low | ~25% padding |
| 2.0 | Loose | Minimal | ~100% padding |

## Alternative Approaches

### Expert Choice (No Aux Loss Needed)

[Expert Choice Routing](../papers/expert-choice-2022.md) eliminates load balancing losses entirely by having experts choose tokens instead of tokens choosing experts. Each expert selects exactly k tokens, guaranteeing perfect balance.

### Soft MoE (No Aux Loss Needed)

[Soft MoE](../papers/soft-moe-2023.md) uses fully differentiable routing with a fixed number of "slots" per expert, avoiding discrete routing decisions and load imbalance entirely.

### Sinkhorn-based Balancing

Used in [BASE layers](https://arxiv.org/abs/2103.16716) [Lewis et al., 2021]:
- Solves a linear assignment problem using the Sinkhorn algorithm
- Provides optimal balanced assignment of tokens to experts
- More compute overhead than auxiliary losses but better balance

## Practical Recommendations

Based on [ST-MoE](../papers/st-moe-2022.md) and subsequent work:

1. **Always use router z-loss** — improves stability with negligible compute overhead
2. **Use α = 0.01** for the Switch balancing loss as a starting point
3. **Capacity factor 1.25** is a good default for top-1 routing
4. **Monitor expert utilization** during training — uniform is not always optimal
5. **Don't over-balance** — some imbalance is natural and forcing perfect balance can hurt quality

## See Also

- [Routing Mechanisms](routing.md)
- [Training Challenges](training.md)
- [Switch Transformer](../papers/switch-transformer-2021.md)
- [ST-MoE](../papers/st-moe-2022.md)
- [Expert Choice Routing](../papers/expert-choice-2022.md)
