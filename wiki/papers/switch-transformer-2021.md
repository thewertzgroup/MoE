# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2021)

> **Title**: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
> **Authors**: William Fedus, Barret Zoph, Noam Shazeer
> **Year**: 2021 (published 2022)
> **Venue**: JMLR, Vol. 23, No. 120
> **arXiv**: 2101.03961
> **Significance**: Simplified MoE routing to top-1; scaled to 1.6 trillion parameters

## Summary

Switch Transformer challenged the prevailing assumption that top-k (k≥2) routing was necessary for MoE models. The authors showed that **top-1 routing** (each token goes to exactly one expert) works as well or better than top-2, while being simpler and more efficient. They scaled this approach to **1.6 trillion parameters** and demonstrated significant speedups over dense T5 models.

## Key Innovation: Top-1 Routing

### Why Top-1 Works

Previous work [Shazeer et al., 2017](sparsely-gated-moe-2017.md) used top-2 routing based on the intuition that redundancy helps. Switch Transformer showed:

1. **Simpler routing**: Each token goes to exactly one expert — cleaner gradients, less communication
2. **Reduced compute**: Half the expert compute per token vs. top-2
3. **Better scaling**: Simpler routing enables scaling to more experts
4. **Capacity factor**: A buffer (typically CF=1.25) handles load imbalance

### Simplified Load Balancing

Introduced a cleaner auxiliary loss:

```
L_balance = α · N · Σᵢ fᵢ · pᵢ
```

where fᵢ is fraction of tokens to expert i, pᵢ is mean router probability for expert i, and α ≈ 0.01. This single scalar loss replaced the more complex importance + load losses from prior work.

## Architecture

```
Standard Transformer Block:        Switch Transformer Block:
┌──────────────┐                   ┌──────────────┐
│  Attention   │                   │  Attention   │
├──────────────┤                   ├──────────────┤
│     FFN      │         →         │  Switch FFN  │
│              │                   │ (MoE Layer)  │
└──────────────┘                   └──────────────┘
```

- Replace FFN in every (or every other) Transformer block with a Switch MoE layer
- Each expert is a standard FFN (same architecture as the original FFN)
- Router is a single linear layer: `softmax(W · x)` → select argmax

## Results

### Speed vs. T5

At matched quality:
- **Switch-Base (128 experts)**: 7× speedup over T5-Base
- **Switch-Large (128 experts)**: 7× speedup over T5-Large
- **Switch-XXL (64 experts)**: Comparable speedup over T5-XXL

### Scale

- **Switch-C**: 1.6 trillion parameters with 2048 experts
- Trained on C4 dataset
- Demonstrated that pre-training quality scales smoothly with expert count

### Distillation

Showed that large MoE models can be **distilled** into smaller dense models:
- Switch-Base (7.4B) → T5-Base (223M): retained 30% of the quality improvement
- Useful for deployment where MoE inference is impractical

## Technical Contributions

### 1. Selective Precision Training

Used **bfloat16** for most operations but **float32** for the router:
- Router softmax is sensitive to numerical precision
- Rest of the model trains fine in lower precision
- This was later confirmed as essential by [ST-MoE](st-moe-2022.md)

### 2. Expert Parallelism Formalization

Formalized expert parallelism as a distinct parallelism strategy:
- Data parallelism: replicate model, split data
- Model parallelism: split model, replicate data
- **Expert parallelism**: distribute experts across devices, route tokens via all-to-all

### 3. Capacity Factor Analysis

Systematic study of capacity factor effects:
- CF=1.0: 10-20% of tokens dropped
- CF=1.25: <5% of tokens dropped (recommended)
- CF=2.0: Minimal dropping, but wasted compute on padding

### 4. Scaling Behavior

Showed that quality improvements from adding experts follow a **log-linear** pattern:
- Doubling experts from 2→4 helps substantially
- Doubling from 128→256 helps less
- Consistent with diminishing returns at very high expert counts

## Limitations

- **Memory hungry**: 1.6T parameters requires massive memory even though compute is modest
- **Token dropping**: Tokens exceeding capacity factor skip the MoE layer
- **Training instability**: Still present, partially addressed by float32 routing
- **Fine-tuning challenges**: Noted but not fully addressed (deferred to [ST-MoE](st-moe-2022.md))

## Impact

Switch Transformer established the **modern MoE recipe** that most subsequent work builds on:
- Top-1 routing with learned softmax gate
- Simple auxiliary balancing loss
- Capacity factor buffering
- Expert parallelism for distribution

It directly influenced [GShard](gshard-2020.md), [GLaM](glam-2022.md), [ST-MoE](st-moe-2022.md), and the design of production models like [Mixtral](../models/mixtral.md).

## See Also

- [Sparsely-Gated MoE (2017)](sparsely-gated-moe-2017.md) — predecessor
- [ST-MoE (2022)](st-moe-2022.md) — stability improvements
- [GShard (2020)](gshard-2020.md) — concurrent scaling work
- [Routing Mechanisms](../concepts/routing.md)
- [Load Balancing](../concepts/load-balancing.md)
