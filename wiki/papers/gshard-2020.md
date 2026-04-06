# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (2020)

> **Title**: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding
> **Authors**: Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen
> **Year**: 2020
> **Venue**: ICLR 2021
> **arXiv**: 2006.16668
> **Significance**: First MoE model at 600B parameters; introduced capacity factor and group-level top-2 gating

## Summary

GShard scaled Mixture of Experts to **600 billion parameters** for multilingual machine translation, using up to 2048 TPU v3 cores. It introduced practical engineering solutions for distributed MoE training, including the **capacity factor**, **group-level dispatching**, and a compiler-based automatic sharding system.

## Architecture

### MoE Integration

- **Base model**: Encoder-decoder Transformer for machine translation
- **MoE placement**: Replace the FFN in every other Transformer block with an MoE layer
- **Expert count**: 2048 experts per MoE layer
- **Routing**: Top-2 gating with group-level dispatching
- **Total parameters**: 600B
- **Active parameters per token**: ~few billion (top-2 of 2048 experts + shared layers)

### Group-Level Top-2 Gating

GShard refined top-2 routing with a **two-pass** dispatch:

1. **First pass**: Each token selects its top-1 expert
2. **Second pass**: Each token selects its second-best expert, subject to capacity constraints
3. If the second expert's buffer is full, the token is processed by only its top-1 expert

This prevents worst-case load imbalance from top-2 routing.

### Capacity Factor

GShard formalized the **capacity factor** concept:

```
Expert buffer size = capacity_factor × (tokens_per_batch / num_experts)
```

- Introduced as a way to bound the maximum number of tokens per expert
- Tokens exceeding capacity are **dropped** (pass through residual only)
- Typical value: CF = 2.0 (later work reduced this to 1.25)

## Key Contributions

### 1. SPMD Automatic Sharding

GShard introduced an **annotation-based sharding** system:
- Developers annotate tensors with partition strategies
- A compiler automatically generates the SPMD (Single Program, Multiple Data) code
- Handles data parallelism + expert parallelism seamlessly
- Made it practical to train MoE on thousands of TPU cores

### 2. Random Routing for Second Expert

For the second expert in top-2:
- Route to the second expert with probability proportional to its gate weight
- This stochastic routing acts as a regularizer
- Reduces communication overhead when the second expert's contribution is small

### 3. Scaling Results

On multilingual machine translation (100+ language pairs):
- **600B MoE** achieved **13.5 BLEU improvement** over a 2.3B dense baseline
- Trained in **4 days** on 2048 TPU v3 cores
- Quality improved consistently with more experts (up to 2048)
- GShard with 600B params used similar compute to a ~10B dense model

## Technical Details

| Parameter | Value |
|-----------|-------|
| Total parameters | 600B |
| Experts per MoE layer | 2048 |
| MoE layers | Every other encoder/decoder layer |
| Routing | Top-2 with group dispatch |
| Capacity factor | 2.0 |
| Training hardware | 2048 TPU v3 cores |
| Training time | ~4 days |

## Load Balancing

GShard used:
1. An auxiliary loss similar to [Shazeer et al., 2017](sparsely-gated-moe-2017.md)
2. Capacity factor to bound worst-case imbalance
3. Random routing for the second expert to reduce hot-spotting

## Impact

GShard was a pivotal scaling demonstration:
- Proved MoE could work at **600B parameters** — larger than any model at the time
- The automatic sharding system influenced Google's subsequent distributed training infrastructure
- Capacity factor became standard in all subsequent MoE work
- Directly led to [GLaM](glam-2022.md) and informed [Switch Transformer](switch-transformer-2021.md)

## Relation to Other Work

GShard and [Switch Transformer](switch-transformer-2021.md) were developed concurrently at Google. Key differences:
- **GShard**: Top-2 routing, encoder-decoder, translation focus, capacity factor emphasis
- **Switch Transformer**: Top-1 routing, encoder-only/decoder-only, language modeling focus, simpler dispatch

## See Also

- [Sparsely-Gated MoE (2017)](sparsely-gated-moe-2017.md) — foundation
- [Switch Transformer (2021)](switch-transformer-2021.md) — concurrent work
- [GLaM (2022)](glam-2022.md) — successor
- [Load Balancing](../concepts/load-balancing.md)
- [Scaling Laws](../concepts/scaling-laws.md)
