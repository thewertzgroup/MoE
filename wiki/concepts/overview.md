# Mixture of Experts (MoE) — Overview

## What is MoE?

A **Mixture of Experts (MoE)** is a neural network architecture that divides computation among multiple specialized sub-networks ("experts"), with a learned **gating mechanism** (router) that selects which experts process each input. The key insight is that not all parameters need to be active for every input — different experts can specialize in different aspects of the data, enabling models to scale total parameter count without proportionally scaling compute.

## Why MoE Matters

The fundamental tension in deep learning is between **model capacity** (more parameters = more knowledge) and **compute cost** (more parameters = more FLOPs per forward pass). MoE resolves this by introducing **conditional computation**: the model has many parameters but only activates a subset for each input token.

A typical MoE model might have 8× the total parameters of a dense model but use only 1-2× the FLOPs per token, because only a fraction of experts are activated per input. This means:

- **Better quality per FLOP**: MoE models achieve the quality of much larger dense models at a fraction of the training compute
- **Faster training**: For a fixed compute budget, MoE models converge to better quality faster
- **Scaling beyond dense limits**: MoE enables models with hundreds of billions or trillions of parameters that would be infeasible as dense models

## Architecture

A standard Transformer-based MoE replaces the **feed-forward network (FFN)** in each (or every Nth) Transformer block with an MoE layer:

```
Input Token
    │
    ▼
┌──────────┐
│  Router   │ ──→ Produces probability distribution over N experts
└──────────┘
    │
    ▼ (select top-k experts)
┌────┐ ┌────┐ ┌────┐       ┌────┐
│ E1 │ │ E2 │ │ E3 │  ...  │ EN │
└────┘ └────┘ └────┘       └────┘
    │     │                    │
    ▼     ▼                    ▼
  Weighted sum of selected expert outputs
    │
    ▼
Output Token
```

Each expert is typically a standard FFN (two linear layers with an activation function). The router is a simple learned linear layer that maps the input to a probability distribution over experts.

### Key Components

1. **Experts**: Independent sub-networks (usually FFNs), each with identical architecture but different learned parameters
2. **Router / Gating Network**: Determines which experts process each token. See [Routing Mechanisms](routing.md)
3. **Load Balancing**: Auxiliary objectives to prevent expert collapse. See [Load Balancing](load-balancing.md)
4. **Capacity Factor**: Controls how many tokens each expert can process per batch

### Sparse vs. Dense MoE

- **Dense MoE** (original, 1991): All experts are consulted for every input, outputs weighted by gating probabilities. Doesn't save compute.
- **Sparse MoE** (2017 onward): Only top-k experts (typically k=1 or k=2) are activated per token. This is what modern MoE refers to.

## Historical Arc

| Year | Milestone | Significance |
|------|-----------|-------------|
| 1991 | [Adaptive Mixtures of Local Experts](../papers/adaptive-mixtures-1991.md) | Original MoE concept |
| 2017 | [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md) | First sparse MoE at scale in neural nets |
| 2020 | [GShard](../papers/gshard-2020.md) | Scaled MoE to 600B parameters |
| 2021 | [Switch Transformer](../papers/switch-transformer-2021.md) | Simplified to top-1 routing |
| 2021 | [V-MoE](../papers/vision-moe-2021.md) | MoE applied to Vision Transformers |
| 2022 | [GLaM](../papers/glam-2022.md) | 1.2T parameter MoE LLM |
| 2022 | [ST-MoE](../papers/st-moe-2022.md) | Training stability best practices |
| 2022 | [Expert Choice](../papers/expert-choice-2022.md) | Experts choose tokens, not vice versa |
| 2023 | [Soft MoE](../papers/soft-moe-2023.md) | Fully differentiable routing |
| 2023 | [Mixtral 8x7B](../models/mixtral.md) | First strong open-source MoE LLM |
| 2024 | [DeepSeek-V2/V3](../models/deepseek.md) | Fine-grained MoE with shared experts |
| 2024 | [DBRX](../models/dbrx.md), [Arctic](../models/arctic.md), [Jamba](../models/jamba.md) | MoE goes mainstream |

## MoE vs. Dense Models

| Property | Dense | MoE |
|----------|-------|-----|
| Parameters per FLOP | All active | Subset active |
| Memory footprint | = compute cost | >> compute cost |
| Training efficiency | Baseline | 2-4× better quality/FLOP |
| Inference challenges | Standard | Expert parallelism, memory |
| Implementation complexity | Simple | Router, load balancing, communication |

For detailed analysis, see [Scaling Laws](scaling-laws.md).

## Key Challenges

1. **[Training Instability](training.md)**: MoE models are prone to training instabilities, especially at scale
2. **[Load Balancing](load-balancing.md)**: Without auxiliary losses, routers tend to collapse to using few experts
3. **[Inference Cost](inference.md)**: All parameters must be in memory even though only a fraction are used per token
4. **Expert Specialization**: Understanding what individual experts learn remains an open research question
5. **[Fine-tuning](fine-tuning.md)**: Standard fine-tuning can disrupt expert specialization

## See Also

- [Routing Mechanisms](routing.md)
- [Load Balancing](load-balancing.md)
- [Training Challenges](training.md)
- [Inference Optimization](inference.md)
- [Scaling Laws](scaling-laws.md)
- [Key Researchers](../people/key-researchers.md)
