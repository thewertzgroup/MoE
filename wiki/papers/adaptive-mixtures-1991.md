# Adaptive Mixtures of Local Experts (1991)

> **Title**: Adaptive Mixtures of Local Experts
> **Authors**: Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton
> **Year**: 1991
> **Venue**: Neural Computation, Vol. 3, No. 1, pp. 79-87
> **Significance**: The foundational paper that introduced the Mixture of Experts concept

## Summary

This paper introduced the **Mixture of Experts (MoE)** framework — a modular neural network architecture where multiple "expert" networks each specialize in different regions of the input space, and a "gating network" learns to assign inputs to the appropriate expert(s).

## Key Ideas

### Architecture
- Multiple expert networks, each a simple neural network (e.g., single-layer perceptron)
- A **gating network** that takes the same input and outputs a probability distribution over experts
- Final output is a **weighted combination** of all expert outputs, weighted by gating probabilities

```
Output = Σᵢ gᵢ(x) · Eᵢ(x)
```

where `gᵢ(x)` is the gating probability for expert i and `Eᵢ(x)` is expert i's output.

### Training via Competitive Learning

The critical innovation was the training procedure. Rather than training the whole system end-to-end with a single loss, the authors proposed a **competitive** scheme:

1. **Experts compete** to explain each training example
2. The expert that best fits the current example gets the strongest learning signal
3. The gating network learns to predict which expert will best handle each input
4. This creates a **soft partitioning** of the input space

### Error Function

For regression, the error function is:
```
E = -log Σᵢ gᵢ · φ(y | μᵢ, σᵢ)
```

where `φ` is a Gaussian density centered on expert i's prediction. This is equivalent to maximum likelihood estimation of a mixture model.

### Comparison with Backpropagation

The paper showed that the competitive MoE training procedure:
- Converges faster than a single monolithic network trained with backprop
- Produces better generalization on tasks with natural sub-structure
- Learns interpretable expert specializations (each expert handles a distinct region)

## Historical Context

The paper built on several threads:
- **Competitive learning** (Rumelhart & Zipser, 1985)
- **Mixture models** in statistics
- **Modular neural networks** (Jacobs & Jordan's earlier work)
- **Product of experts** ideas from Hinton

The key contribution was formalizing the combination of competitive specialization with a learned gating function in a principled probabilistic framework.

## Limitations

- **Dense computation**: All experts are evaluated for every input (no sparsity)
- **Small scale**: Demonstrated on toy problems with a handful of experts
- **Shallow experts**: Used single-layer networks as experts
- The gap between this and modern MoE (2017+) is the introduction of **sparsity** — only activating a subset of experts per input

## Legacy

This paper planted the seed for all subsequent MoE work. The 26-year gap before [Shazeer et al., 2017](sparsely-gated-moe-2017.md) brought MoE to neural NLP reflects the field's focus on other architectures (CNNs, RNNs, then Transformers) before the scaling era made conditional computation essential.

The four authors went on to become enormously influential:
- **[Geoffrey Hinton](../people/key-researchers.md)**: Turing Award winner, "Godfather of Deep Learning"
- **[Michael Jordan](../people/key-researchers.md)**: Pioneered probabilistic graphical models, one of the most cited ML researchers
- **Robert Jacobs**: Continued work on modular and mixture models
- **Steven Nowlan**: Contributed to weight-sharing and mixture models

## See Also

- [MoE Overview](../concepts/overview.md)
- [Sparsely-Gated MoE (2017)](sparsely-gated-moe-2017.md) — the modern revival
- [Routing Mechanisms](../concepts/routing.md)
- [Key Researchers](../people/key-researchers.md)
