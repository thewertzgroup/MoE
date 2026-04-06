# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)

> **Title**: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
> **Authors**: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
> **Year**: 2017
> **Venue**: ICLR 2017
> **arXiv**: 1701.06538
> **Significance**: Revived and scaled MoE for modern deep learning; introduced sparse gating

## Summary

This paper took the [1991 MoE concept](adaptive-mixtures-1991.md) and made it practical for modern deep learning at massive scale. The key innovation was **sparse gating** — instead of consulting all experts for every input, only the top-k experts are activated per token. This enabled models with up to **137 billion parameters** that used only a small fraction of compute per step.

## Architecture

### Sparsely-Gated MoE Layer

Placed between stacked LSTM layers in a language model:

```
y = Σᵢ G(x)ᵢ · Eᵢ(x)
```

where `G(x)` is a **sparse** gating vector with at most k non-zero entries.

### Noisy Top-k Gating

The gating mechanism adds **learned, input-dependent noise** before selecting the top-k experts:

```
H(x)ᵢ = (x · Wg)ᵢ + StandardNormal() · Softplus((x · Wₙₒᵢₛₑ)ᵢ)
G(x) = Softmax(KeepTopK(H(x), k))
```

- `Wg`: Learned gating weights
- `Wₙₒᵢₛₑ`: Learned noise weights (input-dependent noise amplitude)
- `KeepTopK`: Sets all but top-k values to -∞ before softmax
- The noise encourages exploration of different experts during training

### Scale

- Up to **131,072 experts** per layer (though typical experiments used 4-256)
- Models with up to **137 billion parameters**
- Applied to LSTM-based language models and machine translation

## Key Contributions

### 1. Sparse Gating
The fundamental insight: you don't need to evaluate all experts. Top-k selection (typically k=2 or k=4) provides nearly all the benefit at a fraction of the compute.

### 2. Load Balancing
Introduced two auxiliary losses:
- **Importance loss**: Encourages equal total gating weight across experts
- **Load loss**: Encourages equal number of tokens routed to each expert

Without these losses, the model collapses to using 1-2 experts.

### 3. Scaling Results

On language modeling (1 Billion Word Benchmark):
- MoE models achieved **lower perplexity** than dense baselines with similar compute
- A model with 4B params and 4 active experts per token outperformed a 600M dense LSTM
- Scaling to 137B parameters showed continued improvements

On machine translation (WMT'14 En→Fr):
- MoE model achieved **40.56 BLEU** — state of the art at the time
- Used only a fraction of the compute of comparable dense models

### 4. Conditional Computation

Formalized the idea that **model capacity (total params) can scale independently of compute (active params)**. This is now the defining principle of MoE.

## Technical Details

- **Expert architecture**: Each expert is a feed-forward network
- **Routing granularity**: Token-level (each token independently routed)
- **Training**: Data parallelism with model parallelism for experts
- **Expert placement**: Distributed across multiple devices
- **Gating**: Top-k with k=2 or k=4 in most experiments
- **Number of experts**: 4 to 131,072 (64-2048 typical)

## Challenges Identified

1. **Load imbalance**: Experts receiving vastly different numbers of tokens
2. **Memory overhead**: All parameters must be stored even though only k are active
3. **Communication cost**: All-to-all routing in distributed setting
4. **Batch size sensitivity**: Routing statistics unstable with small batches
5. **Training instability**: More fragile than dense model training

## Impact

This paper is the direct ancestor of virtually all modern MoE work:
- [GShard](gshard-2020.md) scaled the idea to Transformers
- [Switch Transformer](switch-transformer-2021.md) simplified it to top-1
- [Mixtral](../models/mixtral.md), [DeepSeek](../models/deepseek.md), and other modern MoE LLMs all build on this foundation

[Noam Shazeer](../people/key-researchers.md), the first author, went on to co-found Character.AI and remains one of the most influential figures in modern AI architecture design.

## See Also

- [Adaptive Mixtures (1991)](adaptive-mixtures-1991.md) — the original MoE paper
- [Switch Transformer (2021)](switch-transformer-2021.md) — the simplified successor
- [Routing Mechanisms](../concepts/routing.md)
- [Load Balancing](../concepts/load-balancing.md)
- [Key Researchers](../people/key-researchers.md)
