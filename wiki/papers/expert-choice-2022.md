# Mixture-of-Experts with Expert Choice Routing (2022)

> **Title**: Mixture-of-Experts with Expert Choice Routing
> **Authors**: Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, James Laudon
> **Year**: 2022
> **Venue**: NeurIPS 2022
> **arXiv**: 2202.09368
> **Significance**: Inverted routing paradigm — experts choose tokens instead of tokens choosing experts

## Summary

Expert Choice Routing fundamentally **reverses the routing direction**: instead of each token choosing its top-k experts, each **expert chooses its top-k tokens** from the batch. This elegant inversion guarantees perfect load balancing by construction, eliminates the need for auxiliary losses, and achieves better quality than standard top-k routing.

## The Insight

### Standard Routing (Token Choice)
```
Token → selects Expert(s)
Problem: Some experts get too many tokens, others too few
Solution: Auxiliary losses, capacity factors
```

### Expert Choice Routing
```
Expert → selects Token(s)
Guarantee: Each expert processes exactly k tokens
No auxiliary loss needed, no capacity factor needed
```

## How It Works

1. Compute router scores: `S = softmax(X · W_g)` — matrix of shape [tokens × experts]
2. **Transpose the perspective**: for each expert, look at all tokens' scores for that expert
3. Each expert selects its **top-k tokens** (by that expert's column in S)
4. Each expert processes exactly k tokens
5. Combine outputs, weighted by gate scores

```
For expert e:
  selected_tokens = TopK(S[:, e], k)    # expert e's top-k tokens
  output_e = Expert_e(X[selected_tokens]) * S[selected_tokens, e]
```

### Capacity Parameter

Each expert processes exactly:
```
k = capacity_factor × (total_tokens / num_experts)
```

- With CF=1.0, total computation equals one FFN per token (same as dense)
- With CF>1.0, some tokens get processed by multiple experts (more compute, better quality)

## Key Results

### Quality Improvement

Compared to [Switch Transformer](switch-transformer-2021.md) (top-1) and top-2 routing:
- **2% improvement** in pre-training quality over top-2 routing
- Larger gains over top-1 routing
- Improvements consistent across model sizes

### Perfect Load Balancing

By construction, every expert processes exactly the same number of tokens:
- No auxiliary loss needed
- No wasted compute from padding/capacity buffer
- No expert collapse possible

### Variable Expert Computation per Token

Unlike standard routing where every token gets exactly k experts:
- Some tokens may be selected by **0 experts** (processed only via residual)
- Other tokens may be selected by **many experts** (highly processed)
- The model learns to allocate **more compute to harder tokens**
- This heterogeneous computation is a feature, not a bug

### Analysis of Token Selection

The paper found:
- **Stop words and punctuation**: Selected by fewer experts (easier tokens)
- **Content words and rare tokens**: Selected by more experts (harder tokens)
- Naturally implements **adaptive computation** without explicit mechanisms

## Advantages

1. **No auxiliary loss engineering** — eliminates a finicky hyperparameter
2. **No token dropping** — every selected token is processed
3. **Better compute allocation** — harder tokens get more experts
4. **Simpler implementation** — no capacity factor tuning
5. **Better quality** — consistent improvements over token-choice routing

## Limitations

1. **Tokens can be skipped**: If a token isn't selected by any expert, it only gets the residual
2. **Batch-dependent**: Routing depends on other tokens in the batch (not token-independent)
3. **Autoregressive caution**: In causal models, expert choice can introduce token-to-token dependencies that complicate KV caching
4. **Communication pattern**: Different all-to-all pattern than token-choice (experts pull vs. push)

## Impact

Expert Choice routing influenced:
- Thinking about routing as a **resource allocation** problem
- [Soft MoE](soft-moe-2023.md) further developed the "experts process combinations" idea
- Production models have experimented with expert-choice variants
- Highlighted that load balancing and quality can be **simultaneously optimized**

## See Also

- [Routing Mechanisms](../concepts/routing.md)
- [Load Balancing](../concepts/load-balancing.md)
- [Switch Transformer (2021)](switch-transformer-2021.md)
- [Soft MoE (2023)](soft-moe-2023.md)
- [Training Challenges](../concepts/training.md)
