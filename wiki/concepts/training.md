# Training Challenges & Solutions

Training Mixture of Experts models is significantly more complex than training dense models. This page catalogs the major challenges and the solutions developed by the research community.

## 1. Expert Collapse

**The problem**: The router converges to using only a small subset of experts, wasting the capacity of the rest. This is the most common MoE failure mode.

**Symptoms**:
- A few experts receive >90% of tokens
- Most experts have near-zero gate values
- Model quality plateaus well below expectations

**Solutions**:
- [Auxiliary load-balancing loss](load-balancing.md) — penalizes uneven routing
- [Router z-loss](../papers/st-moe-2022.md) — prevents overconfident routing
- Noisy gating — adds noise to encourage exploration
- [Expert Choice routing](../papers/expert-choice-2022.md) — guarantees balance by construction
- Random expert initialization with different seeds (avoid symmetric initialization)

## 2. Training Instability

**The problem**: MoE models are prone to loss spikes and divergence during training, especially at large scale.

**Symptoms**:
- Sudden loss spikes during training
- Gradient explosion in router weights
- NaN/Inf values in expert outputs
- Training divergence after thousands of stable steps

**Causes**:
- Router logits growing unbounded → large softmax outputs → large expert weights
- Discrete routing decisions create non-smooth optimization landscape
- Expert capacity mismatches causing token dropping dynamics to shift abruptly

**Solutions from [ST-MoE](../papers/st-moe-2022.md)**:
1. **Router z-loss**: Penalizes large router logits, preventing softmax saturation
2. **Smaller learning rate for router**: Use 0.1× the base learning rate for router parameters
3. **bfloat16 for router**: Avoid float16 for routing computations (range issues)
4. **Gradient clipping**: More aggressive clipping than dense models (0.3 vs 1.0)
5. **Dropout in experts**: Standard dropout within expert FFNs

**Solutions from practice**:
- **Warm-up the router**: Train with uniform routing for initial steps, then enable learned routing
- **Increase batch size**: Larger batches smooth out routing statistics
- **Monitor router entropy**: Alert on sudden drops in routing entropy

## 3. Representation Collapse

**The problem**: Different experts learn nearly identical representations, defeating the purpose of having multiple experts.

**Symptoms**:
- High cosine similarity between expert weight matrices
- Similar activation patterns across experts
- No meaningful specialization

**Solutions**:
- Diverse initialization schemes
- Expert-level dropout (randomly disable entire experts during training)
- Regularization encouraging expert diversity
- [Shared + routed expert design](../models/deepseek.md) — explicit separation of shared/specialized knowledge

## 4. Token Dropping

**The problem**: When using capacity factors, tokens that exceed an expert's buffer are dropped (skip the MoE layer).

**Impact**:
- Dropped tokens only get the residual connection, losing the FFN transformation
- Can be significant at low capacity factors (5-10% of tokens dropped)
- Non-deterministic — different tokens dropped in different training steps

**Solutions**:
- Increase capacity factor (1.25 → 1.5 or 2.0), at the cost of more memory
- [Expert Choice routing](../papers/expert-choice-2022.md) — no token dropping by design
- [Soft MoE](../papers/soft-moe-2023.md) — all tokens always processed
- Auxiliary "no-op" expert that processes overflow tokens

## 5. Communication Overhead

**The problem**: In distributed training, tokens must be shuffled between devices to reach their assigned experts (all-to-all communication).

**Impact**:
- All-to-all communication is expensive and hard to overlap with compute
- Scales poorly with number of devices
- Can dominate training time at large scale

**Solutions**:
- **Expert parallelism**: Distribute experts across devices, use all-to-all to route tokens
- **Top-1 routing**: Halves communication vs. top-2
- **Capacity factor**: Limits buffer sizes, reducing communication volume
- **Hierarchical MoE**: Route first to device-local expert groups, then to global experts
- **Expert placement optimization**: Co-locate frequently co-activated experts

## 6. Reproducibility and Determinism

**The problem**: MoE training is less deterministic than dense training due to discrete routing decisions and capacity-based token dropping.

**Impact**:
- Harder to debug training issues
- Results may vary across runs more than dense models
- Token dropping patterns differ across data parallel replicas

**Solutions**:
- Deterministic routing (e.g., [hash routing](routing.md)) for debugging
- Synchronized routing decisions across replicas
- Logging router statistics for post-hoc analysis

## Training Recipe Checklist

Based on best practices from [ST-MoE](../papers/st-moe-2022.md), [Switch Transformer](../papers/switch-transformer-2021.md), and [Mixtral](../models/mixtral.md):

- [ ] Use router z-loss (coefficient ~0.001)
- [ ] Use Switch balancing loss (coefficient ~0.01)
- [ ] Use bfloat16 (not float16) for router computations
- [ ] Apply gradient clipping at 0.3-1.0
- [ ] Start with capacity factor 1.25 for top-1, 1.5 for top-2
- [ ] Use smaller learning rate for router parameters
- [ ] Monitor per-expert token counts and router entropy
- [ ] Initialize experts with different random seeds
- [ ] Use large batch sizes for stable routing statistics

## See Also

- [Load Balancing](load-balancing.md)
- [Routing Mechanisms](routing.md)
- [ST-MoE](../papers/st-moe-2022.md)
- [Inference Optimization](inference.md)
