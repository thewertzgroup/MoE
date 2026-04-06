# ST-MoE: Designing Stable and Transferable Sparse Expert Models (2022)

> **Title**: ST-MoE: Designing Stable and Transferable Sparse Expert Models
> **Authors**: Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, William Fedus
> **Year**: 2022
> **Venue**: arXiv preprint
> **arXiv**: 2202.08906
> **Significance**: Definitive guide to MoE training stability; introduced router z-loss

## Summary

ST-MoE is the most comprehensive study of MoE **training stability and transferability** (fine-tuning). It systematically identified causes of training instability, proposed the **router z-loss** as a solution, and provided detailed ablations on fine-tuning MoE models for downstream tasks. This paper is the closest thing to a "practitioner's handbook" for training MoE models.

## Key Contributions

### 1. Router z-Loss

The paper's most impactful contribution. Penalizes large router logits to prevent softmax saturation:

```
L_z = (1/B) Σ_b (log Σ_e exp(z_be))²
```

where `z_be` is the router logit for token b and expert e.

**Why it works**:
- Large router logits → softmax outputs near 0 or 1 → large gradients → instability
- z-loss keeps logits small → smoother softmax → stable training
- Complementary to (not a replacement for) the load balancing loss
- Typical coefficient: 0.001

**Results**: Router z-loss eliminated most training instabilities without hurting quality. Now considered **essential practice** for MoE training.

### 2. Training Stability Analysis

Systematic study of what causes MoE training instability:

| Cause | Mechanism | Solution |
|-------|-----------|----------|
| Large router logits | Softmax saturation → gradient explosion | Router z-loss |
| float16 router | Numerical overflow in softmax | Use bfloat16 or float32 for router |
| High learning rate | Router weights update too aggressively | Lower LR for router (0.1×) |
| Small batch size | Noisy routing statistics | Increase batch size |
| Token dropping | Dynamic capacity changes | Higher capacity factor |

### 3. Transferability (Fine-Tuning)

Comprehensive study of fine-tuning MoE models on downstream tasks:

**Key findings**:
- MoE models are **harder to fine-tune** than dense models of equivalent quality
- They are prone to **overfitting** on small datasets, especially sparse models
- Higher dropout rates help (40-50% for expert layers, vs. 10% for dense)
- **Expert dropout** (randomly dropping entire experts during fine-tuning) is effective
- Quality gap between MoE and dense narrows after fine-tuning

**Fine-tuning recipe**:
1. Use higher dropout in expert FFNs (0.4-0.5)
2. Use standard dropout in attention/shared layers (0.1)
3. Lower learning rate than dense fine-tuning
4. Larger batch sizes for routing stability
5. Continue using load balancing and z-loss during fine-tuning

### 4. Expert Specialization Analysis

The paper analyzed what MoE experts learn:
- Experts show **subtle specialization** but not clean semantic categories
- Some experts handle common/function words (high-frequency)
- Other experts handle domain-specific or rare tokens
- Specialization is **layer-dependent**: early layers show less specialization than later layers
- No simple mapping of "expert N = topic X"

## Architecture Details

The paper studied a 269B parameter encoder-decoder MoE model:

| Parameter | Value |
|-----------|-------|
| Total parameters | 269B |
| Experts per MoE layer | 32 |
| Active experts per token | Top-2 |
| Encoder/decoder layers | 24/24 |
| Hidden dimension | 2,048 |
| FFN dimension | 8,192 |
| MoE placement | Every other layer |

## Ablation Results

### Number of Experts
- 8, 16, 32, 64, 128 experts compared
- 32 experts provided best quality/complexity trade-off
- Diminishing returns above 64 experts for this model size

### MoE Layer Frequency
- Every layer vs. every 2nd vs. every 4th layer
- Every other layer: best trade-off between quality and parameter efficiency

### Capacity Factor
- CF=1.25 recommended for top-1 routing
- CF=1.5-2.0 for top-2 routing
- Higher CF → fewer dropped tokens but more padding waste

## Practical Recommendations Summary

From the paper's "Recommendations" section:

1. **Always use router z-loss** (coefficient 0.001)
2. **Use bfloat16** (not float16) for router computation
3. **Top-2 routing** slightly better than top-1 when compute allows
4. **32-64 experts** is a good range for most model sizes
5. **Every other layer** for MoE placement
6. **Load balancing coefficient α = 0.01**
7. **Higher dropout during fine-tuning** (0.4 for expert layers)
8. **Monitor router entropy** — drops indicate instability

## Impact

ST-MoE's practical recommendations have become the **de facto standard** for MoE training:
- Router z-loss is used in essentially all subsequent MoE models
- The stability findings informed [Mixtral](../models/mixtral.md), [DeepSeek](../models/deepseek.md), and other production MoE models
- The fine-tuning analysis guided community efforts to fine-tune open MoE models

## See Also

- [Switch Transformer (2021)](switch-transformer-2021.md)
- [Training Challenges](../concepts/training.md)
- [Load Balancing](../concepts/load-balancing.md)
- [Fine-Tuning MoE](../concepts/fine-tuning.md)
- [Key Researchers](../people/key-researchers.md)
