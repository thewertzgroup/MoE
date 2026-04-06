# V-MoE: Scaling Vision with Sparse Mixture of Experts (2021)

> **Title**: Scaling Vision with Sparse Mixture of Experts
> **Authors**: Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, Neil Houlsby
> **Year**: 2021
> **Venue**: NeurIPS 2021
> **arXiv**: 2106.05974
> **Significance**: First successful application of sparse MoE to Vision Transformers

## Summary

V-MoE (Vision Mixture of Experts) applied sparse MoE to **Vision Transformers (ViT)**, scaling them to **15 billion parameters** — the largest vision model at the time. The paper demonstrated that MoE's efficiency benefits transfer from NLP to computer vision, and that vision experts show **clearer specialization patterns** than language experts.

## Architecture

Built on top of the **Vision Transformer (ViT)** architecture:

| Parameter | Value |
|-----------|-------|
| Base architecture | ViT-Huge |
| Total parameters | Up to 15B |
| Experts per MoE layer | 32 |
| Active experts per token (patch) | Top-2 |
| MoE placement | Every other FFN layer |
| Input | Image patches (tokens) |

### MoE for Image Patches

In ViT, an image is split into patches (e.g., 16×16 pixels), each treated as a "token." V-MoE routes each **patch** to different experts:

```
Image → Patches → [CLS] + patch tokens → Transformer + MoE → Classification
```

Different patches (regions of the image) can be routed to different experts, enabling spatial and semantic specialization.

## Key Results

### Scaling Efficiency

- V-MoE-H/14 (15B params, top-2 of 32 experts): **Matched ViT-H/14 quality** with **~2× less inference compute**
- Or, for the same compute budget, V-MoE achieved **higher accuracy**
- ImageNet-1k accuracy: 90.35% (state of the art at the time)

### Expert Specialization in Vision

Unlike language MoE where expert specialization is subtle, V-MoE showed **clear spatial and semantic patterns**:

- Some experts specialized in **foreground objects**
- Others handled **background/texture regions**
- Experts showed preferences for specific **spatial positions** (e.g., center vs. edges)
- Some experts correlated with **object categories** (e.g., animals vs. vehicles)

This interpretability was a notable finding compared to the murkier specialization in [ST-MoE](st-moe-2022.md).

### Priority Routing

V-MoE introduced **Batch Priority Routing (BPR)** for handling capacity constraints:
- Tokens are sorted by their maximum router score (confidence)
- High-confidence tokens are routed first
- Low-confidence tokens are more likely to be dropped
- Rationale: uncertain tokens contribute less to the final prediction

## Contributions

1. **First Vision MoE**: Proved sparse MoE works for vision, not just NLP
2. **Batch Priority Routing**: Better token dropping policy than random
3. **Expert specialization analysis**: Showed vision experts specialize more cleanly than language experts
4. **Scaling vision models**: 15B parameters was the largest vision model at the time
5. **Foundation for Soft MoE**: This team went on to develop [Soft MoE](soft-moe-2023.md)

## Limitations

- **Image classification focus**: Limited exploration of other vision tasks (detection, segmentation)
- **Requires large batch sizes**: Routing statistics need sufficient patches per batch
- **Inference overhead**: Same memory vs. compute tension as language MoE
- **Patch-level routing**: Doesn't capture sub-patch features

## Impact

V-MoE opened the door for MoE in computer vision:
- Led to [Soft MoE](soft-moe-2023.md) from the same research group
- Influenced multimodal MoE designs
- Demonstrated MoE is a **general architecture principle**, not NLP-specific
- The expert specialization findings informed understanding of MoE dynamics

## See Also

- [MoE Overview](../concepts/overview.md)
- [Soft MoE (2023)](soft-moe-2023.md) — follow-up from same group
- [Routing Mechanisms](../concepts/routing.md)
- [Sparse Upcycling](../concepts/sparse-upcycling.md) — related team
