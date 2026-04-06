# Sparse Upcycling: Dense to MoE Conversion

**Sparse upcycling** is the process of converting a pre-trained dense model into a Mixture of Experts model, reusing the dense model's weights to initialize the MoE architecture. This avoids training an MoE from scratch, leveraging the knowledge already captured in the dense model.

## Key Paper

**"Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"**
- **Authors**: Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, Neil Houlsby
- **Year**: 2022
- **Venue**: ICLR 2023
- **arXiv**: 2212.05055

## How It Works

### Basic Procedure

1. **Start with a pre-trained dense model** (e.g., a standard Transformer with FFN layers)
2. **Replicate each FFN** N times to create N experts (all initialized identically)
3. **Add a randomly initialized router** for each MoE layer
4. **Continue training** the MoE model on additional data

```
Dense FFN                    MoE Layer
┌─────────┐           ┌─────────┐ (copy 1)
│   FFN   │  ──copy→  │ Expert 1│
│         │           ├─────────┤ (copy 2)
│         │  ──copy→  │ Expert 2│
│         │           ├─────────┤ (copy 3)
│         │  ──copy→  │ Expert 3│
│         │           ├─────────┤ (copy N)
│         │  ──copy→  │ Expert N│
└─────────┘           └─────────┘
                      ┌─────────┐
                      │ Router  │ (random init)
                      └─────────┘
```

### Breaking Symmetry

Since all experts start with identical weights, the router must break symmetry:
- **Random router initialization** provides initial differentiation signal
- **Noisy gating** helps exploration during early training
- **Load balancing loss** prevents collapse back to using one expert
- Experts gradually **diverge** and specialize as training continues

## Results

From the original paper:
- Sparse upcycling consistently outperforms both:
  - **Continued dense training** of the original model (same compute)
  - **MoE from scratch** trained for the same total compute
- The advantage is largest in the **low additional compute** regime
- Gap narrows as compute budget increases (eventually, training from scratch catches up)

### Key Findings

1. **Immediate quality boost**: Even before additional training, the upcycled model starts at the dense model's quality level
2. **Faster convergence**: Upcycled MoE converges faster than MoE from scratch
3. **Works across scales**: Demonstrated on both vision (ViT) and language models (T5)
4. **Expert divergence**: Experts gradually specialize away from the initial shared weights
5. **Not all layers need conversion**: Converting every other FFN layer to MoE works well

## Variations

### Partial Upcycling
- Convert only a subset of FFN layers to MoE (e.g., every other layer)
- Reduces memory overhead while capturing most of the quality gains
- Common in practice: [Mixtral](../models/mixtral.md) uses MoE in every layer

### Diverse Initialization
Instead of identical copies, introduce diversity at initialization:
- **Permuted copies**: Randomly permute weights of each expert copy
- **Noisy copies**: Add random noise to each copy
- **Subset copies**: Initialize each expert from a different training checkpoint

### Gradual Upcycling
- Start with 2 experts, train, then split to 4, train, etc.
- Provides more gradual specialization
- More complex training pipeline

## Practical Applications

- **Extending model capabilities**: Take a strong dense model and expand its capacity without starting over
- **Cost-effective MoE training**: Avoids the expensive early phase of MoE training where routing is unstable
- **Model families**: Create MoE variants of existing dense model families (e.g., Llama → Llama-MoE)

### Community Examples
- **LLaMA-MoE**: Community projects upcycling Meta's LLaMA models to MoE
- **OpenMoE**: Open-source efforts to create MoE models via upcycling

## Comparison with Training from Scratch

| Aspect | Sparse Upcycling | From Scratch |
|--------|------------------|-------------|
| Initial quality | Dense model's level | Random |
| Convergence speed | Faster | Slower |
| Total compute (same quality) | Less | More |
| Expert diversity | Starts low, grows | Random from start |
| Training stability | More stable early on | More instabilities |
| Maximum quality ceiling | Possibly slightly lower | Possibly slightly higher |

## See Also

- [Fine-Tuning MoE](fine-tuning.md)
- [Training Challenges](training.md)
- [Overview](overview.md)
- [Mixtral](../models/mixtral.md)
