# MoE Wiki — Index

A comprehensive knowledge base on Mixture of Experts (MoE) research, covering foundational theory, key papers, production models, and practical considerations.

---

## Concepts

Core technical concepts in MoE architecture and training.

| Page | Description |
|------|-------------|
| [Overview](concepts/overview.md) | What MoE is, why it matters, architecture overview, historical arc |
| [Routing Mechanisms](concepts/routing.md) | Top-k, expert choice, hash, soft, shared expert routing |
| [Load Balancing](concepts/load-balancing.md) | Auxiliary losses, capacity factors, why balance matters |
| [Training Challenges](concepts/training.md) | Expert collapse, instability, token dropping, solutions |
| [Inference Optimization](concepts/inference.md) | Expert parallelism, offloading, quantization, batching |
| [Scaling Laws](concepts/scaling-laws.md) | MoE vs. dense compute efficiency, scaling behavior |
| [Fine-Tuning MoE](concepts/fine-tuning.md) | LoRA, routing disruption, instruction tuning MoE |
| [Sparse Upcycling](concepts/sparse-upcycling.md) | Converting dense models to MoE, initialization strategies |

## Papers

Key research papers in chronological order.

| Page | Year | Key Contribution |
|------|------|-----------------|
| [Adaptive Mixtures of Local Experts](papers/adaptive-mixtures-1991.md) | 1991 | Original MoE concept (Jacobs, Jordan, Nowlan, Hinton) |
| [Sparsely-Gated MoE](papers/sparsely-gated-moe-2017.md) | 2017 | Sparse gating at scale, noisy top-k (Shazeer et al.) |
| [GShard](papers/gshard-2020.md) | 2020 | 600B MoE for translation, capacity factor (Lepikhin et al.) |
| [Switch Transformer](papers/switch-transformer-2021.md) | 2021 | Top-1 routing, 1.6T params, distillation (Fedus, Zoph, Shazeer) |
| [V-MoE](papers/vision-moe-2021.md) | 2021 | First vision MoE, batch priority routing (Riquelme et al.) |
| [GLaM](papers/glam-2022.md) | 2022 | 1.2T MoE LLM, 1/3 GPT-3 energy cost (Du et al.) |
| [ST-MoE](papers/st-moe-2022.md) | 2022 | Training stability, router z-loss (Zoph et al.) |
| [Expert Choice](papers/expert-choice-2022.md) | 2022 | Experts choose tokens, perfect balance (Zhou et al.) |
| [Soft MoE](papers/soft-moe-2023.md) | 2023 | Fully differentiable routing (Puigcerver et al.) |
| [DeepSeekMoE](papers/deepseek-moe-2024.md) | 2024 | Fine-grained experts, shared expert isolation (Dai et al.) |

## Models

Production MoE models and their characteristics.

| Page | Org | Total Params | Active Params | Experts | Open? |
|------|-----|-------------|---------------|---------|-------|
| [Mixtral](models/mixtral.md) | Mistral AI | 46.7B / 141B | 12.9B / 39B | 8×top-2 | Yes |
| [DeepSeek](models/deepseek.md) | DeepSeek | 236B / 671B | 21B / 37B | 160-256×top-6-8 + shared | Yes |
| [Grok](models/grok.md) | xAI | 314B | ~86B | 8×top-2 | Partial |
| [DBRX](models/dbrx.md) | Databricks | 132B | 36B | 16×top-4 | Yes |
| [Arctic](models/arctic.md) | Snowflake | 480B | 17B | 128×top-2 + dense | Yes |
| [Jamba](models/jamba.md) | AI21 Labs | 52B / 398B | 12B / 94B | 16×top-2 + Mamba | Yes |
| [OLMoE](models/olmoe.md) | AI2 | 6.9B | 1.3B | 64×top-8 | Fully |
| [Llama 4](models/llama4.md) | Meta | 109B / 400B | 17B | 16-128×top-1 + shared | Yes |
| [Qwen MoE](models/qwen-moe.md) | Alibaba | 14.3B / 235B | 2.7B / 22B | Various | Yes |

## People

| Page | Description |
|------|-------------|
| [Key Researchers](people/key-researchers.md) | Founders, architects, and industry leaders of MoE research |

---

*Last updated: 2026-04-06*
*Maintained by LLM following the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)*
