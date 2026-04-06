# Key Researchers in MoE

This page highlights the researchers who have made the most significant contributions to the development of Mixture of Experts architectures.

## Founders

### Geoffrey Hinton

**Affiliation**: University of Toronto → Google → Independent
**Key MoE contributions**:
- Co-author of the [original MoE paper](../papers/adaptive-mixtures-1991.md) (1991)
- Co-author of the [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md) paper (2017)
- Turing Award winner (2018) for deep learning contributions
- Bridged the gap from classical MoE to modern sparse MoE

### Michael I. Jordan

**Affiliation**: UC Berkeley
**Key MoE contributions**:
- Co-author of the [original MoE paper](../papers/adaptive-mixtures-1991.md) (1991)
- Developed the theoretical framework for MoE as a probabilistic mixture model
- Influential work on graphical models that informed gating network design
- One of the most cited researchers in machine learning

### Robert A. Jacobs

**Affiliation**: University of Rochester
**Key MoE contributions**:
- First author of the [original MoE paper](../papers/adaptive-mixtures-1991.md) (1991)
- Continued work on modular neural networks and mixture models
- Developed Hierarchical Mixture of Experts with Jordan

## Modern Era Architects

### Noam Shazeer

**Affiliation**: Google → Character.AI
**Key MoE contributions**:
- First author of the [Sparsely-Gated MoE](../papers/sparsely-gated-moe-2017.md) paper (2017) — the paper that revived MoE for modern deep learning
- Co-author of [Switch Transformer](../papers/switch-transformer-2021.md)
- Co-author of [GShard](../papers/gshard-2020.md)
- Co-author of [ST-MoE](../papers/st-moe-2022.md)

Also known for:
- Co-inventor of the **Transformer** architecture ("Attention Is All You Need", 2017)
- Inventor of **multi-query attention**, **SwiGLU**, and other efficiency innovations
- Co-founder of Character.AI
- Arguably the single most influential individual in modern MoE development

### William Fedus

**Affiliation**: Google Brain → OpenAI
**Key MoE contributions**:
- First author of [Switch Transformer](../papers/switch-transformer-2021.md) — simplified MoE with top-1 routing
- Co-author of [ST-MoE](../papers/st-moe-2022.md) — training stability study
- Developed practical scaling laws for sparse models
- Made MoE accessible through clear writing and systematic ablations

### Barret Zoph

**Affiliation**: Google Brain
**Key MoE contributions**:
- Co-author of [Switch Transformer](../papers/switch-transformer-2021.md)
- First author of [ST-MoE](../papers/st-moe-2022.md) — definitive MoE training guide
- Introduced the **router z-loss** that stabilized MoE training
- Earlier known for Neural Architecture Search (NAS) work

### Dmitry Lepikhin

**Affiliation**: Google
**Key MoE contributions**:
- First author of [GShard](../papers/gshard-2020.md) — scaled MoE to 600B parameters
- Developed the **capacity factor** concept
- Pioneered automatic sharding for distributed MoE training
- Enabled practical MoE at datacenter scale

## Routing & Architecture Innovators

### Yanqi Zhou

**Affiliation**: Google Brain
**Key MoE contributions**:
- First author of [Expert Choice Routing](../papers/expert-choice-2022.md) — inverted the routing paradigm
- Co-author of [GLaM](../papers/glam-2022.md)
- Demonstrated that experts choosing tokens is superior to tokens choosing experts

### Joan Puigcerver

**Affiliation**: Google DeepMind
**Key MoE contributions**:
- First author of [Soft MoE](../papers/soft-moe-2023.md) — fully differentiable routing
- Co-author of [V-MoE](../papers/vision-moe-2021.md) — first vision MoE
- Co-author of [Sparse Upcycling](../concepts/sparse-upcycling.md)
- Developed the progression from sparse → soft MoE

### Carlos Riquelme

**Affiliation**: Google DeepMind
**Key MoE contributions**:
- First author of [V-MoE](../papers/vision-moe-2021.md) — brought MoE to computer vision
- Co-author of [Soft MoE](../papers/soft-moe-2023.md)
- Co-author of [Sparse Upcycling](../concepts/sparse-upcycling.md)
- Demonstrated expert specialization patterns in vision

### Neil Houlsby

**Affiliation**: Google DeepMind
**Key MoE contributions**:
- Co-author of [V-MoE](../papers/vision-moe-2021.md), [Soft MoE](../papers/soft-moe-2023.md), and [Sparse Upcycling](../concepts/sparse-upcycling.md)
- Led the research group that produced the V-MoE → Soft MoE progression
- Also known for adapter methods in NLP

## Industry Leaders

### Nan Du

**Affiliation**: Google
**Key MoE contributions**:
- First author of [GLaM](../papers/glam-2022.md) — 1.2T parameter MoE LLM
- Demonstrated MoE's superiority over GPT-3 at lower compute cost
- Co-author of [Expert Choice](../papers/expert-choice-2022.md) and [ST-MoE](../papers/st-moe-2022.md)

### Mistral AI Team

**Key members**: Arthur Mensch, Timothée Lacroix, Guillaume Lample, Albert Jiang
**Contributions**:
- Created [Mixtral 8x7B and 8x22B](../models/mixtral.md) — the first strong open MoE LLMs
- Proved MoE could be competitive with proprietary models as open-weight
- Catalyzed the open-source MoE ecosystem

### DeepSeek Team

**Key members**: Damai Dai, Chengqi Deng, and the DeepSeek research team
**Contributions**:
- Developed [fine-grained MoE with shared experts](../papers/deepseek-moe-2024.md)
- Created [DeepSeek-V2 and V3](../models/deepseek.md) — cost-efficient frontier MoE models
- Introduced auxiliary-loss-free load balancing
- Pioneered Multi-head Latent Attention (MLA)

## Contributions Timeline

| Year | Person/Team | Contribution |
|------|-------------|-------------|
| 1991 | Jacobs, Jordan, Nowlan, Hinton | Original MoE concept |
| 2017 | Shazeer et al. | Sparse gating at scale |
| 2020 | Lepikhin et al. | GShard (600B MoE) |
| 2021 | Fedus, Zoph, Shazeer | Switch Transformer (top-1) |
| 2021 | Riquelme et al. | V-MoE (vision) |
| 2022 | Zhou et al. | Expert Choice routing |
| 2022 | Zoph et al. | ST-MoE (stability) |
| 2022 | Du et al. | GLaM (1.2T MoE LLM) |
| 2023 | Puigcerver et al. | Soft MoE |
| 2023 | Mistral AI | Mixtral 8x7B (open MoE) |
| 2024 | DeepSeek | Fine-grained MoE architecture |
| 2024 | Databricks | DBRX |
| 2024 | Snowflake | Arctic |
| 2024 | AI21 Labs | Jamba (Mamba+MoE) |
| 2024 | AI2 | OLMoE (fully open) |

## See Also

- [MoE Overview](../concepts/overview.md)
- [All Papers](../index.md)
