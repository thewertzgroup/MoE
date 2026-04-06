# MoE Wiki — Change Log

Append-only chronological record of all wiki operations.

---

## 2026-04-06 — Initial Wiki Creation

**Operation**: Full wiki build from scratch

**Pages created** (23 total):

### Concepts (8 pages)
- `concepts/overview.md` — MoE overview, architecture, historical arc, comparison with dense
- `concepts/routing.md` — Top-k, expert choice, hash, BASE, soft MoE routing mechanisms
- `concepts/load-balancing.md` — Auxiliary losses, capacity factors, Sinkhorn balancing
- `concepts/training.md` — Expert collapse, instability, token dropping, training recipes
- `concepts/inference.md` — Expert parallelism, offloading, quantization, batching strategies
- `concepts/scaling-laws.md` — MoE vs. dense scaling, compute efficiency analysis
- `concepts/fine-tuning.md` — LoRA for MoE, routing disruption, instruction tuning
- `concepts/sparse-upcycling.md` — Dense-to-MoE conversion, initialization strategies

### Papers (10 pages)
- `papers/adaptive-mixtures-1991.md` — Jacobs, Jordan, Nowlan, Hinton (1991)
- `papers/sparsely-gated-moe-2017.md` — Shazeer et al. (2017)
- `papers/gshard-2020.md` — Lepikhin et al. (2020)
- `papers/switch-transformer-2021.md` — Fedus, Zoph, Shazeer (2021)
- `papers/vision-moe-2021.md` — Riquelme et al. (2021)
- `papers/glam-2022.md` — Du et al. (2022)
- `papers/st-moe-2022.md` — Zoph et al. (2022)
- `papers/expert-choice-2022.md` — Zhou et al. (2022)
- `papers/soft-moe-2023.md` — Puigcerver et al. (2023)
- `papers/deepseek-moe-2024.md` — Dai et al. (2024)

### Models (7 pages)
- `models/mixtral.md` — Mistral AI's Mixtral 8x7B and 8x22B
- `models/deepseek.md` — DeepSeek-V2, V3, and R1
- `models/grok.md` — xAI's Grok-1, 2, 3
- `models/dbrx.md` — Databricks' DBRX
- `models/arctic.md` — Snowflake Arctic
- `models/jamba.md` — AI21 Labs' Jamba (Mamba+MoE hybrid)
- `models/olmoe.md` — AI2's OLMoE (fully open)

### People (1 page)
- `people/key-researchers.md` — Hinton, Jordan, Shazeer, Fedus, Zoph, and more

### Navigation (2 files)
- `index.md` — Content catalog with tables and one-line summaries
- `log.md` — This file

### Schema
- `CLAUDE.md` — Wiki conventions, directory layout, and workflows

**Sources**: Built from comprehensive survey of MoE research literature spanning 1991-2025, including foundational papers, scaling studies, production model reports, and community resources.

**Cross-references**: All pages interlinked via relative markdown links. Each page includes a "See Also" section with related pages.
