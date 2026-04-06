# MoE Wiki

A comprehensive, LLM-maintained knowledge base covering all significant research on **Mixture of Experts (MoE)** architectures in machine learning.

Built following the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — a structured, interconnected markdown wiki where an LLM handles synthesis, cross-referencing, and maintenance while humans direct the inquiry.

## Quick Start

- **[Wiki Index](wiki/index.md)** — Full catalog of all pages
- **[MoE Overview](wiki/concepts/overview.md)** — Start here if you're new to MoE
- **[Change Log](wiki/log.md)** — Chronological record of all updates

## What's Inside

### Concepts (8 pages)
Core technical topics: [routing mechanisms](wiki/concepts/routing.md), [load balancing](wiki/concepts/load-balancing.md), [training challenges](wiki/concepts/training.md), [inference optimization](wiki/concepts/inference.md), [scaling laws](wiki/concepts/scaling-laws.md), [fine-tuning](wiki/concepts/fine-tuning.md), and [sparse upcycling](wiki/concepts/sparse-upcycling.md).

### Papers (10 pages)
Key research papers from [the 1991 original](wiki/papers/adaptive-mixtures-1991.md) through [DeepSeekMoE (2024)](wiki/papers/deepseek-moe-2024.md), including [Sparsely-Gated MoE](wiki/papers/sparsely-gated-moe-2017.md), [Switch Transformer](wiki/papers/switch-transformer-2021.md), [GShard](wiki/papers/gshard-2020.md), [GLaM](wiki/papers/glam-2022.md), [ST-MoE](wiki/papers/st-moe-2022.md), [Expert Choice](wiki/papers/expert-choice-2022.md), [V-MoE](wiki/papers/vision-moe-2021.md), and [Soft MoE](wiki/papers/soft-moe-2023.md).

### Models (7 pages)
Production MoE models: [Mixtral](wiki/models/mixtral.md), [DeepSeek](wiki/models/deepseek.md), [Grok](wiki/models/grok.md), [DBRX](wiki/models/dbrx.md), [Arctic](wiki/models/arctic.md), [Jamba](wiki/models/jamba.md), and [OLMoE](wiki/models/olmoe.md).

### People (1 page)
[Key researchers](wiki/people/key-researchers.md) who shaped MoE: Hinton, Jordan, Shazeer, Fedus, Zoph, and more.

## Structure

```
CLAUDE.md           ← Schema: wiki conventions and workflows
wiki/
├── index.md        ← Content catalog
├── log.md          ← Change log
├── concepts/       ← Core technical concepts (8 pages)
├── papers/         ← Key paper summaries (10 pages)
├── models/         ← Production model profiles (7 pages)
└── people/         ← Researcher profiles (1 page)
```

## Usage

This wiki is designed for three operations:

1. **Browse**: Navigate from the [index](wiki/index.md) or follow cross-references between pages
2. **Query**: Ask questions against the wiki; the LLM retrieves relevant pages and synthesizes answers
3. **Ingest**: Process new sources — the LLM reads, extracts insights, updates pages, and logs changes

## Stats

- **28 pages** covering the full MoE research landscape (1991–2025)
- **10 key papers** summarized with metadata, contributions, and cross-references
- **7 production models** profiled with architecture details and comparisons
- **35+ years** of research history documented
