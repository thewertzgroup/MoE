# MoE Wiki — Schema & Conventions

## Purpose

This is an LLM-maintained wiki covering all significant research on **Mixture of Experts (MoE)** architectures in machine learning. It follows the [LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## Three-Layer Architecture

1. **Raw Sources** — The papers, blog posts, and technical reports referenced throughout. Cited inline with `[Author et al., Year]` format and collected in each page's References section.
2. **The Wiki** — The `wiki/` directory of interconnected markdown pages. The LLM owns this layer completely.
3. **The Schema** — This file (`CLAUDE.md`). Defines structure, conventions, and workflows.

## Directory Layout

```
wiki/
├── index.md            # Content catalog organized by category
├── log.md              # Append-only chronological record of changes
├── concepts/           # Core technical concepts
│   ├── overview.md     # What MoE is, why it matters
│   ├── routing.md      # Routing mechanisms (top-k, expert choice, hash, soft, etc.)
│   ├── load-balancing.md
│   ├── training.md     # Training challenges & solutions
│   ├── inference.md    # Inference optimization
│   ├── scaling-laws.md # MoE vs dense scaling
│   ├── fine-tuning.md  # Fine-tuning & adaptation
│   └── sparse-upcycling.md
├── papers/             # Key paper summaries
│   ├── adaptive-mixtures-1991.md
│   ├── sparsely-gated-moe-2017.md
│   ├── gshard-2020.md
│   ├── switch-transformer-2021.md
│   ├── vision-moe-2021.md
│   ├── glam-2022.md
│   ├── st-moe-2022.md
│   ├── expert-choice-2022.md
│   ├── soft-moe-2023.md
│   └── deepseek-moe-2024.md
├── models/             # Production MoE model pages
│   ├── mixtral.md
│   ├── deepseek.md
│   ├── grok.md
│   ├── dbrx.md
│   ├── arctic.md
│   ├── jamba.md
│   ├── olmoe.md
│   ├── llama4.md
│   └── qwen-moe.md
└── people/             # Key researchers
    └── key-researchers.md
```

## Page Conventions

- **Title**: H1 at top of every page
- **Cross-references**: Use relative links `[Switch Transformer](../papers/switch-transformer-2021.md)`
- **Paper metadata**: Each paper page starts with a metadata block (title, authors, year, venue, arxiv link)
- **Model metadata**: Each model page includes parameter counts (total / active), expert count, routing method
- **Tags**: Each page ends with a `## See Also` section linking related pages
- **Citations**: `[Author et al., Year]` inline, full references at bottom

## Workflows

### Ingest
When processing a new source: read it, extract key claims, update relevant wiki pages, add cross-references, append to `log.md`, update `index.md`.

### Query
When answering a question: search relevant wiki pages, synthesize answer with citations. If the answer reveals a gap, create a new page.

### Lint
Periodically check for: contradictions between pages, stale claims, orphaned pages (no inbound links), missing cross-references, incomplete metadata.
