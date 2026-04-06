# Fine-Tuning MoE Models

Fine-tuning Mixture of Experts models requires special consideration because the sparse routing structure can be disrupted by standard fine-tuning approaches.

## Challenges

### Routing Disruption
- Pre-trained routing patterns may not transfer well to downstream tasks
- Fine-tuning can collapse routing to a few experts, wasting capacity
- Small fine-tuning datasets don't provide enough signal for stable routing

### Expert Forgetting
- Updating all expert parameters can cause catastrophic forgetting
- Experts that specialized in general knowledge may be overwritten with task-specific patterns
- The sparse activation pattern means each expert sees even fewer fine-tuning examples

### Load Imbalance Amplification
- Task-specific data may activate certain experts disproportionately
- Load balancing losses calibrated for pre-training may be wrong for fine-tuning
- Smaller batches during fine-tuning make routing statistics noisier

## Approaches

### Full Fine-Tuning

Update all parameters (router + experts + attention):
- Works for large fine-tuning datasets
- Risk of expert collapse on small datasets
- Expensive — requires memory for all expert gradients
- Recommendation: reduce learning rate 2-5× vs. dense fine-tuning; increase load balance coefficient

### Router-Only Fine-Tuning

Freeze expert weights, only train the router:
- Cheap and fast
- Useful when task distribution differs from pre-training
- Limited capacity for learning new knowledge
- Good as a first step before full fine-tuning

### Expert Subset Fine-Tuning

Select a subset of experts to fine-tune, freeze the rest:
- Middle ground between full and router-only fine-tuning
- Can select based on utilization (fine-tune most-used experts)
- Preserves knowledge in frozen experts

### LoRA for MoE

Apply [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) to MoE models:
- Add LoRA adapters to expert FFN layers and/or attention
- Much lower memory footprint than full fine-tuning
- Can apply different LoRA ranks to different experts
- **MoLoRA**: Apply LoRA only to the top-k most utilized experts
- Effective in practice and widely used for Mixtral fine-tuning

### Task-Specific Expert Addition

Add new expert(s) for each downstream task:
- Keep pre-trained experts frozen
- Train new task-specific expert(s) + update router
- Enables multi-task serving with shared base experts
- Similar in spirit to adapter methods

## Instruction Tuning MoE

For chat/instruction-following fine-tuning (e.g., Mixtral → Mixtral-Instruct):

- Use supervised fine-tuning (SFT) on instruction data
- Follow with DPO/RLHF alignment as with dense models
- Key finding: MoE models respond well to instruction tuning despite routing concerns
- [Mixtral-8x7B-Instruct](../models/mixtral.md) demonstrates strong instruction-following capability

## Practical Recommendations

1. **Start with smaller learning rates** (1e-5 or lower for full fine-tuning)
2. **Monitor expert utilization** throughout fine-tuning — watch for collapse
3. **Use LoRA** as the default approach unless you have a large dataset and compute
4. **Increase load balance loss coefficient** slightly during fine-tuning
5. **Larger batch sizes** help stabilize routing during fine-tuning
6. **Evaluate routing statistics** before and after fine-tuning to verify expert diversity

## See Also

- [Sparse Upcycling](sparse-upcycling.md)
- [Training Challenges](training.md)
- [Mixtral](../models/mixtral.md)
- [Overview](overview.md)
