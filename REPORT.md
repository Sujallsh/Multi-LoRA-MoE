# Multi-LoRA with Mixture-of-Experts for Multi-Task Text Classification

## Objective

This project studies whether a learned Mixture-of-Experts router improves multi-task text classification when each task has its own low-rank feature adapter.

## Datasets

| Task | Dataset | Classes |
|---|---|---:|
| News topic classification | AG News | 4 |
| Emotion classification | Emotion | 6 |
| Ontology classification | DBpedia 14 | 14 |

The training script uses balanced samples so the largest dataset does not dominate optimization.

## Method

The frozen `google/bert_uncased_L-2_H-128_A-2` encoder produces a pooled representation. Three feature-space LoRA experts learn low-rank residual transformations. A router predicts a sparse distribution over the experts and combines their representations. Each task has a separate classification head because the tasks have different label sets.

Training has three stages:

1. Train each LoRA expert on its associated task.
2. Freeze the experts and train the router on balanced batches from all tasks.
3. Jointly fine-tune the experts, router, and classification heads with a smaller learning rate.

The router receives task supervision during training. It infers the task from the text during inference, while the task name selects the correct output head.

## Evaluation

Report accuracy and macro F1 for every task. The implementation compares:

- `uniform`: averages all expert features equally.
- `moe`: uses the learned sparse router.
- `task_expert`: selects the known task expert as an oracle reference.

Copy the values from `outputs/cpu_demo/comparison.json` after training.

| Routing method | AG News macro F1 | Emotion macro F1 | DBpedia macro F1 | Average macro F1 |
|---|---:|---:|---:|---:|
| Uniform experts | 0.8298 | 0.2532 | 0.8833 | 0.6554 |
| Learned MoE | **0.8311** | 0.2576 | **0.9231** | **0.6706** |
| Task expert oracle | 0.8271 | **0.2588** | 0.9205 | 0.6688 |

The learned router improved average macro F1 by 0.0152 over uniform expert averaging. The largest improvement occurred on DBpedia, where macro F1 increased by approximately 0.0398. AG News changed only slightly, and Emotion remained the hardest task. The small frozen encoder and a single expert-training epoch likely limited emotion performance.

The final validation accuracies for learned MoE routing were 0.834 for AG News, 0.500 for Emotion, and 0.928 for DBpedia. Router usage stayed balanced across the evaluation set: 0.335 for AG News, 0.337 for Emotion, and 0.328 for DBpedia.

## Discussion

- Did learned routing outperform uniform expert averaging?
- Which task benefited most from specialization?
- Did router usage match the expected task distribution?
- How many parameters were trained compared with the complete encoder?
- What errors arose from using a tiny frozen encoder?

## Limitations

This implementation applies LoRA to pooled encoder features instead of injecting adapters into every transformer attention layer. This choice makes the experiment practical on a CPU and keeps the MoE routing logic inspectable. A larger reproduction could use PEFT adapters inside the attention layers, a larger base model, and unsupervised routing.
