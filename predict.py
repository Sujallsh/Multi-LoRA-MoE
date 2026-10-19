import argparse

import torch
from transformers import AutoTokenizer

from multilora_moe.data import TASKS
from multilora_moe.model import MultiLoRAMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/cpu_demo/model.pt")
    parser.add_argument("--task", required=True, choices=["ag_news", "emotion", "dbpedia"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--routing", default="moe", choices=["moe", "uniform", "task_expert"])
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MultiLoRAMoE.from_checkpoint(checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["config"]["model_name"])
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs, task_name=args.task, routing=args.routing)
    probabilities = torch.softmax(output["logits"], dim=-1)[0]
    label_id = int(probabilities.argmax())
    print(f"predicted_label={label_id}")
    print(f"predicted_class={TASKS[args.task].label_names[label_id]}")
    print(f"confidence={float(probabilities.max()):.4f}")
    print("router=" + ", ".join(
        f"{name}:{float(weight):.3f}"
        for name, weight in zip(model.task_names, output["router_weights"][0])
    ))


if __name__ == "__main__":
    main()
