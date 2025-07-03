"""
train.py – fine-tune distilroberta-base on our tokenised Arrow splits.

Run (repo root, venv active):
    python -m api.model.train --epochs 3 --batch 8 --lr 2e-5
"""

import argparse, pathlib, torch, evaluate
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ──────────────────────────────────────────────────────────────────────
def main(epochs: int, batch: int, lr: float):

    # 1. load the DatasetDict and rename label column inside every split
    ds: DatasetDict = load_from_disk("data/fake_news_splits")
    for split in ds.keys():
        if "label_bin" in ds[split].column_names:
            ds[split] = ds[split].rename_column("label_bin", "labels")

    # 2. model head for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=2
    )

    # 3. macro-F1 metric
    f1 = evaluate.load("f1")

    def compute(metrics):
        preds = metrics.predictions.argmax(-1)
        return f1.compute(predictions=preds, references=metrics.label_ids)

    # 4. training configuration  (eval_strategy is the new name)
    args = TrainingArguments(
        output_dir="api/model/ckpts",
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),          # mixed precision on GPU
    )

    # 5. train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute,
    )
    trainer.train()

    # 6. save best checkpoint
    out_dir = pathlib.Path("api/model/fakenews_roberta")
    trainer.save_model(out_dir)
    print("🎉 fine-tuned model saved to", out_dir)

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    a = p.parse_args()
    main(a.epochs, a.batch, a.lr)