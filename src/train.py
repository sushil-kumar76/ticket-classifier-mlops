import argparse
import numpy as np
import pandas as pd
import mlflow
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

p = argparse.ArgumentParser()
p.add_argument("--model", default="distilroberta-base")
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--batch", type=int, default=32)
args = p.parse_args()

labels = open("data/raw/labels.txt").read().splitlines()
train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

tok = AutoTokenizer.from_pretrained(args.model)
def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=64, padding="max_length")

full = Dataset.from_pandas(train_df).train_test_split(test_size=0.1, seed=42)
train_ds = full["train"].map(tokenize, batched=True)
val_ds = full["test"].map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels),
    id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})

def metrics(pred):
    y = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, y),
            "f1_macro": f1_score(pred.label_ids, y, average="macro")}

mlflow.set_experiment("ticket-classifier")
with mlflow.start_run():
    mlflow.log_params(vars(args))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="models/checkpoints",
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            fp16=True,
            report_to="none"),
        train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=metrics)

    trainer.train()

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    mlflow.log_metrics({k: v for k, v in test_metrics.items() if isinstance(v, float)})
    print(test_metrics)

    trainer.save_model("models/final")
    tok.save_pretrained("models/final")
    mlflow.log_artifacts("models/final", artifact_path="model")
