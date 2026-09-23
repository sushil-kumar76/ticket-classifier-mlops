"""Create a tiny randomly-initialised model so CI can test the API contract
without downloading or training the real one."""
import argparse
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaConfig,
)

p = argparse.ArgumentParser()
p.add_argument("--out", default="models/final")
args = p.parse_args()

labels_file = Path("data/raw/labels.txt")
labels = (labels_file.read_text().splitlines() if labels_file.exists()
          else [f"intent_{i}" for i in range(77)])

config = RobertaConfig(
    vocab_size=50265, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=2, intermediate_size=128, max_position_embeddings=130,
    num_labels=len(labels),
    id2label=dict(enumerate(labels)),
    label2id={name: i for i, name in enumerate(labels)})

model = AutoModelForSequenceClassification.from_config(config)
tok = AutoTokenizer.from_pretrained("distilroberta-base")

Path(args.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print(f"stub model written to {args.out} ({len(labels)} labels)")
