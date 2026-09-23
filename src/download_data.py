from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

out = Path("data/raw")

out.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(f"{BASE}/train.csv")

test = pd.read_csv(f"{BASE}/test.csv")

labels = sorted(train["category"].unique())

label2id = {name: i for i, name in enumerate(labels)}

for df, name in [(train, "train"), (test, "test")]:

    df["label"] = df["category"].map(label2id)

    df[["text", "label"]].to_csv(out / f"{name}.csv", index=False)

(out / "labels.txt").write_text("\n".join(labels))

print(f"train={len(train)}, test={len(test)}, labels={len(labels)}")

