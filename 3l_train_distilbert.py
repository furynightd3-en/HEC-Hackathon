print(">>> Model Training <<<")

import os, time, numpy as np, torch
from datasets import load_dataset, ClassLabel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

SEED = 42
LABELS = ["Low", "Medium", "High"]
id2label = {0: "Low", 1: "Medium", 2: "High"}
label2id = {"Low": 0, "Medium": 1, "High": 2}

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

print("RUNNING:", os.path.abspath(__file__))
print("Labels we want to use:", LABELS)
print("\n=== Step 2: Loading CSVs and enforcing 3 labels ===")

from datasets import load_dataset, ClassLabel
import numpy as np

# Load CSVs (expects columns: text,label)
raw = load_dataset("csv", data_files={"train": "train_clean.csv", "test": "test.csv"})

# Force the label column to match our 3-class schema
raw = raw.cast_column("label", ClassLabel(names=LABELS))

# 90/10 stratified split for validation
split = raw["train"].train_test_split(test_size=0.10, seed=SEED, stratify_by_column="label")
train_ds, val_ds, test_ds = split["train"], split["test"], raw["test"]

def summarize(ds, name):
    y = np.array(ds["label"])
    uniq, counts = np.unique(y, return_counts=True)
    print(f"\n{name} split:")
    for u, c in zip(uniq, counts):
        print(f"  id={u} -> {id2label[int(u)]:<6} count={c}")
    missing = set(range(3)) - set(uniq.tolist())
    if missing:
        print("  (labels not present in this split):", [id2label[m] for m in sorted(missing)])

# Show summaries for each split
summarize(train_ds, "TRAIN")
summarize(val_ds,   "VAL")
summarize(test_ds,  "TEST")
print("\n=== Step 3: Tokenizing datasets ===")

tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tok(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize, batched=True).rename_column("label","labels")
val_ds   = val_ds.map(tokenize, batched=True).rename_column("label","labels")
test_ds  = test_ds.map(tokenize, batched=True).rename_column("label","labels")

cols = ["input_ids", "attention_mask", "labels"]
for ds in (train_ds, val_ds, test_ds):
    ds.set_format(type="torch", columns=cols)

print("Tokenization complete. Example:")
print(train_ds[0])
print("\n=== Step 4: Creating DistilBERT model (3 labels) ===")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

print("Model created with num_labels =", model.config.num_labels)
print("id2label mapping:", model.config.id2label)
print("label2id mapping:", model.config.label2id)
print("\n=== Step 5: Training setup ===")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\nValidation Confusion Matrix (rows=true, cols=pred):", flush=True)
    print("               Pred Low   Pred Medium   Pred High", flush=True)
    print(f"True Low      {cm[0][0]:>8}   {cm[0][1]:>11}   {cm[0][2]:>9}", flush=True)
    print(f"True Medium   {cm[1][0]:>8}   {cm[1][1]:>11}   {cm[1][2]:>9}", flush=True)
    print(f"True High     {cm[2][0]:>8}   {cm[2][1]:>11}   {cm[2][2]:>9}", flush=True)

    return {
        "accuracy": acc,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }

from transformers import TrainingArguments, Trainer
stamp = time.strftime("%Y%m%d_%H%M%S")
out_dir   = f"./results_{stamp}"
final_dir = f"./distilbert_3label_priority_{stamp}"

args = TrainingArguments(
    output_dir=out_dir,
    eval_strategy="epoch",    # may warn, but still works on your version
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir=f"{out_dir}/logs",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("Trainer initialized. Output dir:", out_dir)
print("\n=== Step 6: Training ===")
trainer.train()

print("\n=== Step 6b: Evaluation on validation set ===")
print(trainer.evaluate())

print("\n=== Step 6c: Final test on hold-out set ===")
pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=-1)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred, labels=[0,1,2]))

print("\n=== Step 7: Saving final model ===")
model.config.num_labels = 3
model.config.id2label = id2label
model.config.label2id = label2id

model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)

print("Model saved to:", os.path.abspath(final_dir))

# Verify immediately
from transformers import DistilBertForSequenceClassification as M
m = M.from_pretrained(final_dir)
print("Loaded back from disk. num_labels =", m.config.num_labels)
print("id2label:", m.config.id2label)
print("label2id:", m.config.label2id)
