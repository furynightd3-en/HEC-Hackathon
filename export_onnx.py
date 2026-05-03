import os, torch, onnx
from transformers import AutoTokenizer, DistilBertForSequenceClassification

MODEL_DIR = r"./distilbert_3label_priority_20250915_224504"
OUT = os.path.join(MODEL_DIR, "distilbert_priority.onnx")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).eval()

dummy = tok(["hello"], return_tensors="pt", padding=True, truncation=True, max_length=128)
args = (dummy["input_ids"], dummy["attention_mask"])

torch.onnx.export(
    mdl, args, OUT,
    input_names=["input_ids","attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids":{0:"batch",1:"seq"},
                  "attention_mask":{0:"batch",1:"seq"},
                  "logits":{0:"batch"}},
    opset_version=17,
    do_constant_folding=True
)

onnx.checker.check_model(onnx.load(OUT))
print("OK ->", OUT)
