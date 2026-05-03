import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass
class ClassifierConfig:
    onnx_model_path: str
    tokenizer_name: str
    labels: List[str]


class DistilBertOnnxClassifier:
    """
    DistilBERT (or similar) ONNX text classifier.
    - Handles tokenization in Python.
    - Runs ONNX inference using onnxruntime.
    """
    def __init__(self, cfg: ClassifierConfig):
        if not os.path.exists(cfg.onnx_model_path):
            raise FileNotFoundError(
                f"ONNX model not found at: {cfg.onnx_model_path} "
                f"(put your model there or set ONNX_MODEL_PATH)"
            )
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        self.session = ort.InferenceSession(cfg.onnx_model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        # Figure out output name (could be 'logits', 'output_0', etc.)
        self.output_name = self.session.get_outputs()[0].name

        # Map required inputs
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict(self, text: str) -> Dict[str, Any]:
        # Basic sanity
        text = (text or "").strip()
        if len(text) < 3:
            return {"label": "Low", "score": 0.0, "probs": {l: 0.0 for l in self.cfg.labels}}

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="np",
        )
        ort_inputs = {}
        if "input_ids" in self.input_names:
            ort_inputs["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in self.input_names:
            ort_inputs["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in self.input_names and "token_type_ids" in enc:
            ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        logits = self.session.run([self.output_name], ort_inputs)[0]  # shape: (1, num_labels)
        logits = np.asarray(logits).reshape(-1)
        probs = softmax(logits)

        idx = int(np.argmax(probs))
        label = self.cfg.labels[idx] if idx < len(self.cfg.labels) else str(idx)
        score = float(probs[idx])

        return {"label": label, "score": score, "probs": {self.cfg.labels[i]: float(probs[i]) for i in range(len(self.cfg.labels))}}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)
