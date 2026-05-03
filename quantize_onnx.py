from onnxruntime.quantization import quantize_dynamic, QuantType

inp = "distilbert_3label_priority_20250915_224504/distilbert_priority.onnx"
out = "distilbert_3label_priority_20250915_224504/distilbert_priority_int8.onnx"

quantize_dynamic(
    model_input=inp,
    model_output=out,
    weight_type=QuantType.QInt8   # quantize weights to int8
)

print("Wrote:", out)
