import onnx
from onnxconverter_common import float16
model = onnx.load('ckpoints/wav2lip288_fp32.onnx')
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, "ckpoints/wav2lip288_fp16.onnx")
