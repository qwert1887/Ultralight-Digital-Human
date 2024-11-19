import os

from unet import Model
import onnx
import torch

import onnxruntime
import numpy as np
import time
# onnx_path = "./dihuman.onnx"

def check_onnx(torch_out, torch_in, audio):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    import onnxruntime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {ort_session.get_inputs()[0].name: torch_in.cpu().numpy(), ort_session.get_inputs()[1].name: audio.cpu().numpy()}
    for i in range(1):
        t1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t2 = time.time()
        print("onnx time cost::", t2 - t1)

    np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# python -m onnxsim network.onnx networkfp32.onnx
ckpt_path = "checkpoints/checkpoint_zyz_0128/200.pth"
ckpt_dir = os.path.dirname(ckpt_path)
onnx_path = os.path.join(ckpt_dir, "model.onnx")
onnx_path_fp32 = os.path.join(ckpt_dir, "model_fp32.onnx")
net_state_dict = torch.load(ckpt_path)
if "model" in net_state_dict:
    net_state_dict = net_state_dict["model"]
net = Model(6).eval()
net.load_state_dict(net_state_dict)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# img = torch.zeros([1, 6, 160, 160], dtype=torch.float32).to(device).contiguous()
# audio = torch.zeros([1, 32, 32, 32], dtype=torch.float32).to(device).contiguous()
img = torch.zeros([1, 6, 160, 160], dtype=torch.float32).contiguous()
audio = torch.zeros([1, 32, 32, 32], dtype=torch.float32).contiguous()

input_dict = {"input": img, "audio": audio}
dynamic = True
if dynamic:
    dynamic_axes = {
        "input": {0: "batch_size"},
        "audio": {0: "batch_size"}
    }
else:
    dynamic_axes = None

with torch.no_grad():
    torch_out = net(img, audio)
    # print(torch_out.shape)
    torch.onnx.export(
        net,
        (img, audio),
        onnx_path,
        input_names=['input', "audio"],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        # example_outputs=torch_out,
        do_constant_folding=False,
        opset_version=16,
        export_params=True
    )
# 同步生成int32格式。为了tensorrt使用
os.system(f"python -m onnxsim {onnx_path} {onnx_path_fp32} --no-large-tensor")

check_onnx(torch_out, img, audio)
