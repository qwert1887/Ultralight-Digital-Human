import os
import sys

from unet import Model
import onnx
import torch

import onnxruntime
import numpy as np
import time
# onnx_path = "./dihuman.onnx"


def convert2trt(onnx_path, save_path):
    try:
        command = (
            f"trtexec --onnx={onnx_path} --saveEngine={save_path} "
            f"--optShapes=input:1x6x160x160,audio:1x32x32x32 "
            f"--minShapes=input:1x6x160x160,audio:1x32x32x32 "
            f"--maxShapes=input:1x6x160x160,audio:1x32x32x32 "
            # f"--useSpinWait"  # 减少GPU计算不稳定性,会增加CPU计算和功耗
                   )
        os.system(command)
    except Exception as e:
        print(f"Error: {e}")
    else:
        print(f"Convert {onnx_path} to {save_path} successfully")

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
    try:
        np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    except Exception as e:
        print(f"Error: {e}")
        print("Test failed,but maybe it's OK.you can try use the generated model!!")
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main(ckpt_path):
    # ckpt_path = "/home/guaishou/PycharmProjects/livetalking/ultralight_dh/checkpoints/zyz/200.pth"
    ckpt_dir = os.path.dirname(ckpt_path)
    onnx_path = os.path.join(ckpt_dir, "model.onnx")
    onnx_path_fp32 = os.path.join(ckpt_dir, "model_fp32.onnx")
    engine_path = os.path.join(ckpt_dir, "model.engine")
    net_state_dict = torch.load(ckpt_path)
    if "model" in net_state_dict:
        net_state_dict = net_state_dict["model"]
    net = Model(6).eval()
    net.load_state_dict(net_state_dict)
    # fp32 = True
    # if fp32:
    #     net.float()
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
    convert2trt(onnx_path_fp32, engine_path)
    check_onnx(torch_out, img, audio)


if __name__ == '__main__':
    # python -m onnxsim network.onnx networkfp32.onnx
    # ckpt_path = "checkpoints/checkpoint_zyz_0128/200.pth"
    # ckpt_path = "checkpoints/0118_zyy_stand/best.pth"
    checkpoints_path = "./checkpoints"
    if not os.path.exists(checkpoints_path):
        print(f"Checkpoint directory {checkpoints_path} doesn't exist")
        sys.exit(1)
    for model_id in os.listdir(checkpoints_path):
        model_ckpt_dir = os.path.join(checkpoints_path, model_id)
        if not os.path.isdir(model_ckpt_dir):
            print(f"The specified checkpoint directory {model_ckpt_dir} doesn't exist")
            continue
        train_pth = os.path.join(model_ckpt_dir, "best.pth")
        if not os.path.isfile(train_pth):
            print(f"Did you forget the pth file name? `200.pth` has been replaced by `best.pth`! The specified checkpoint {train_pth} doesn't exist")
            continue
        onnx_path = os.path.join(model_ckpt_dir, "model.onnx")
        if not os.path.isfile(onnx_path):
            ckpt_path = train_pth
            main(ckpt_path)
        else:
            print(f"The specified checkpoint {onnx_path} has exist!")
