# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/16 16:01
import os

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # 初始化pycuda 不可删除
import pycuda.driver as cuda


EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

# import torch
# from typing import Union, Optional, Sequence, Dict, Any
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.distributed.rpc.internal import deserialize


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    return engine


def engine_infer(engine, input_image, audio_feat):
    """
    engine: load_engine函数返回的trt模型引擎
    input_image: 模型推理输入图像，尺寸为(batch_size, channel, height, width)
    output：Unet模型推理的结果，尺寸为(batch_size, class_num, height, width)
    """
    print(input_image.dtype, audio_feat.dtype)
    # np.copyto(host_inputs[0], input_image.ravel())
    # np.copyto(host_inputs[1], audio_feat.ravel())
    batch_size = input_image.shape[0]
    image_channel = input_image.shape[1]
    image_height = input_image.shape[2]
    image_width = input_image.shape[3]
    # input_image = input_image.ravel()
    # audio_feat = audio_feat.ravel()
    # Allocate host and device buffers
    # with engine.create_execution_context() as context:
    # Set input shape based on image dimensions for inference
    # print(context.set_input_shape("input", (1, 6, 160, 160)))
    # print(context.set_input_shape("audio", (1, 32, 32, 32)))
    # outputs = []
    bindings = []
    cuda_inputs = []
    host_inputs = []
    host_outputs = []
    cuda_outputs = []
    for binding in engine:
        print(binding)
        binding_idx = engine[binding]
        size = trt.volume(engine.get_tensor_shape(binding))
        # dtype = trt.nptype(engine.get_tensor_dtype(binding))

        # bindings.append(int(cuda_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            # host_mem = cuda.pagelocked_empty(size, dtype=np.float32)
            # cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            if binding == "input":
                input_item = input_image.astype(np.float32)
                host_mem = np.ascontiguousarray(input_item)
                # host_mem = input_item.astype(np.float32).ravel()
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                # np.copyto(host_inputs[0], input_item.ravel())
            elif binding == "audio":
                input_item = audio_feat
                # host_mem = input_item.ravel()
                host_mem = np.ascontiguousarray(input_item)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                # np.copyto(host_inputs[1], input_item.ravel())
            bindings.append(int(cuda_mem))
        else:
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
            # print(f"output_memory:{output_memory}")
            bindings.append(int(cuda_mem))
    print(bindings)
    # Transfer input data to the GPU.
    stream = cuda.Stream()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    cuda.memcpy_htod_async(cuda_inputs[1], host_inputs[1], stream)
    # Run inference
    context = engine.create_execution_context()
    # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    context.execute_v2(bindings=bindings)
    # Transfer prediction output from the GPU.
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    # Synchronize the stream
    stream.synchronize()

    output = np.reshape(host_outputs[0], (3, 160, 160))
    output = output.transpose(1, 2, 0) * 255
    output = output.astype(np.uint8)
    cv2.imwrite("tttt.jpg", output)
    # output = output_buffer
    return output


if __name__ == '__main__':
    engine_path = "zyz_0128_3_fp32.trt"
    engine = load_engine(engine_path)
    image = cv2.imread("test_2.jpg")
    img_masked = cv2.rectangle(image, (5,5,150,145),(0,0,0),-1)
    image = np.transpose(image, (2, 0, 1)) / 255.0
    img_masked = np.transpose(img_masked, (2, 0, 1)) / 255.0
    img_T = np.concatenate((image, img_masked), axis=0)
    img_T = np.expand_dims(img_T, axis=0)
    print(img_T.shape)
    audio_feat = np.load("16_frame_of_muted_audio.npy")
    audio_feat = audio_feat.reshape(1, 32, 32, 32)
    print(audio_feat.shape)
    print(img_T.dtype)
    engine_infer(engine, img_T, audio_feat)
    del engine