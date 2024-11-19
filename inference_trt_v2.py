# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/19 17:29
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from tqdm import tqdm

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

batch = 1
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []


def Inference(engine, image, audio):
    # image = cv2.imread("/usr/src/tensorrt/data/resnet50/airliner.ppm")
    # image = (2.0 / 255.0) * image.transpose((2, 0, 1)) - 1.0
    print(image.dtype, audio.dtype)
    np.copyto(host_inputs[0], image.ravel())
    np.copyto(host_inputs[1], audio.ravel())
    stream = cuda.Stream()
    context = engine.create_execution_context()
    # for i in tqdm(range(10000)):

    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    cuda.memcpy_htod_async(cuda_inputs[1], host_inputs[1], stream)
    context.execute_v2(bindings)  # 速度比 execute_async_v2 略快👍👍🏻👍🏼
    # context.execute_async_v2(bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    print("execute times "+str(time.time()-start_time))

    output = host_outputs[0]
    print(output.shape)
    # print(np.argmax(output))
    output = np.reshape(output, (3, 160, 160))
    output = output.transpose(1, 2, 0) * 255
    print(output.dtype)
    output = output.astype(np.uint8)
    cv2.imwrite("tttt.jpg", output)

def PrepareEngine(engine_path):
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    # create buffer
    for binding in engine:
        print(binding)
        size = trt.volume(engine.get_tensor_shape(binding)) * batch
        host_mem = cuda.pagelocked_empty(size, dtype=np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
            print(binding, "11", host_mem.shape, host_mem.dtype)
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    return engine


if __name__ == "__main__":

    engine_path = "zyz_0128_3_fp32.trt"
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
    engine = PrepareEngine(engine_path)
    print(img_T.dtype)
    Inference(engine, img_T, audio_feat)

    engine = []  # 或 del engine 进行释放,否则cuda报错
