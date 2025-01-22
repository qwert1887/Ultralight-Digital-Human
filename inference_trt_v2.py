# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/19 17:29
import threading

import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from tqdm import tqdm
import atexit

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine(object):

    def __init__(self, engine_file, ctx=None):
        if ctx is None:
            atexit.register(self.__del__)
        self.batch = 1
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.engine = self.init_engine(engine_file)
        self.stream = cuda.Stream()
        self.context = self.engine.create_execution_context()


    def do_inference(self, image, audio):
        # image = cv2.imread("/usr/src/tensorrt/data/resnet50/airliner.ppm")
        # image = (2.0 / 255.0) * image.transpose((2, 0, 1)) - 1.0
        # print(image.dtype, audio.dtype)
        # stream = cuda.Stream()
        # context = self.engine.create_execution_context()
        # for i in tqdm(range(10000)):

        np.copyto(self.host_inputs[0], image.ravel())
        # start_time = time.time()
        np.copyto(self.host_inputs[1], audio.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        cuda.memcpy_htod_async(self.cuda_inputs[1], self.host_inputs[1], self.stream)
        self.context.execute_v2(self.bindings)  # 速度比 execute_async_v2 略快👍👍🏻👍🏼
        # context.execute_async_v2(bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        # s = time.time()
        # print(f"execute times {str(time.time()-start_time)}")
        pred = self.host_outputs[0]
        # print(output.shape)
        # print(np.argmax(output))
        pred = np.reshape(pred, (3, 160, 160))
        pred = pred.transpose(1, 2, 0) * 255
        # print(output.dtype)
        pred = pred.astype(np.uint8)
        # print(time.time() - s)
        return pred

    def init_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        # create buffer
        for binding in engine:
            # print(binding)
            size = trt.volume(engine.get_tensor_shape(binding)) * self.batch
            host_mem = cuda.pagelocked_empty(size, dtype=np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
                print(binding, "11", host_mem.shape, host_mem.dtype)
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return engine

    def __del__(self):
        print(f"Cleaning up on exit...")
        # if hasattr(self, 'engine'):
        #     del self.engine
        if hasattr(self, 'context'):
            del self.context
        print(f"Cleaning up is done!")


def test_main():
    cuda.init()
    ctx = cuda.Device(0).make_context()
    engine_path = "zyz_0128_3_fp32.trt"
    image = cv2.imread("test_2.jpg")
    img_masked = cv2.rectangle(image, (5,5,150,145),(0,0,0),-1)
    image = np.transpose(image, (2, 0, 1)) / 255.0
    print(image.dtype)
    img_masked = np.transpose(img_masked, (2, 0, 1)) / 255.0
    img_T = np.concatenate((image, img_masked), axis=0, dtype=np.float32)
    img_T = np.expand_dims(img_T, axis=0)
    print(img_T.shape)
    audio_feat = np.load("16_frame_of_muted_audio.npy")
    audio_feat = audio_feat.reshape(1, 32, 32, 32)
    print(audio_feat.shape)
    print(f"img_T:{img_T.dtype}, audio_feat:{audio_feat.dtype}")
    engine = TRTEngine(engine_path, ctx=ctx)
    for i in tqdm(range(2000), ncols=100):
        # print(img_T.dtype)
        # img_T = img_T.astype(np.float32)
        output = engine.do_inference(img_T, audio_feat)
        # del engine
        # engine = PrepareEngine(engine_path)
        # Inference(engine, img_T, audio_feat)
        # output = np.reshape(output, (3, 160, 160))
        # output = output.transpose(1, 2, 0) * 255
        # print(output.dtype)
        # output = output.astype(np.uint8)
        cv2.imwrite("tttt.jpg", output)
    print("===========")
    # time.sleep(3)
    ctx.pop()
    # engine = []  # 或 del engine 进行释放,否则cuda报错


if __name__ == "__main__":
    th = threading.Thread(target=test_main)
    th.start()
    th.join()

