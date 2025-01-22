# -*- coding: utf-8 -*-            
# @Author : Dony YUAN
# @Time : 2024/11/21 13:23
# -*- coding: utf-8 -*-
# @Author : Dony YUAN
# @Time : 2024/11/19 17:29
""" cuda 操作使用torch代替pycuda """
import threading
import cv2
import time
import numpy as np
import tensorrt as trt
import torch
from tqdm import tqdm
import atexit

EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTEngine(object):
    def __init__(self, engine_file, ctx="hold"):
        if ctx is None:
            atexit.register(self.__del__)
        self.batch = 1
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.engine = self.init_engine(engine_file)
        self.stream = torch.cuda.Stream()
        self.context = self.engine.create_execution_context()

    def do_inference(self, image, audio):
        # image = cv2.imread("/usr/src/tensorrt/data/resnet50/airliner.ppm")
        # image = (2.0 / 255.0) * image.transpose((2, 0, 1)) - 1.0
        # with torch.no_grad():
        # self.cuda_inputs[0].copy_(image.ravel())
        # self.cuda_inputs[1].copy_(audio.ravel())
        self.cuda_inputs[0].copy_(torch.from_numpy(image.ravel()))
        self.cuda_inputs[1].copy_(torch.from_numpy(audio.ravel()))
        # self.cuda_inputs[0].copy_(self.host_inputs[0], non_blocking=True)
        # self.cuda_inputs[1].copy_(self.host_inputs[1], non_blocking=True)
        self.context.execute_v2(self.bindings)
        # self.cuda_outputs[0].copy_(self.host_outputs[0], non_blocking=True)
        self.host_outputs[0].copy_(self.cuda_outputs[0], non_blocking=True)
        self.stream.synchronize()
        # s = time.time()
        pred = self.host_outputs[0].numpy()
        # print("6666", pred)
        pred = pred.reshape((3, 160, 160)).transpose(1, 2, 0) * 255
        pred = pred.astype(np.uint8)
        # print(time.time() - s)
        return pred

    def init_engine(self, engine_path):
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        for binding in engine:
            size = trt.volume(engine.get_tensor_shape(binding)) * self.batch
            # dtype = trt.nptype(engine.get_tensor_dtype(binding))
            tensor_cuda = torch.empty(size, dtype=torch.float32, device='cuda')
            self.bindings.append(tensor_cuda.data_ptr())
            # print(f"dtype:{dtype}-size:{size}, self.bindings:{self.bindings}")
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                # self.host_inputs.append(torch.empty(size, dtype=torch.float32, device='cpu'))
                self.cuda_inputs.append(tensor_cuda)
            else:
                self.host_outputs.append(torch.empty(size, dtype=torch.float32, device='cpu'))
                self.cuda_outputs.append(tensor_cuda)
        return engine

    def __del__(self):
        print(f"Cleaning up on exit...")
        if hasattr(self, 'context'):
            # pass
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        print(f"Cleaning up is done!")

def test_main():
    engine_path = "zyz_0128_3_fp32.trt"
    image = cv2.imread("test_2.jpg")
    img_masked = cv2.rectangle(image, (5,5,150,145),(0,0,0),-1)
    image = np.transpose(image, (2, 0, 1))
    print(image.dtype)
    img_masked = np.transpose(img_masked, (2, 0, 1))
    img_T = np.concatenate((image / 255.0, img_masked / 255.0), axis=0, dtype=np.float32)
    img_T = np.expand_dims(img_T, axis=0)
    # img_real_ex_T = torch.from_numpy(image / 255.0)
    # img_masked_T = torch.from_numpy(img_masked / 255.0)
    # img_T = torch.cat([img_real_ex_T, img_masked_T], dim=0)[None]
    print(img_T.shape)
    # audio_feat = torch.from_numpy(np.load("16_frame_of_muted_audio.npy"))
    audio_feat = np.load("16_frame_of_muted_audio.npy")
    audio_feat = audio_feat.reshape(1, 32, 32, 32)
    print(audio_feat.shape)
    print(f"img_T:{img_T.dtype}, audio_feat:{audio_feat.dtype}")
    engine = TRTEngine(engine_path)
    for i in tqdm(range(1), ncols=100):
        output = engine.do_inference(img_T, audio_feat)
        # print(output)
        cv2.imwrite("tttt.jpg", output)
    print("===========")

if __name__ == "__main__":
    test_main()
    # th = threading.Thread(target=test_main)
    # th.start()
    # th.join()