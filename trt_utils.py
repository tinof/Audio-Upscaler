import torch
import tensorrt as trt
from torch2trt import torch2trt

def convert_to_trt(model, input_shape=(1, 2, 48000)):
    """Convert AudioSR model to TensorRT"""
    x = torch.randn(input_shape).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,  # Enable FP16 for RTX 4070
        max_workspace_size=1 << 30,  # 1GB workspace
        max_batch_size=1
    )
    return model_trt

def save_trt_model(model_trt, path):
    torch.save(model_trt.state_dict(), path)

def load_trt_model(path, model):
    model_trt = torch2trt(model, [torch.randn(1, 2, 48000).cuda()])
    model_trt.load_state_dict(torch.load(path))
    return model_trt 