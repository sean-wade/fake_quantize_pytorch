'''
这是 风兴 的量化模型导出onnx代码，需要注意的事项有：
    1、这个代码一般应该不会用到，我已经将 onnx/txt/npy 导出合到一份代码里了：
        quantize_export_onnx_txt_compare_v3tiny_V1.py
       这个代码仅做保留

    2、这个代码的主要功能是：
        a. 将训练好的量化模型加载进来
        b. 新建一个常规 yolo 模型，将量化模型的 state_dict 赋予 这个常规模型
        c. 导出这个常规模型
        注意：
            导出的只是onnx模型，缺少量化表，需要配合 export_txt 进行量化表导出（export_txt功能参考）
'''
import onnx
import torch
import torch.onnx
from onnxsim import simplify
from collections import OrderedDict

from models.yolo import Model

from quantize_utils import torch2quant, load_yolo_from_q_ckpt, fix_state_dict
from qqquantize.savehook import register_intermediate_hooks


DEVICE = "cuda"
CONFIG = "models/yolov3-tiny.yaml"
CKPT_PATH = "weights/voc_tiny_quant_fix2_perchannel_0.416.pt"
ONNX_SAVE = "weights/voc_tiny_quant_fix2_perchannel_0.416.onnx"


def recreate_model(new_hook_data, cfg=CONFIG, ch=3, nc=20, device=DEVICE):
    # recreate model
    new_model = Model(cfg, ch, nc).to(device)
    new_model.model[-1].export = True
    new_state_dict = new_model.state_dict()
    for k in list(new_state_dict.keys()):
        if 'bn' in k:
            if 'running_mean' not in k:
                if 'running_var' not in k:
                    if 'num_batches_tracked' not in k:
                        v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
                        new_state_dict[k] = v.view(-1)
        else:
            if 'running_mean' not in k:
                if 'running_var' not in k:
                    if 'num_batches' not in k:
                        if 'anchors' not in k:
                            if 'anchor_grid' not in k:
                                v = torch.tensor(new_hook_data[f'{k}_quant']['values'][0])
                                new_state_dict[k] = v
    new_model.load_state_dict(new_state_dict)
    return new_model


if __name__ == '__main__':
    
    model = load_yolo_from_q_ckpt(CKPT_PATH, device=DEVICE, cfg=CONFIG, ch=3, nc=20)
    hook = register_intermediate_hooks(model)
    
    test_img = torch.zeros(1, 3, 416, 416).to(DEVICE)
    feat = model(test_img)
    hook_data = hook.output_data()
    new_hook_data = fix_state_dict(hook_data)
    new_model = recreate_model(new_hook_data)

    torch.onnx.export(new_model, 
                      test_img, 
                      ONNX_SAVE, 
                      verbose=False, 
                      opset_version=10, 
                      input_names=['image'], 
                      output_names=['layer1', 'layer2'], 
                      training=True)
    
    # m = onnx.load(ONNX_SAVE)
    # model_simp, check = simplify(m)
    # onnx.save(model_simp, 
    #           ONNX_SAVE.replace(".onnx", "_simple.onnx"))
    # assert check, "Simplified ONNX model could not be validated"
