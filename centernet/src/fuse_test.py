import torch
import torch.nn as nn
import torchvision as tv
from fused_module import *


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x


def fuse(conv, bn):
    w = conv.weight.float()
    mean = bn.running_mean.float()
    var_sqrt = torch.sqrt(bn.running_var.float() + bn.eps)

    beta = bn.weight.float()
    gamma = bn.bias.float()

    if conv.bias is not None:
        b = conv.bias.float()
    else:
        b = mean.new_zeros(mean.shape).float()

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = QConv2d_16(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, IdentityBN):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, QConv2d_16):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m):
    p = torch.randn([1, 3, 224, 224])
    import time
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)

    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    # assert(o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


def test_layer():
    p = torch.randn([1, 3, 112, 112])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())


# m = tv.models.resnet152(True)
# m.eval()
# print("Layer level test: ")
# test_layer()

# print("============================")
# print("Module level test: ")
# m = tv.models.resnet18(True)
# m = resnet101(pretrained=True)
# m.eval()
# test_net(m)