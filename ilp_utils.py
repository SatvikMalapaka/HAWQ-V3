import numpy as np
import torch
import torch.nn as nn

def quantize_tensor(tensor, num_bits):
    #Uniform symmetric since weights. Activations would've been asymmetric
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    max_val = tensor.abs().max()
    if max_val == 0:
        return torch.zeros_like(tensor)

    scale = max_val / qmax
    tensor_q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    tensor_deq = tensor_q * scale
    return tensor_deq

#Finding the layerwise l2 norm of quantised and unquantised model
def l2_diff_conv_linear(model, bit_a=8, device="cpu"):
    l2_diffs = {}

    for name, module in model.named_modules():
        # Handle Conv2d and Linear layers. Ignore rest cuz less params (BN will be merged)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data.to(device)
            w_q_a = quantize_tensor(w, bit_a)

            diff = w_q_a - w
            l2_norm = torch.norm(diff, p=2).item()
            l2_diffs[name] = l2_norm

    return l2_diffs

def get_conv_bitops(model, input_size=(1, 3, 32, 32), w_bits=4, a_bits=8, device="cpu"):
    model = model.to(device)
    model.eval()

    bitops_dict = {}
    layer_out_shapes = {}

    def hook_fn(name):
        def hook(module, input, output):
            layer_out_shapes[name] = output.shape
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Run random input with same shape as image
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(dummy_input)

    for h in hooks:
        h.remove()

    # Compute BitOps
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out_shape = layer_out_shapes[name]
            Cout, Hout, Wout = out_shape[1], out_shape[2], out_shape[3]
            Cin = module.in_channels
            Kh, Kw = module.kernel_size

            macs = Hout * Wout * Cin * Kh * Kw * Cout
            bitops = macs * w_bits * a_bits
            bitops_dict[name] = bitops

        elif isinstance(module, nn.Linear):
            out_features = module.out_features
            in_features = module.in_features

            macs = in_features * out_features
            bitops = macs * w_bits * a_bits
            bitops_dict[name] = bitops

    return bitops_dict

