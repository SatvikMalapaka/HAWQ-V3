import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm 
from torch import optim

def evaluate(model, dataloader, device):
    model.eval()
    top1, total = 0, 0
    model = model.to(device)
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total += labels.size(0)

            preds = outputs.argmax(dim=1)
            top1 += (preds == labels).sum().item()

    top1_acc = 100 * top1 / total
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    return top1_acc

def uniform_quantize(tensor, num_bits):
    if num_bits == 32:
        return tensor

    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)

    q_tensor = torch.round(tensor / (scale + 1e-8) + zero_point)
    q_tensor.clamp_(qmin, qmax)
    dq_tensor = (q_tensor - zero_point) * scale
    return dq_tensor


def uniform_quantize_asymmetric(tensor, num_bits):
    if num_bits == 32:
        return tensor

    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val, max_val = tensor.min(), tensor.max()

    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = qmin - min_val / (scale + 1e-8)
    zero_point = torch.clamp(torch.round(zero_point), qmin, qmax)

    q_tensor = torch.round(tensor / (scale + 1e-8) + zero_point)
    q_tensor.clamp_(qmin, qmax)
    dq_tensor = (q_tensor - zero_point) * scale

    return dq_tensor

class QuantizedConv2d(nn.Module):
    def __init__(self, conv_layer, w_bits=8, a_bits=8):
        super().__init__()
        self.conv = conv_layer
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        # Quantize input activation
        x_q = uniform_quantize_asymmetric(x, self.a_bits)

        # Quantize weights
        w_q = uniform_quantize(self.conv.weight, self.w_bits)

        return nn.functional.conv2d(
            x_q,
            w_q,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, w_bits=8, a_bits=8):
        super().__init__()
        self.linear = linear_layer
        self.w_bits = w_bits
        self.a_bits = a_bits

    def forward(self, x):
        # Quantize activations
        x_q = uniform_quantize_asymmetric(x, self.a_bits)

        # Quantize weights
        w_q = uniform_quantize(self.linear.weight, self.w_bits)

        # Perform quantized linear operation
        return F.linear(x_q, w_q, self.linear.bias)


def quantize_model(model, w_bits=8, a_bits=8):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, QuantizedConv2d(module, w_bits, a_bits))
        elif isinstance(module, nn.Linear):
            setattr(model, name, QuantizedLinear(module, w_bits, a_bits))
        else:
            quantize_model(module, w_bits, a_bits)
    return model

def quantize_model_layerwise(model, w_bits_dict, a_bits=8, parent_name=""):
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, nn.Conv2d):
            bits = w_bits_dict.get(full_name, 32)
            print(f"Quantizing Conv2d {full_name} → {bits}-bit")
            setattr(model, name, QuantizedConv2d(module, w_bits=bits, a_bits=a_bits))

        elif isinstance(module, nn.Linear):
            bits = w_bits_dict.get(full_name, 32)
            print(f"Quantizing Linear {full_name} → {bits}-bit")
            setattr(model, name, QuantizedLinear(module, w_bits=bits, a_bits=a_bits))

        else:
            quantize_model_layerwise(module, w_bits_dict, a_bits=a_bits, parent_name=full_name)

    return model

def fine_tune(
    model,
    train_loader,
    val_loader=None,
    num_epochs=5,
    lr=1e-4,
    device=None,
    save_best=True,
):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss/len(train_loader.dataset):.4f} | Train Acc: {train_acc:.2f}%")

        # Optional validation
        if val_loader is not None:
            val_acc = evaluate_model_loss(model, val_loader, criterion, device)
            if save_best and val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_finetuned_quantized.pth")

    if val_loader is not None:
        print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    return model


def evaluate_model_loss(model, dataloader, criterion, device):
    model.eval()
    total, correct, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Val Loss: {val_loss/len(dataloader.dataset):.4f} | Val Acc: {acc:.2f}%")
    return acc


def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise ValueError("Expected (nn.Conv2d, nn.BatchNorm2d)")

    # Prepare params / device / dtype
    W = conv.weight.data.clone()
    device = W.device
    dtype = W.dtype

    if conv.bias is not None:
        b = conv.bias.data.clone().to(device=device, dtype=dtype)
    else:
        b = torch.zeros(conv.out_channels, device=device, dtype=dtype)

    # BatchNorm params (handle case bn.affine==False)
    if bn.affine:
        gamma = bn.weight.data.to(device=device, dtype=dtype)
        beta = bn.bias.data.to(device=device, dtype=dtype)
    else:
        gamma = torch.ones(conv.out_channels, device=device, dtype=dtype)
        beta = torch.zeros(conv.out_channels, device=device, dtype=dtype)

    mean = bn.running_mean.to(device=device, dtype=dtype)
    var = bn.running_var.to(device=device, dtype=dtype)
    eps = bn.eps

    # Compute folding
    std = torch.sqrt(var + eps)
    if W.dim() == 4:
        factor = (gamma / std).reshape([-1, 1, 1, 1])
    else:
        factor = (gamma / std).reshape([-1, 1])

    W_folded = W * factor
    b_folded = beta + (b - mean) * (gamma / std)

    # Create a new conv with same configuration but ensure bias=True
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=True
    ).to(device=device, dtype=dtype)

    fused_conv.weight.data.copy_(W_folded)
    fused_conv.bias.data.copy_(b_folded)

    return fused_conv

def fuse_module_conv_bn(module: nn.Module) -> nn.Module:
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        module._modules[name] = fuse_module_conv_bn(child)

    prev_name = None
    prev_mod = None
    for name, child in list(module._modules.items()):
        if isinstance(prev_mod, nn.Conv2d) and isinstance(child, nn.BatchNorm2d):
            fused = fold_bn_into_conv(prev_mod, child)
            module._modules[prev_name] = fused
            module._modules[name] = nn.Identity()
            prev_name, prev_mod = None, None
            continue
        prev_name, prev_mod = name, child

    return module


# Straight-Through Estimator for rounding
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LearnableQuantizer(nn.Module):
    def __init__(self, num_bits=8, init_min=-1.0, init_max=1.0):
        super().__init__()
        self.num_bits = num_bits
        self.qmin = 0
        self.qmax = 2 ** num_bits - 1

        # Learnable parameters
        self.scale = nn.Parameter(torch.tensor((init_max - init_min) / (self.qmax - self.qmin)))
        self.zero_point = nn.Parameter(torch.tensor(-init_min / (self.scale + 1e-8)))

    def forward(self, x):
        x_q = STEQuantize.apply(x / (self.scale + 1e-8) + self.zero_point)
        x_q = torch.clamp(x_q, self.qmin, self.qmax)
        dq_x = (x_q - self.zero_point) * self.scale
        return dq_x

class QuantizedConv2dLearnableQuantizer(nn.Module):
    def __init__(self, conv_layer, w_bits=8, a_bits=8):
        super().__init__()
        self.conv = conv_layer
        self.w_quant = LearnableQuantizer(w_bits)
        self.a_quant = LearnableQuantizer(a_bits)

    def forward(self, x):
        # Quantize activations
        x_q = self.a_quant(x)
        # Quantize weights
        w_q = self.w_quant(self.conv.weight)
        return F.conv2d(
            x_q,
            w_q,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QuantizedLinearLearnableQuantizer(nn.Module):
    def __init__(self, linear_layer, w_bits=8, a_bits=8):
        super().__init__()
        self.linear = linear_layer
        self.w_quant = LearnableQuantizer(w_bits)
        self.a_quant = LearnableQuantizer(a_bits)

    def forward(self, x):
        x_q = self.a_quant(x)
        w_q = self.w_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)

def quantize_model_layerwise_LearnableQuantizer(model, w_bits_dict, a_bits=8, parent_name=""):
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, nn.Conv2d):
            bits = w_bits_dict.get(full_name, 32)
            print(f"Quantizing Conv2d {full_name} → {bits}-bit")
            setattr(model, name, QuantizedConv2dLearnableQuantizer(module, w_bits=bits, a_bits=a_bits))

        elif isinstance(module, nn.Linear):
            bits = w_bits_dict.get(full_name, 32)
            print(f"Quantizing Linear {full_name} → {bits}-bit")
            setattr(model, name, QuantizedLinearLearnableQuantizer(module, w_bits=bits, a_bits=a_bits))

        else:
            quantize_model_layerwise_LearnableQuantizer(module, w_bits_dict, a_bits=a_bits, parent_name=full_name)

    return model

def recalibrate_bn(model, loader, device, num_batches=10):
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
    print(f"Recalibrated BN with {num_batches} batches.")
