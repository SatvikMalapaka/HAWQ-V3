import torch
import torch.nn as nn
import numpy as np
def hutchinson(model, inputs, targets, num_samples=10):
    traces = {}
    model.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    for (name, param), grad in zip(model.named_parameters(), grads):
        if not param.requires_grad:
            continue

        num_params = param.numel()
        layer_trace = 0.0

        for _ in range(num_samples):
            # Rademacher vector (+1/-1) 
            v = torch.randint_like(param, low=0, high=2).float() * 2 - 1
            Hv = torch.autograd.grad(
                grad, param, grad_outputs=v, retain_graph=True
            )[0]

            layer_trace += torch.sum(v * Hv).item()

        trace_estimate = layer_trace / num_samples
        normalized_trace = trace_estimate / num_params

        traces[name] = {
            "normalized_trace": normalized_trace,
            "num_params": num_params
        }

    return traces
