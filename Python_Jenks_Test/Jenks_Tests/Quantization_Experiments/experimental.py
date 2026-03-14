import torch
import torch.nn as nn
import math

def conv_operator_sketch(block_fp, block_q, input_shape, probes=32):

    device = next(block_fp.parameters()).device

    loss = 0
    for (name_fp, layer_fp), (name_q, layer_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):
        if isinstance(layer_fp, torch.nn.Conv2d) and isinstance(layer_q, torch.nn.Conv2d):
            for _ in range(probes):

                x = torch.randn(input_shape, device=device)

                y = block_fp(x)
                yq = block_q(x)

                loss += ((y - yq)**2).mean()

    return loss / probes



def generate_fourier_probe(shape, freq_x, freq_y):

    B, C, H, W = shape

    x = torch.arange(W).float()
    y = torch.arange(H).float()

    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

    probe = torch.sin(
        2 * math.pi * (
            freq_x * grid_x / W +
            freq_y * grid_y / H
        )
    )

    probe = probe.unsqueeze(0).unsqueeze(0)
    probe = probe.repeat(B, C, 1, 1)

    return probe

def fourier_probe_loss(block_fp, block_q, freqs=None):

    loss = 0
    if freqs==None:
        freqs = [
    (0,0),
    (1,0),
    (0,1),
    (2,0),
    (0,2),
    (1,1),]  
    for (name_fp, layer_fp), (name_q, layer_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):
        if isinstance(layer_fp, torch.nn.Conv2d) and isinstance(layer_q, torch.nn.Conv2d):
            shape = layer_fp.shape
            for fx, fy in freqs:

                probe = generate_fourier_probe(shape, fx, fy).to(layer_fp.weight.device)

                with torch.no_grad():
                    y_fp = layer_fp(probe)

                y_q = layer_q(probe)

                loss += torch.mean((y_q - y_fp)**2)

    return loss / len(freqs)

def gram_operator_loss(conv_fp, conv_q):

    W = conv_fp.weight
    Wq = conv_q.weight

    W = W.view(W.shape[0], -1)
    Wq = Wq.view(Wq.shape[0], -1)

    G = W @ W.T
    Gq = Wq @ Wq.T

    loss = ((G - Gq)**2).mean()

    return loss

def gram_operator_loss_blocks(block_fp, block_q):
    for (name_fp, layer_fp), (name_q, layer_q) in zip(
        block_fp.named_modules(), block_q.named_modules()
    ):
        if (isinstance(layer_fp, torch.nn.Conv2d) and isinstance(layer_q, torch.nn.Conv2d)) or (isinstance(layer_fp, torch.nn.Linear) and isinstance(layer_q, torch.nn.Linear)):
            W = layer_fp.weight
            Wq = layer_q.weight_quantizer()

            W = W.view(W.shape[0], -1)
            Wq = Wq.view(Wq.shape[0], -1)

            G = W @ W.T
            Gq = Wq @ Wq.T

            loss += ((G - Gq)**2).mean()

    return loss