import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def conv_fft_spectrum(conv_layer):

    W = conv_layer.weight.data.clone()

    # shape: (out_channels, in_channels, k, k)
    oc, ic, k, _ = W.shape

    spectra = []

    for o in range(oc):
        for i in range(ic):

            kernel = W[o, i]

            fft = torch.fft.fft2(kernel)
            fft = torch.fft.fftshift(fft)

            energy = torch.abs(fft)**2
            spectra.append(energy)

    spectra = torch.stack(spectra)

    return spectra.mean(dim=0)


def visualize_spectrum(spectrum):

    plt.imshow(spectrum.cpu().numpy())
    plt.colorbar()
    plt.title("Average Convolution Filter Spectrum")
    plt.show()


def simulate_quantization(weights, bitwidth):

    w = weights.clone()

    qmin = -(2**(bitwidth-1))
    qmax = (2**(bitwidth-1)) - 1

    scale = w.abs().max() / qmax

    q = torch.round(w / scale)
    q = torch.clamp(q, qmin, qmax)

    wq = q * scale

    return wq


def frequency_error(conv_layer, bitwidth):

    W = conv_layer.weight.data

    Wq = simulate_quantization(W, bitwidth)

    oc, ic, k, _ = W.shape

    errors = []

    for o in range(oc):
        for i in range(ic):

            w = W[o,i]
            wq = Wq[o,i]

            fft_w = torch.fft.fftshift(torch.fft.fft2(w))
            fft_q = torch.fft.fftshift(torch.fft.fft2(wq))

            error = torch.abs(fft_q - fft_w)**2

            errors.append(error)

    return torch.stack(errors).mean(dim=0)


def get_first_conv(model):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m

    raise ValueError("No Conv2d layer found")

def test_vis(model, filename):
    conv = get_first_conv(model)
    spectrum = conv_fft_spectrum(conv)
    error = frequency_error(conv, bitwidth=2)

    plt.subplot(1,2,1)
    plt.title("Filter Energy Spectrum")
    plt.imshow(spectrum.cpu())
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title("Quantization Error Spectrum")
    plt.imshow(error.cpu())
    plt.colorbar()  
    plt.show()
    plt.savefig(filename)