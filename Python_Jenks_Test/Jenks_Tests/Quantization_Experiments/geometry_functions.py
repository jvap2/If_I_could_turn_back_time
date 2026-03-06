import torch
import torch.nn as nn

def get_conv_layers(model):
    """
    Returns list of (name, module) for all convolution layers.
    """

    conv_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    return conv_layers


def conv_fft_spectrum(conv_layer):
    """
    Compute average FFT energy spectrum of convolution filters.
    """

    W = conv_layer.weight.data

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


def radial_energy(spectrum):
    """
    Compute radial frequency energy distribution.
    """

    h, w = spectrum.shape

    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing='ij'
    )

    center = h // 2

    r = torch.sqrt((x-center)**2 + (y-center)**2)
    r = r.long()

    max_r = r.max()

    energy = torch.zeros(max_r+1)

    for i in range(max_r+1):
        mask = r == i
        energy[i] = spectrum[mask].mean()

    return energy

def uniform_quantize(weights,bitwidth, per_channel=True):
        qmin = -(2**(bitwidth-1))
        qmax = (2**(bitwidth-1))-1
        if per_channel:
            if weights.ndim == 4:
                max_val = weights.abs().amax(dim=(1,2,3), keepdim=True)
            elif weights.ndim ==2:
                max_val = weights.abs().amax(dim=1, keepdim=True)
            else:
                raise ValueError("Bad")
        else:
            max_val = weights.abs().max()
        scale = max_val/qmax

def spectral_distortion(conv_layer, bitwidth):

    W = conv_layer.weight.data
    Wq = uniform_quantize(W, bitwidth)

    oc, ic, k, _ = W.shape
    errors = []

    for o in range(oc):
        for i in range(ic):

            w = W[o, i]
            wq = Wq[o, i]

            fft_w = torch.fft.fftshift(torch.fft.fft2(w))
            fft_q = torch.fft.fftshift(torch.fft.fft2(wq))

            error = torch.abs(fft_q - fft_w)**2
            errors.append(error)

    errors = torch.stack(errors)

    return errors.mean(dim=0)