import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas
from torch.autograd import grad
from torch import sigmoid
import copy
import gc
from custom_schedulers import Min_OBJ_Lr_Scheduler

def CrossLayerEqualization(model, input_shape):
    """
    Applies Cross-Layer Equalization to the given model.

    Args:
        model (torch.nn.Module): The neural network model to be equalized.
        input_shape (tuple): The shape of the input tensor.
    Returns:
        torch.nn.Module: The equalized model.
    """
    model.eval()
    with torch.no_grad():
        pass  # Placeholder for actual Cross-Layer Equalization logic
    return model

def WeightRangeAdjustment(model):
    pass

def QuantNetwork(model, quant_bias = False, act_bit_width=8, weight_bit_width=4):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)):
                module.quant_weight()
                if quant_bias:
                    module.quant_bias()
    return model


def compute_input_covariance(X):
    """
    X: (d, N)
    Returns C = X X^T
    """
    return X @ X.T


def output_operator(Delta, C):
    """
    Computes H_O Δ = 2 Δ C
    """
    return 2 * Delta @ C


def gram_operator(Delta, W):
    """
    Computes H_G Δ = 4(Δ W W^T + W Δ^T W)
    """
    WWt = W @ W.T
    return 4 * (Delta @ WWt + W @ Delta.T @ W)


def output_loss(Delta, X):
    return torch.norm(Delta @ X, p='fro')**2


def gram_loss(Delta, W):
    return torch.norm(W @ Delta.T + Delta @ W.T, p='fro')**2


#############################################
# Rayleigh Quotient Optimization
#############################################

def solve_generalized_eigen(W, X,
                            steps=300,
                            lr=1e-2,
                            device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Minimizes Rayleigh quotient:
    ||ΔX||^2 / ||WΔ^T + ΔW^T||^2

    Returns:
        Delta_star
        smallest_eigenvalue
        diagnostics
    """

    W = W.to(device)
    X = X.to(device)

    Delta = torch.randn_like(W, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([Delta], lr=lr)

    history = []

    for step in range(steps):

        out = output_loss(Delta, X)
        gram = gram_loss(Delta, W)

        rq = out / (gram + 1e-12)

        optimizer.zero_grad()
        rq.backward()
        optimizer.step()

        # normalize for stability (fix Gram norm)
        with torch.no_grad():
            gram_norm = gram_loss(Delta, W).sqrt()
            Delta /= (gram_norm + 1e-12)

        history.append(rq.item())

    smallest_eigenvalue = history[-1]

    diagnostics = {
        "rayleigh_history": history,
        "final_output_loss": output_loss(Delta, X).item(),
        "final_gram_loss": gram_loss(Delta, W).item(),
    }

    return Delta.detach(), smallest_eigenvalue, diagnostics


def rayleigh_objective(W, X):
    Delta = torch.zeros_like(W, requires_grad=True)

    out = torch.norm(Delta @ X, p='fro')**2
    gram = torch.norm(W @ Delta.T + Delta @ W.T, p='fro')**2

    R = out / (gram + 1e-12)
    return R



def rayleigh_wrt_W(W, X, Delta_star):
    W = W.clone().detach().requires_grad_(True)

    out = torch.norm(Delta_star @ X, p='fro')**2
    gram = torch.norm(W @ Delta_star.T + Delta_star @ W.T, p='fro')**2

    R = out / (gram + 1e-12)

    R.backward()
    return W.grad

def geometry_aware_round_fast(W, X, Delta_star, scale):
    grad = rayleigh_wrt_W(W, X, Delta_star)

    q_down = torch.floor(W / scale) * scale
    q_up   = torch.ceil(W / scale) * scale

    eps_down = q_down - W
    eps_up   = q_up - W

    cost_down = grad * eps_down
    cost_up   = grad * eps_up

    mask = cost_down < cost_up
    W_q = torch.where(mask, q_down, q_up)

    return W_q


def find_scale_output_optimal(W, X, bitwidth, grid_points=50):
    max_val = W.abs().max()
    scales = torch.linspace(max_val*0.1, max_val, grid_points)

    best_s = None
    best_loss = float('inf')

    for s in scales:
        q = torch.round(W / s).clamp(
            -(2**(bitwidth-1)),
             2**(bitwidth-1)-1
        ) * s

        loss = torch.norm(W @ X - q @ X, p='fro')**2

        if loss < best_loss:
            best_loss = loss
            best_s = s

    return best_s


def custom_round_layer(module, calibration_loader, lambda_gram=0.1):

    device = next(module.parameters()).device
    W = module.weight.detach()
    s = module.weight_quant.scale().detach()

    # Collect input covariance
    inputs = []
    def hook(module, inp, out):
        inputs.append(inp[0].detach())

    handle = module.register_forward_hook(hook)

    # run few batches
    module.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            x = x.to(device)
            module(x)
            if i > 10:
                break

    handle.remove()

    X = torch.cat(inputs, dim=0)
    X = X.view(X.size(0), X.size(1), -1).mean(-1).T
    Cx_diag = torch.diag(X @ X.T)

    Wn = W / s
    floor = torch.floor(Wn)
    ceil = torch.ceil(Wn)

    candidates = torch.stack([floor, ceil], dim=0)

    best_choice = torch.zeros_like(Wn)

    row_norm = torch.norm(W, dim=1)**2

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):

            scores = []
            for k in range(2):

                r = candidates[k, i, j]
                delta = s * r - W[i, j]

                score = delta**2 * (
                    Cx_diag[j] + lambda_gram * row_norm[i]
                )

                scores.append(score.item())

            best_choice[i, j] = candidates[
                torch.argmin(torch.tensor(scores)), i, j
            ]

    W_quant = s * best_choice

    module.weight.data.copy_(W_quant)


    import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import brevitas
from brevitas.nn import QuantLinear, QuantConv2d






# ============================================================
# 2. SYLVESTER SOLVER (RELAXED PROBLEM)
# ============================================================

# def solve_sylvester_relaxed(W, Wq, Cx, lambda_reg=1e-4):
#     """
#     Solve:
#         (W Cx W^T + λI) ΔW + ΔW (Cx + λI) = -2 (W - Wq)
#     """

#     device = W.device

#     # Ensure contiguity everywhere
#     W = W.contiguous()
#     Wq = Wq.contiguous()
#     Cx = Cx.contiguous()

#     A = W @ Cx @ W.T
#     B = Cx

#     A = A + lambda_reg * torch.eye(A.shape[0], device=device)
#     B = B + lambda_reg * torch.eye(B.shape[0], device=device)

#     A = A.contiguous()
#     B = B.contiguous()

#     RHS = -2 * (W - Wq)
#     RHS = RHS.contiguous()

#     m, n = W.shape

#     I_m = torch.eye(m, device=device).contiguous()
#     I_n = torch.eye(n, device=device).contiguous()

#     # VERY IMPORTANT: make B.T contiguous
#     B_T = B.T.contiguous()

#     K = torch.kron(I_n, A) + torch.kron(B_T, I_m)
#     rhs = RHS.reshape(-1)

#     delta = torch.linalg.solve(K, rhs)
#     delta = delta.reshape(m, n)

#     return delta

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
#                 BASIC UNIFORM QUANTIZATION
# ============================================================

def symmetric_uniform_quantize_tensor(W: torch.Tensor, bitwidth: int):
    """
    Symmetric per-tensor uniform quantization.

    Args:
        W: weight tensor
        bitwidth: number of bits (e.g. 2, 4, 8)

    Returns:
        W_q: quantized tensor
        scale: quantization scale
    """
    if bitwidth < 1:
        raise ValueError("Bitwidth must be >= 1")

    qmax = 2 ** (bitwidth - 1) - 1
    max_val = W.abs().max()

    if max_val == 0:
        return W.clone(), torch.tensor(1.0, device=W.device)

    scale = max_val / qmax
    W_int = torch.round(W / scale)
    W_int = torch.clamp(W_int, -qmax, qmax)

    W_q = W_int * scale
    return W_q, scale

def symmetric_uniform_quantize_network(model, bitwidth=4):
    """
    Applies symmetric uniform quantization to all Conv2d and Linear layers in the model.
    Memory-efficient version with progress tracking.
    """
    import gc
    
    layer_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_count += 1
    
    current = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                current += 1
                print(f"Quantizing {name} at {bitwidth}-bit ({current}/{layer_count})")
                
                # Quantize weights
                W = module.weight.data
                W_q, scale = symmetric_uniform_quantize_tensor(W, bitwidth)
                module.weight.data.copy_(W_q)
                del W, W_q, scale
                
                # Quantize bias if present
                if hasattr(module, 'bias') and module.bias is not None:
                    b = module.bias.data
                    b_q, _ = symmetric_uniform_quantize_tensor(b, bitwidth)
                    module.bias.data.copy_(b_q)
                    del b, b_q
                
                # Clear cache every few layers to prevent memory overflow
                if current % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
# ============================================================
#          RELAXED SYLVESTER SOLVER (GEOMETRY STEP)
# ============================================================

# def solve_sylvester_relaxed(W, Wq, Cx, lambda_reg=1e-4):
#     """
#     Solves:

#         λΔ + (W+Δ) Cx = Wq Cx

#     Rearranged:

#         λΔ + Δ Cx = (Wq - W) Cx

#     Which is:

#         AΔ + ΔB = C

#     with:
#         A = λ I
#         B = Cx
#         C = (Wq - W) Cx

#     Uses vectorized Kronecker solve.
#     """

#     device = W.device
#     m, n = W.shape

#     A = lambda_reg * torch.eye(m, device=device)
#     B = Cx
#     C = (Wq - W) @ Cx

#     I_m = torch.eye(m, device=device)
#     I_n = torch.eye(n, device=device)

#     # Ensure contiguous
#     A = A.contiguous()
#     B = B.contiguous()

#     # Kronecker system
#     K = torch.kron(I_n, A) + torch.kron(B.T.contiguous(), I_m)
#     C_vec = C.reshape(-1, 1)

#     delta_vec = torch.linalg.solve(K, C_vec)
#     delta = delta_vec.reshape(m, n)

#     return delta

def solve_sylvester_relaxed(W, Wq, Cx, lambda_reg=1e-4):
    device = W.device
    n = Cx.shape[0]
    I = torch.eye(n, device=device)

    RHS = (Wq - W) @ Cx

    delta = torch.linalg.solve(
        lambda_reg * I + Cx,
        RHS.T
    ).T

    return delta

# ============================================================
#         INPUT COVARIANCE ESTIMATION (LATE LAYERS)
# ============================================================

@torch.no_grad()
def estimate_input_covariance(model, layer, dataloader, device, max_batches=1):
    """
    Estimate Cx = E[x x^T] for the TRUE input of a layer
    by running full forward passes and capturing activations.
    Uses only 1 batch by default to save memory.
    """
    
    inputs_list = []

    def hook(module, input, output):
        x = input[0].detach()

        if isinstance(module, nn.Conv2d):
            x = F.unfold(
                x,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            x = x.transpose(1, 2)
            x = x.reshape(-1, x.shape[-1])
        else:
            x = x.reshape(x.shape[0], -1)

        inputs_list.append(x.cpu())  # Keep on CPU to save GPU memory

    handle = layer.register_forward_hook(hook)

    model.eval()
    model.to(device)

    batch_count = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            model(images)  # <-- FULL MODEL FORWARD
            batch_count += 1
            torch.cuda.empty_cache()
            if batch_count >= max_batches:
                break

    handle.remove()

    if len(inputs_list) == 0:
        raise RuntimeError("No activations captured for covariance estimation.")

    X = torch.cat(inputs_list, dim=0).to(device)
    Cx = (X.T @ X) / X.shape[0]
    
    del inputs_list, X
    torch.cuda.empty_cache()
    gc.collect()

    return Cx


@torch.no_grad()
def estimate_input_quant_covariance(model, quant_model, layer, quant_layer, dataloader, device, max_batches=1):
    ''' In this function there are three matrices we need to compute:
    1) XX^T
    2) X Xq^T
    3) Xq Xq^T
    We have computed the current quantized input in our function that calls this and it is saved in quant_input
    '''
    
    inputs_list = []
    quant_inputs_list = []
    def make_hook(storage_list):
        def hook(module, input, output):
            x = input[0].detach()

            if isinstance(module, nn.Conv2d):
                x = F.unfold(
                    x,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                x = x.transpose(1, 2)
                x = x.reshape(-1, x.shape[-1])
            else:
                x = x.reshape(x.shape[0], -1)

            storage_list.append(x.cpu())  # Keep on CPU to save GPU memory
        return hook

    handle = layer.register_forward_hook(make_hook(inputs_list))
    quant_handle = quant_layer.register_forward_hook(make_hook(quant_inputs_list))
    model.eval()
    model.to(device)
    quant_model.eval()
    quant_model.to(device)
    batch_count = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            model(images)  # <-- FULL MODEL FORWARD
            quant_model(images)  # <-- FULL QUANT MODEL FORWARD
            batch_count += 1
            torch.cuda.empty_cache()
            if batch_count >= max_batches:
                break

    handle.remove()
    quant_handle.remove()
    if len(inputs_list) == 0:
        raise RuntimeError("No activations captured for covariance estimation.")
    
    X = torch.cat(inputs_list, dim=0).to(device)
    Xq = torch.cat(quant_inputs_list, dim=0).to(device)
    Cx = (X.T @ X) / X.shape[0]
    Cx_cross = (X.T @ Xq) / X.shape[0]
    Cx_q = (Xq.T @ Xq) / Xq.shape[0]

    return Cx, Cx_cross, Cx_q
# ============================================================
#         GEOMETRY-AWARE QUANTIZATION (PER LAYER)
# ============================================================

@torch.no_grad()
def geometry_aware_quantize_layer(
    model,
    layer,
    dataloader,
    device,
    bitwidth=4,
    lambda_reg=1e-4,
    use_geometry=True,
):
    """
    Quantizes a layer with optional geometry-aware refinement.
    """

    if not isinstance(layer, (nn.Linear, nn.Conv2d)):
        return

    W = layer.weight.data

    # Preserve pruning mask if present
    mask = None
    if hasattr(layer, "weight_mask"):
        mask = layer.weight_mask

    # Basic quantization
    W_q, _ = symmetric_uniform_quantize_tensor(W, bitwidth)

    if not use_geometry:
        layer.weight.data = W_q
        return

    # Estimate covariance
    Cx = estimate_input_covariance(model, layer, dataloader, device)

    # Reshape weights
    if isinstance(layer, nn.Conv2d):
        W_mat = W.reshape(W.shape[0], -1)
        Wq_mat = W_q.reshape(W.shape[0], -1)
    else:
        W_mat = W
        Wq_mat = W_q

    # Solve Sylvester
    delta = solve_sylvester_relaxed(W_mat, Wq_mat, Cx, lambda_reg)

    W_refined = W_mat + delta

    if isinstance(layer, nn.Conv2d):
        W_refined = W_refined.reshape_as(W)

    # Re-apply pruning mask if exists
    if mask is not None:
        W_refined = W_refined * mask

    layer.weight.data = W_refined


# ============================================================
#         APPLY TO FULL PYTORCH MODEL
# ============================================================

@torch.no_grad()
def apply_geometry_aware_quantization(
    model,
    dataloader,
    device,
    bitwidth=4,
    lambda_reg=1e-4,
    use_geometry=True,
):
    """
    Applies quantization to all Conv2d and Linear layers.
    """

    model.to(device)
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"Quantizing {name} at {bitwidth}-bit")
            geometry_aware_quantize_layer(
                model,
                module,
                dataloader,
                device,
                bitwidth=bitwidth,
                lambda_reg=lambda_reg,
                use_geometry=use_geometry,
            )

    return model

def snap_to_grid(W, scale, bitwidth):
    qmax = 2**(bitwidth-1) - 1
    qmin = -qmax
    W_int = torch.clamp((W / scale).round(), qmin, qmax)
    W_quantized = W_int * scale
    return W_quantized



# ============================================================
#         GEOMETRY-AWARE ROUNDING ATTEMPT 2
#         This method uses projected gradient steps
# ============================================================


def get_second_last_layer(model):

    quant_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    if len(quant_layers) == 0:
        raise ValueError("No Conv2d or Linear layers found.")

    return quant_layers[-2]  # second to last one so we can test the geometry-aware step on the last layer without it being affected by subsequent layers


def get_last_layer(model):
    quant_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    if len(quant_layers) == 0:
        raise ValueError("No Conv2d or Linear layers found.")

    return quant_layers[-1]  # return last two layers for geometry-aware step testing

def calculate_alpha(model, dataloader, device):

    model = model.to(device)
    model.eval()
    alpha = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                inputs = []
                def hook(module, inp, out):
                    inputs.append(inp[0].detach())

                handle = module.register_forward_hook(hook)

                # run few batches
                for i, (x, _) in enumerate(dataloader):
                    x = x.to(device)
                    model(x)
                    if i > 10:
                        break

                handle.remove()

                X = torch.cat(inputs, dim=0)

                # Build feature matrix X_feat of shape (num_samples, d_in)
                if isinstance(module, nn.Conv2d):
                    # X: (N, C, H, W) -> unfold to patches (N, C*kh*kw, L)
                    patches = F.unfold(
                        X,
                        kernel_size=module.kernel_size,
                        dilation=module.dilation,
                        padding=module.padding,
                        stride=module.stride,
                    )
                    # (N, C*kh*kw, L) -> (N*L, d_in)
                    patches = patches.transpose(1, 2).contiguous().view(-1, patches.size(1))
                    X_feat = patches
                else:
                    # Linear or flattened input: ensure (N, d_in)
                    if X.dim() > 2:
                        X_feat = X.view(X.size(0), -1)
                    else:
                        X_feat = X

                # Compute covariance Sigma = (X_feat^T X_feat) / Nsamples
                Nsamples = X_feat.size(0)
                if Nsamples == 0:
                    alpha[name] = 1.0
                    continue
                Sigma = (X_feat.t() @ X_feat) / float(Nsamples)

                # Flatten weight to (out, in)
                W = module.weight
                if W.dim() == 4:
                    W_flat = W.view(W.size(0), -1)
                else:
                    W_flat = W

                # If dimensions mismatch, skip this layer
                if W_flat.size(1) != Sigma.size(0):
                    alpha[name] = 1.0
                    continue

                M = W_flat @ Sigma @ W_flat.t()
                numerator = torch.norm(M, p='fro')**2
                denominator = torch.trace(M)**2
                alpha[name] = numerator / (denominator + 1e-12)
    '''Find the max alpha across layers and normalize all alphas by this max value to ensure they are in a reasonable range.'''    
    max_alpha = max(alpha.values())
    for key in alpha:
        alpha[key] /= max_alpha
    return alpha


def GenerateQ(model, bitwidth):
    Q = {}
    s = {}

    qmax = 2**(bitwidth - 1) - 1
    qmin = -2**(bitwidth - 1)

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):
            W = module.weight.data

            # Per-output-channel max
            max_val = W.abs().amax(dim=(1,2,3), keepdim=True)

            # Prevent division by zero
            max_val = torch.where(max_val == 0,
                                  torch.ones_like(max_val),
                                  max_val)

            scale = max_val / qmax

            Q[name] = torch.floor(W / scale)
            Q[name] = torch.clamp(Q[name], qmin, qmax)

            s[name] = scale


        elif isinstance(module, nn.Linear):
            W = module.weight.data

            # Per-output-neuron max
            max_val = W.abs().amax(dim=1, keepdim=True)

            max_val = torch.where(max_val == 0,
                                  torch.ones_like(max_val),
                                  max_val)

            scale = max_val / qmax

            Q[name] = torch.floor(W / scale)
            Q[name] = torch.clamp(Q[name], qmin, qmax)

            s[name] = scale

    return Q, s



class QuantizationObjective(nn.Module):
    """
    Implements:

    L(Theta) =
        tr(C Σ C^T
           - s(Theta Σ C^T + C Σ Theta^T)
           + s^2 Theta Σ Theta^T)
      + alpha * || W Σ C^T
                   - s W Σ Theta^T
                   + C Σ W^T
                   - s Theta Σ W^T ||_F^2

    We are going to try implementing a regularizer in order to encourage Theta to be close to binary values (0 or 1) to make it more likely that the geometry-aware step will snap weights to the correct quantization grid points. The regularizer can be something like:
    binary_reg = (Theta * (1 - Theta)).sum()
    """

    def __init__(self, W, Sigma, Q, s,
                 lamb=0.005,
                 last_layer=False,
                 beta=2):

        super().__init__()

        # ---- Force 2D weight matrix ----
        # Works for both Linear and Conv (already flattened outside ideally)
        assert W.dim() == 2
        assert Q.dim() == 2

        d_out, d_in = W.shape

        # ---- Shape safety checks ----
        assert Sigma.shape == (d_in, d_in), \
            f"Sigma must be ({d_in},{d_in}), got {Sigma.shape}"

        # ---- Per-channel scaling enforcement ----
        # s must be (d_out, 1)
        if s.dim() == 1:
            s = s.view(-1, 1)
        elif s.dim() == 0:
            s = s.view(1, 1).expand(d_out, 1)
        elif s.dim() == 4:              # Conv broadcast format
            s = s.view(s.size(0), 1)

        assert s.shape == (d_out, 1), \
            f"s must be ({d_out},1), got {s.shape}"

        # ---- Register tensors ----
        self.W = W
        self.Sigma = Sigma
        self.Q = Q
        self.s = s
        self.lamb_reg = 2e-4
        self.lamb = lamb if last_layer else 0.0
        self.beta = beta
        self.last_layer = last_layer

        # Precompute C = W - sQ  (broadcast safe)
        self.C = self.W - self.s * self.Q

        # Trainable parameter (logits)
        self.Theta = nn.Parameter(torch.zeros_like(self.W))
        # For experimentation, set Theta to be C/s
        with torch.no_grad():
            self.Theta.data = self.C / (self.s + 1e-12)  # Avoid division by zero
    def forward(self):

        W = self.W                    # (d_out, d_in)
        Sigma = self.Sigma            # (d_in, d_in)
        C = self.C                    # (d_out, d_in)
        s = self.s                    # (d_out, 1)
        Q = self.Q                    # (d_out, d_in)
        Theta = torch.clip(torch.sigmoid(self.Theta)*1.2 -.1, 0.0, 1.0)  # (d_out, d_in)

        d_out, d_in = W.shape

        activation_loss = torch.zeros(1, device=W.device)
        geom_loss = torch.zeros(1, device=W.device)

        # ---- Row-wise quadratic objective ----
        # for i in range(d_out):

        #     theta_i = Theta[i].unsqueeze(1)     # (d_in,1)
        #     C_i = C[i].unsqueeze(1)             # (d_in,1)
        #     s_i = s[i]                          # scalar

        #     # Linear term: -2 s_i θ_i^T C_i
        #     term1 = -2.0 * s_i * (theta_i.T @ C_i)

        #     # Quadratic term: s_i^2 θ_i^T Σ θ_i
        #     term2 = (s_i ** 2) * (theta_i.T @ Sigma @ theta_i)

        #     activation_loss += (term1 + term2).squeeze()  # Make sure it's a scalar

        # ThSig = Theta @ Sigma                          # (d_out, d_in)
        # term1 = -2.0*(s * (ThSig * C).sum(dim=1, keepdim=True)).sum()
        # term2 = (s**2 * (ThSig * Theta).sum(dim=1, keepdim=True)).sum()
        # # term3 should be trace(C @ Sigma @ C^T)
        # term3 = torch.trace(C @ Sigma @ C.t())
        # activation_loss = term1 + term2 + term3
        DeltaW = W - s * (Q + Theta)
        term = DeltaW @ Sigma @ DeltaW.t()
        activation_loss = torch.trace(term)
        print(f"Activation loss: {activation_loss.item()}")

        # ---- Binary regularizer ----
        # This was trial one
        # binary_reg = ((Theta * (1 - Theta)) ** 2).sum()
        # Trial 2 is Adarounds form of  regularizer which is 1-|2Theta-1|^\beta
        binary_reg = self.lamb_reg * (1 - torch.abs(2*Theta - 1) ** self.beta).sum()
        # print(f"Binary regularizer: {binary_reg.item()}")
        activation_loss += binary_reg

        # ---- Geometric term (row-wise safe) ----
        # geom_loss = torch.zeros(1, device=W.device)
        if self.last_layer:
            for i in range(d_out):

                theta_i = Theta[i].unsqueeze(1)
                W_i = W[i].unsqueeze(1)
                C_i = C[i].unsqueeze(1)
                s_i = s[i]

                K_i = W_i.T @ Sigma @ C_i + C_i.T @ Sigma @ W_i
                M_i = W_i.T @ Sigma @ theta_i + theta_i.T @ Sigma @ W_i

                geom_loss += torch.norm(K_i - s_i * M_i, p="fro") ** 2
        else:
            geom_loss = torch.zeros(1, device=W.device)
        print(f"Geometric loss: {geom_loss.item()}")

        total_loss = activation_loss + self.lamb * geom_loss

        return total_loss
    
class QuantizationObjective_v2(nn.Module):
    """
    Implements:

    L(Theta) =
        tr(C Σ C^T
           - s(Theta Σ C^T + C Σ Theta^T)
           + s^2 Theta Σ Theta^T)
      + alpha * || W Σ C^T
                   - s W Σ Theta^T
                   + C Σ W^T
                   - s Theta Σ W^T ||_F^2

    We are going to try implementing a regularizer in order to encourage Theta to be close to binary values (0 or 1) to make it more likely that the geometry-aware step will snap weights to the correct quantization grid points. The regularizer can be something like:
    binary_reg = (Theta * (1 - Theta)).sum()
    """

    def __init__(self, W, Sigma, Sigma_cross, Sigma_q,
                Q, s,
                lamb=1e-4,
                second_last_layer=False,
                last_layer=False,
                beta=2):

        super().__init__()

        # ---- Force 2D weight matrix ----
        # Works for both Linear and Conv (already flattened outside ideally)
        assert W.dim() == 2
        assert Q.dim() == 2

        d_out, d_in = W.shape

        # ---- Shape safety checks ----
        assert Sigma.shape == (d_in, d_in), \
            f"Sigma must be ({d_in},{d_in}), got {Sigma.shape}"
        assert Sigma_cross.shape == (d_in, d_in), \
            f"Sigma_cross must be ({d_in},{d_in}), got {Sigma_cross.shape}"
        assert Sigma_q.shape == (d_in, d_in), \
            f"Sigma_q must be ({d_in},{d_in}), got {Sigma_q.shape}"
        # ---- Per-channel scaling enforcement ----
        # s must be (d_out, 1)
        if s.dim() == 1:
            s = s.view(-1, 1)
        elif s.dim() == 0:
            s = s.view(1, 1).expand(d_out, 1)
        elif s.dim() == 4:              # Conv broadcast format
            s = s.view(s.size(0), 1)

        assert s.shape == (d_out, 1), \
            f"s must be ({d_out},1), got {s.shape}"

        # ---- Register tensors ----
        self.W = W
        self.Sigma = Sigma
        self.Sigma_cross = Sigma_cross
        self.Sigma_q = Sigma_q
        self.Q = Q
        self.s = s
        self.lamb_reg = lamb
        self.lamb = lamb if last_layer else 0.0
        self.beta = beta
        self.last_layer = last_layer
        self.second_last_layer = second_last_layer

        # Precompute C = W - sQ  (broadcast safe)
        self.C = self.W - self.s * self.Q

        # Trainable parameter (logits)
        self.Theta = nn.Parameter(torch.zeros_like(self.W))
        # For experimentation, set Theta to be C/s
        with torch.no_grad():
            self.Theta.data = self.C / (self.s + 1e-12)  # Avoid division by zero
    def forward(self):

        W = self.W                    # (d_out, d_in)
        Sigma = self.Sigma            # (d_in, d_in)
        Sigma_cross = self.Sigma_cross  # (d_in, d_in)
        Sigma_q = self.Sigma_q          # (d_in, d_in)
        C = self.C                    # (d_out, d_in)
        s = self.s                    # (d_out, 1)
        Q = self.Q                    # (d_out, d_in)
        Theta = torch.clip(torch.sigmoid(self.Theta)*1.2 -.1, 0.0, 1.0)  # (d_out, d_in)

        d_out, d_in = W.shape

        activation_loss = torch.zeros(1, device=W.device)
        geom_loss = torch.zeros(1, device=W.device)
        term_1 = torch.trace(W @ Sigma @ W.t())
        term_2 = -1.0* torch.trace(W @ Sigma_cross @ (s*(Q+Theta)).t()) -1.0* torch.trace((s*(Q+Theta)) @ Sigma_cross.t() @ W.t())
        term_3 = torch.trace((s*(Q+Theta)) @ Sigma_q @ (s*(Q+Theta)).t())
        activation_loss = term_1 + term_2 + term_3
        print(f"Activation loss: {activation_loss.item()}")

        # ---- Binary regularizer ----
        # This was trial one
        # binary_reg = ((Theta * (1 - Theta)) ** 2).sum()
        # Trial 2 is Adarounds form of  regularizer which is 1-|2Theta-1|^\beta
        binary_reg = self.lamb_reg * (1 - torch.abs(2*Theta - 1) ** self.beta).sum()
        # print(f"Binary regularizer: {binary_reg.item()}")
        activation_loss += binary_reg

        # ---- Geometric term (row-wise safe) ----
        geom_loss = torch.zeros(1, device=W.device)
        if self.second_last_layer:
            W_quant = s * (Q + Theta)
            K = W @ Sigma @ W.t()
            M = W_quant @ Sigma_q @ W_quant.t()
            geom_loss = torch.norm(K - M, p="fro") ** 2
        if self.last_layer:
            '''Going to check the Gram Matrix of just the weights without the input covariance term to ensure simplex ETF.'''
            W_quant = s * (Q + Theta)
            K = W @ W.t()
            M = W_quant @ W_quant.t()
            geom_loss = torch.norm(K - M, p="fro") ** 2
        print(f"Geometric loss: {geom_loss.item()}")

        total_loss = activation_loss + self.lamb * geom_loss

        return total_loss


class QuantizationObjective_v3(nn.Module):
    """
    Implements:

    L(Theta) =
        tr(C Σ C^T
           - s(Theta Σ C^T + C Σ Theta^T)
           + s^2 Theta Σ Theta^T)
      + alpha * || W Σ C^T
                   - s W Σ Theta^T
                   + C Σ W^T
                   - s Theta Σ W^T ||_F^2

    We are going to try implementing a regularizer in order to encourage Theta to be close to binary values (0 or 1) to make it more likely that the geometry-aware step will snap weights to the correct quantization grid points. The regularizer can be something like:
    binary_reg = (Theta * (1 - Theta)).sum()
    """

    def __init__(self, W, Sigma, Sigma_cross, Sigma_q,
                Q, s, M,
                lamb=1e-4,
                second_last_layer=False,
                last_layer=False,
                beta=2):

        super().__init__()

        # ---- Force 2D weight matrix ----
        # Works for both Linear and Conv (already flattened outside ideally)
        assert W.dim() == 2
        assert Q.dim() == 2

        d_out, d_in = W.shape

        # ---- Shape safety checks ----
        assert Sigma.shape == (d_in, d_in), \
            f"Sigma must be ({d_in},{d_in}), got {Sigma.shape}"
        assert Sigma_cross.shape == (d_in, d_in), \
            f"Sigma_cross must be ({d_in},{d_in}), got {Sigma_cross.shape}"
        assert Sigma_q.shape == (d_in, d_in), \
            f"Sigma_q must be ({d_in},{d_in}), got {Sigma_q.shape}"
        # ---- Per-channel scaling enforcement ----
        # s must be (d_out, 1)
        if s.dim() == 1:
            s = s.view(-1, 1)
        elif s.dim() == 0:
            s = s.view(1, 1).expand(d_out, 1)
        elif s.dim() == 4:              # Conv broadcast format
            s = s.view(s.size(0), 1)

        assert s.shape == (d_out, 1), \
            f"s must be ({d_out},1), got {s.shape}"

        # ---- Register tensors ----
        self.W = W
        self.Sigma = Sigma
        self.Sigma_cross = Sigma_cross
        self.Sigma_q = Sigma_q
        self.Q = Q * M  # Apply mask to Q to ensure pruned weights are not optimized towards nonzero values
        self.M = M  # Mask from pruning to ensure we only optimize over unpruned weights
        self.s = s
        self.lamb_reg = lamb
        self.lamb = lamb if last_layer else 0.0
        self.beta = beta
        self.last_layer = last_layer
        self.second_last_layer = second_last_layer

        # Precompute C = W - sQ  (broadcast safe)
        self.C = self.W - self.s * self.Q

        # Trainable parameter (logits)
        self.Theta = nn.Parameter(torch.zeros_like(self.W))
        # For experimentation, set Theta to be C/s
        with torch.no_grad():
            self.Theta.data = self.C / (self.s + 1e-12)  # Avoid division by zero
    def forward(self):

        W = self.W                    # (d_out, d_in)
        Sigma = self.Sigma            # (d_in, d_in)
        Sigma_cross = self.Sigma_cross  # (d_in, d_in)
        Sigma_q = self.Sigma_q          # (d_in, d_in)
        M = self.M                    # Mask to ensure we only optimize over unpruned weights
        C = self.C                    # (d_out, d_in)
        s = self.s                    # (d_out, 1)
        Q = self.Q                    # (d_out, d_in)
        Theta = torch.clip(torch.sigmoid(self.Theta)*1.2 -.1, 0.0, 1.0)  # (d_out, d_in)

        d_out, d_in = W.shape

        activation_loss = torch.zeros(1, device=W.device)
        W_bar = s * M * (Q + Theta)
        geom_loss = torch.zeros(1, device=W.device)
        term_1 = torch.trace(W @ Sigma @ W.t())
        term_2 = -1.0* torch.trace(W @ Sigma_cross @ W_bar.t()) -1.0* torch.trace(W_bar @ Sigma_cross.t() @ W.t())
        term_3 = torch.trace(W_bar @ Sigma_q @ W_bar.t())
        activation_loss = term_1 + term_2 + term_3
        print(f"Activation loss: {activation_loss.item()}")

        # ---- Binary regularizer ----
        # This was trial one
        # binary_reg = ((Theta * (1 - Theta)) ** 2).sum()
        # Trial 2 is Adarounds form of  regularizer which is 1-|2Theta-1|^\beta
        binary_reg = self.lamb_reg * (1 - torch.abs(2*Theta - 1) ** self.beta).sum()
        # print(f"Binary regularizer: {binary_reg.item()}")
        activation_loss += binary_reg

        # ---- Geometric term (row-wise safe) ----
        geom_loss = torch.zeros(1, device=W.device)
        if self.second_last_layer:
            W_quant = s * M * (Q + Theta)
            K = W @ Sigma @ W.t()
            M = W_quant @ Sigma_q @ W_quant.t()
            geom_loss = torch.norm(K - M, p="fro") ** 2
        if self.last_layer:
            '''Going to check the Gram Matrix of just the weights without the input covariance term to ensure simplex ETF.'''
            W_quant = s * M * (Q + Theta)
            K = W @ W.t()
            M = W_quant @ W_quant.t()
            geom_loss = torch.norm(K - M, p="fro") ** 2
        print(f"Geometric loss: {geom_loss.item()}")

        total_loss = activation_loss + self.lamb * geom_loss

        return total_loss



def run_pgd(model, lr=7.5e-3, steps=6000):

    # optimizer = torch.optim.SGD([model.Theta], lr=lr)

    optimizer = torch.optim.Adam([model.Theta], lr=lr)
    beta_0 = 2
    beta_final = 32
    beta_step = ((beta_final-beta_0)/steps)
    # scheduler = Min_OBJ_Lr_Scheduler(optimizer)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        model.beta += beta_step  # Gradually increase the strength of the binary regularizer

    # Return a projected Theta in [0,1] without reassigning the Parameter
    with torch.no_grad():
        Theta_proj = torch.clip((torch.sigmoid(model.Theta))*1.2 -.1, 0.0, 1.0) 
    return Theta_proj


def Set_Theta(theta):
    with torch.no_grad():
        theta[theta>.5] = 1.0
        theta[theta<=.5] = 0.0
    

def geometry_aware_rounding(model, dataloader, device, bitwidth=4):

    lamb = .0005
    print("lambda has been generated")

    Q, s = GenerateQ(model, bitwidth)
    print("Quantization grid Q has been generated")
    (second_last_name, _) = get_second_last_layer(model)[0]
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            masks[name] = (module.weight.data != 0).float()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            W = module.weight.data
            W_orig_shape = W.shape
            W_flat  = W.view(W.shape[0], -1)
            Q_flat = Q[name].view(W.shape[0], -1)
            s_flat = s[name].view(s[name].size(0), -1)
            if name in masks:
                mask_flat = masks[name].view(W_flat.shape).to(W_flat.device)
                W_flat = W_flat * mask_flat
                Q_flat = Q_flat * mask_flat
            # if name == "conv1":
            #     print("Q_flat[0] sample:", Q_flat[0, :9])  # first 9 elements = 3x3 kernel
            #     print("s_flat[0]:", s_flat[0])
            #     print("W_flat[0] sample:", W_flat[0, :9])
            if s_flat.size(1) != 1:
                s_flat = s_flat[:, :1]  # force (d_out,1)
            # print(f"The shape of s_flat for layer {name} is {s_flat.shape}")
            # print(f"The scales for layer {name} are {s_flat.squeeze()}")
            Sigma = estimate_input_covariance(model, module, dataloader, device)
            d_in = W_flat.size(1)
            assert Sigma.shape == (d_in, d_in), \
                f"Sigma mismatch: expected {(d_in,d_in)}, got {Sigma.shape}"
            
            objective = QuantizationObjective(W_flat, Sigma, Q_flat, s_flat, lamb=lamb, last_layer=(name == second_last_name))
            Theta_opt = run_pgd(objective)
            print(f"PGD optimization completed for layer {name}.")
            print(f"Resulting Theta is {Theta_opt}")
            # Single sigmoid application, detached clean tensor
            Set_Theta(Theta_opt)
            # if name == "conv1":
            #     print("Theta_opt[0] sample:", Theta_opt[0, :9])
            #     print("W_quant_flat[0] sample:", (s_flat * (Q_flat + Theta_opt))[0, :9])
            print(f"Optimized Theta for {name} obtained.")
            ## Print the number of ones in Theta_opt to see how many weights are being adjusted by the geometry-aware step
            num_ones = (Theta_opt > 0.5).sum().item()
            print(f"Number of weights adjusted by geometry-aware step in {name}: {num_ones}")
            ## Check if theta is just zeros or ones, if so, skip the geometry-aware update for this layer
            W_quant_flat = s_flat*(Q_flat + Theta_opt)
            
            # Reshape back to original shape for Conv2d
            W_quant = W_quant_flat.view(W_orig_shape)
            if name in masks:
                W_quant = W_quant * masks[name].to(W_quant.device)
            module.weight.data.copy_(W_quant)
            # print("Final weight[0] after copy:", module.weight.data[0])
    
def geometry_aware_rounding_v2(model, dataloader, device, bitwidth=4):

    lamb = 5e-5
    print("lambda has been generated")
    quant_model = copy.deepcopy(model)
    Q, s = GenerateQ(model, bitwidth)
    print("Quantization grid Q has been generated")
    (second_last_name, _) = get_second_last_layer(model)
    (last_name, _) = get_last_layer(model)
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            masks[name] = (module.weight.data != 0).float()
    for ((name, module),(quant_name, quant_module)) in zip(model.named_modules(), quant_model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            W = module.weight.data
            W_orig_shape = W.shape
            W_flat  = W.view(W.shape[0], -1)
            Q_flat = Q[name].view(W.shape[0], -1)
            s_flat = s[name].view(s[name].size(0), -1)
            Mask_flat = masks[name].view(W_flat.shape).to(W_flat.device) if name in masks else torch.ones_like(W_flat)
            if name in masks:
                mask_flat = masks[name].view(W_flat.shape).to(W_flat.device)
                W_flat = W_flat * mask_flat
                Q_flat = Q_flat * mask_flat
            if s_flat.size(1) != 1:
                s_flat = s_flat[:, :1]  # force (d_out,1)
            Sigma, Sigma_Cross, Sigma_Quant = estimate_input_quant_covariance(model, quant_model, module, quant_module, dataloader, device)
            d_in = W_flat.size(1)
            assert Sigma.shape == (d_in, d_in), \
                f"Sigma mismatch: expected {(d_in,d_in)}, got {Sigma.shape}"
            assert Sigma_Cross.shape == (d_in, d_in), \
                f"Sigma_Cross mismatch: expected {(d_in,d_in)}, got {Sigma_Cross.shape}"
            assert Sigma_Quant.shape == (d_in, d_in), \
                f"Sigma_Quant mismatch: expected {(d_in,d_in)}, got {Sigma_Quant.shape}"
            objective = QuantizationObjective_v3(W_flat, Sigma, Sigma_Cross, Sigma_Quant, Q_flat, s_flat, Mask_flat, lamb=lamb,second_last_layer=(name == second_last_name), last_layer=(name == last_name))
            Theta_opt = run_pgd(objective)
            print(f"PGD optimization completed for layer {name}.")
            print(f"Resulting Theta is {Theta_opt}")
            # Single sigmoid application, detached clean tensor
            Set_Theta(Theta_opt)
            print(f"Optimized Theta for {name} obtained.")
            ## Print the number of ones in Theta_opt to see how many weights are being adjusted by the geometry-aware step
            num_ones = (Theta_opt > 0.5).sum().item()
            print(f"Number of weights adjusted by geometry-aware step in {name}: {num_ones}")
            ## Check if theta is just zeros or ones, if so, skip the geometry-aware update for this layer
            W_quant_flat = s_flat*(Q_flat + Theta_opt)
            
            # Reshape back to original shape for Conv2d
            W_quant = W_quant_flat.view(W_orig_shape)
            if name in masks:
                W_quant = W_quant * masks[name].to(W_quant.device)
            quant_module.weight.data.copy_(W_quant)
            ## Perform a forward pass with data to get a quantized output for the next layer's covariance estimation to be more accurate
            # This is a key difference from v1 where we only do the geometry-aware step after
    return quant_model


    
def compute_compression(model,
                        original_bitwidth=32,
                        quantized_bitwidth=8,
                        include_index_cost=False,
                        index_bitwidth=32):
    """
    Computes sparsity-aware compression ratio.

    Args:
        model: PyTorch model
        original_bitwidth: bitwidth before quantization (default 32)
        quantized_bitwidth: bitwidth after quantization
        include_index_cost: whether to account for sparse index storage
        index_bitwidth: bitwidth for storing sparse indices (default 32)

    Returns:
        dict with:
            total_params
            nonzero_params
            sparsity
            original_bits
            compressed_bits
            compression_ratio
    """

    total_params = 0
    nonzero_params = 0

    for param in model.parameters():
        if param.requires_grad:
            data = param.data
            total_params += data.numel()
            nonzero_params += torch.count_nonzero(data).item()

    sparsity = 1 - (nonzero_params / total_params)

    original_bits = total_params * original_bitwidth

    if include_index_cost:
        compressed_bits = nonzero_params * (quantized_bitwidth + index_bitwidth)
    else:
        compressed_bits = nonzero_params * quantized_bitwidth

    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else float("inf")

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity": sparsity,
        "original_bits": original_bits,
        "compressed_bits": compressed_bits,
        "compression_ratio": compression_ratio
    }