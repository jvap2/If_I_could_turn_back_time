#!/usr/bin/env bash
set -e

cat > llm_quant_eval.py << 'PYFILE_EOF_MARKER_9f3a'
"""
llm_quant_eval.py - Runs the ACTUAL Hessian-weighted E2M1 CUDA kernels
(hessian_accumulate, hessian_damp_blocks, hessian_weight_solve - same
compiled extension bench.py builds) on a REAL pretrained LLM
(facebook/opt-125m), using REAL calibration activations captured from a real
forward pass, and measures WikiText-2 perplexity before vs. after quantizing
every targeted weight matrix. This is the "run it on a real model, not just
random matrices" version of the correctness checks in bench.py.

Runs TWO quantized configurations on the real model, both using the real
compiled CUDA kernels (no Python-side stand-ins for the actual quantize
math), plus an FP16 baseline and a naive-rounding weight-only control:

  1. W4A16 (weight-only): quantizes weights in their ORIGINAL (un-rotated)
     basis using the Hessian-weighted E2M1 kernels; activations stay FP16.
     Matches the fused GEMV kernel's own stated design point (see
     README.md, "Weight-only (W4A16), not W4A4..."). Does NOT replace the
     model's forward-pass matmuls with the custom fused GEMV kernel
     (e2m1_fused_gemv/e2m1_naive_gemv) - that kernel is a single-token
     (GEMV) op, already benchmarked and correctness-checked in bench.py
     section 4 in isolation; wiring it into a full batched, parallel-
     sequence forward pass for perplexity eval would need decode
     reimplemented as a token-by-token loop, which is real future work.
  2. W4A4 (weight + activation): both weights and activations get the same
     randomized blockwise Hadamard rotation (real hadamard_fwht kernel,
     applied to W's columns and to every activation tensor consistently, so
     W_had @ x_had = W @ x); the Hessian weight-quant then runs on the
     ROTATED weights/activations (real hessian_accumulate/damp_blocks/
     weight_solve kernels); activations are quantized every forward pass
     with the real GF4 encode/decode kernels, including the mean-centering
     (mu) and per-layer clip-ratio calibration search added in the previous
     pass (real gf4_encode(mu=...)/gf4_decode/clip-ratio-search kernels);
     bias_correction = W_had_q @ mu is added back after each (fake-quant)
     matmul. This is the configuration that actually exercises mu,
     bias_correction, and the clip-ratio search on a real model - W4A16
     above never touches them, since there's no activation quantization to
     need them.

A third, isolation-only configuration ("W4A16-Hadamard") reuses W4A4's
rotated Hessian-quantized weights but leaves activations at FP16 (real
hadamard_fwht rotation only, no GF4 encode/decode) - added specifically to
separate "did rotating the weights before quantizing them help" from "did
quantizing activations cost anything," since W4A16 and W4A4 differ in both
dimensions at once and conflate the two effects otherwise.

Both configurations use "fake-quantization" exactly as the deck's own PTQ
pipeline slide describes it: codes are computed by the real kernels, then
reconstructed to FP16 and substituted into the forward pass (via
module.weight.data for W4A16, via a patched module.forward for W4A4 since
that also needs the activation-side rotate/center/encode/decode), so the
*existing* FP16 GEMM machinery does the actual matmul. That is the same
protocol every number in the deck's Table 1 uses.

SCALING TO LARGER MODELS
-------------------------
Two features exist specifically for models too big for the smallest setups
this was first tested on (OPT-125M/1.3B, Llama-3.2-1B):

- Calibration activations are captured for every target layer in one pass
  (unavoidable - one forward pass fires every layer's hook), but are now
  QUANTIZED (both bases) and FREED immediately per layer, right after
  capture, instead of being held in CPU RAM for the entire script. Peak
  CPU RAM for calibration data is therefore O(one layer's activations),
  not O(all layers') - the difference between a few hundred MB and, at
  7B+ scale with the default calibration window count, tens of GB.
- `--device-map` loads the model with HF/accelerate's device_map="auto",
  spreading layers across available GPUs and, with `--max-memory`/
  `--offload-folder`, offloading to CPU RAM or disk when even that isn't
  enough. Every CUDA kernel call in this script explicitly stages its
  inputs on a fixed KERNEL_DEVICE and writes results back to wherever
  that specific layer's weight actually lives (module.weight.device),
  so it works correctly no matter how accelerate has placed things -
  the kernels themselves have no multi-GPU dispatch logic of their own.

NOT addressed by the above: `original_weights` (a full clone of every
target layer's original weights, kept so the naive/raw/rotated ablations
can each start from a clean slate) is still one extra full-size copy of
most of the model, resident for the whole run. At 7B-13B this is a
manageable, if real, VRAM/RAM cost; past that it would need to become a
CPU-resident or disk-backed snapshot restored on demand instead of an
in-memory clone - flagged here rather than silently left as a surprise.

I don't have GPU access in this environment, so none of the device_map/
offload code path has been run - treat the first attempt at a large model
exactly like every other kernel change in this project: a debug session,
not a guarantee.

Requires:  pip install transformers datasets accelerate
Run from the same directory as bindings.cpp / *.cu (reuses the exact
compiled `gf4_kernels` extension bench.py builds/caches - if you've already
run bench.py successfully, this will NOT need to recompile).

    python3 llm_quant_eval.py
    python3 llm_quant_eval.py --model meta-llama/Llama-2-7b-hf --device-map
    python3 llm_quant_eval.py --model meta-llama/Llama-2-13b-hf --device-map \
        --max-memory "0=20GiB,cpu=64GiB" --offload-folder /tmp/offload \
        --num-calib-windows 4

Expect a few minutes total on a single GPU for OPT-125M with the default
window counts below (model download aside). Raise NUM_EVAL_WINDOWS for a
more rigorous number if you have time before the interview - the deck's own
Table 1 numbers use the full WikiText-2 test set; this script defaults to a
smaller slice to keep first-run turnaround low. For bigger models, lowering
--num-calib-windows and --num-eval-windows is the fastest way to keep first-
run turnaround sane; raise them back up once you know it works.
"""
import os
import time
import zlib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Must be set before the first CUDA call (cuBLAS reads this at context init),
# so it goes before `import torch`. Without it, cuBLAS is free to pick a
# different GEMM algorithm/reduction order between runs, which makes the
# real model's own forward pass (used both to capture calibration
# activations AND to compute perplexity) non-bit-reproducible - the
# Hadamard-seed fix alone wasn't enough, since this is a second, independent
# source of run-to-run floating-point jitter that's large enough to flip a
# near-tied argmin (e.g. the clip-ratio search, or which E2M1 bias candidate
# wins) and cascade into a visibly different PPL.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True: fall back to non-deterministic behavior (with a warning)
# for any op that genuinely has no deterministic GPU implementation, rather
# than hard-crashing this script over an op we don't otherwise care about.
torch.use_deterministic_algorithms(True, warn_only=True)

import argparse


def _parse_max_memory(s):
    """Parses '0=20GiB,cpu=64GiB' into {0: '20GiB', 'cpu': '64GiB'} for
    transformers'/accelerate's max_memory kwarg. Device keys that look like
    plain integers become int GPU indices; everything else (e.g. 'cpu',
    'disk') stays a string key, matching what accelerate expects."""
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        k = k.strip()
        out[int(k) if k.isdigit() else k] = v.strip()
    return out


_cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_cli.add_argument("--model", default="facebook/opt-125m",
                   help="HF model name/path, e.g. facebook/opt-1.3b or meta-llama/Llama-3.2-1B "
                        "(default: facebook/opt-125m). RETAIN_FP16_SUBSTRINGS below covers "
                        "OPT-family (fc2) and Llama/Mistral/Qwen-family (down_proj) FFN naming "
                        "plus lm_head - other architectures (e.g. GPT-2's mlp.c_proj) may need "
                        "that list extended, or layers will silently get quantized that the "
                        "deck's policy says should stay FP16.")
_cli.add_argument("--seqlen", type=int, default=2048)
_cli.add_argument("--num-eval-windows", type=int, default=20)
_cli.add_argument("--num-calib-windows", type=int, default=8)
_cli.add_argument("--device-map", action="store_true",
                   help="Load the model with HF/accelerate's device_map='auto' instead of a "
                        "single .to(cuda) call - spreads layers across all visible GPUs and, "
                        "combined with --max-memory/--offload-folder, offloads to CPU RAM or "
                        "disk when the model doesn't fit in GPU memory alone. Requires "
                        "`pip install accelerate`. All CUDA kernel calls in this script stage "
                        "their own inputs on a fixed device and write results back to wherever "
                        "each layer actually lives, so this is safe to combine with offloading.")
_cli.add_argument("--max-memory", default=None,
                   help="Per-device memory cap for --device-map, e.g. '0=20GiB,cpu=64GiB' "
                        "(GPU 0 gets up to 20GiB, everything else spills to CPU RAM up to "
                        "64GiB, and then to --offload-folder if that's also exceeded).")
_cli.add_argument("--offload-folder", default=None,
                   help="Disk folder for accelerate to offload layers to when --device-map is "
                        "set and GPU+CPU memory (per --max-memory) both run out. Created if it "
                        "doesn't exist.")
_args = _cli.parse_args()

# ============================================================================
# Config
# ============================================================================
MODEL_NAME = _args.model
SEQLEN = _args.seqlen          # matches the deck's stated "non-overlapping 2048-token" protocol
NUM_EVAL_WINDOWS = _args.num_eval_windows      # raise for a more rigorous PPL number if time allows
NUM_CALIB_WINDOWS = _args.num_calib_windows    # separate windows (from TRAIN split) used only to build per-layer Hessians
HESS_BLOCK = 32                # must divide every target layer's in_features; must be <= HESSIAN_MAX_BLOCK (32) in the kernel
KAPPA = 100.0
POWER_ITERS = 30
# --- W4A4 (activation quantization) config ---
ACT_HAD_BLOCK = 32              # blockwise Hadamard block size for the activation-side rotation, same
                                 # convention as HESS_BLOCK/GF4_BLOCK elsewhere in this codebase; simpler
                                 # than trying to match a specific block size from the notebook, and still
                                 # a real randomized-Hadamard rotation, not a placeholder
CLIP_RATIO_CANDIDATES = (1.5, 2.0, 2.5, 3.0, 4.0)  # matches calibrate_model_gf4's real search grid
GF4_BLOCK = 32
# Outlier-layer retention, matching the deck's own "down_proj / fc2 / LM head
# stay FP16" policy. The FFN down-projection has a different name per model
# family - OPT calls it "fc2", Llama/Mistral/Qwen call it "down_proj" - so
# both substrings are listed to keep the retention policy equivalent across
# --model architectures. "lm_head" matches the (tied or untied) output
# embedding in both families. Verify this list still makes sense before
# trusting results on any newly-tried architecture - e.g. GPT-2's naming
# ("mlp.c_proj") isn't covered by either substring.
RETAIN_FP16_SUBSTRINGS = ("fc2", "down_proj", "lm_head")

# KERNEL_DEVICE is where every compiled CUDA kernel call in this script
# actually runs, regardless of --device-map. The compiled extension has no
# multi-GPU dispatch of its own - it just operates on whatever device its
# input tensors are on - so every kernel call below explicitly moves its
# inputs here first, then moves results back to wherever the destination
# (module.weight.device, or the input activation's original device inside
# the patched forwards) actually lives. Without --device-map this is simply
# "the" GPU, same as before.
KERNEL_DEVICE = torch.device("cuda:0")
torch.manual_seed(0)


def _input_device(model):
    """Device to send tokenized input_ids to. With --device-map spreading
    layers across GPUs/CPU/disk, this must be wherever the input embedding
    layer lives (accelerate's hooks handle moving activations to each
    subsequent layer's device automatically from there); without
    --device-map it's just the single device the whole model was moved to."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


# ============================================================================
# 0. Load/compile the real CUDA extension (identical to bench.py's build)
# ============================================================================
print("Loading CUDA extension (reuses bench.py's cached build if present)...")
ext = load(
    name="gf4_kernels",
    sources=[
        "bindings.cpp",
        "hadamard_kernel.cu",
        "gf4_encode_kernel.cu",
        "e2m1_fused_gemv_kernel.cu",
        "hessian_weight_quant_kernel.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Extension ready.\n")

# ============================================================================
# 1. Load model + tokenizer
# ============================================================================
_load_kwargs = dict(torch_dtype=torch.float16)
if _args.device_map:
    _load_kwargs["device_map"] = "auto"
    if _args.max_memory:
        _load_kwargs["max_memory"] = _parse_max_memory(_args.max_memory)
    if _args.offload_folder:
        os.makedirs(_args.offload_folder, exist_ok=True)
        _load_kwargs["offload_folder"] = _args.offload_folder

print(f"Loading {MODEL_NAME}"
      f"{' with device_map=auto (multi-GPU/CPU/disk offload enabled)' if _args.device_map else ''}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_load_kwargs)
if not _args.device_map:
    model = model.to(KERNEL_DEVICE)
model.eval()
if _args.device_map and hasattr(model, "hf_device_map"):
    print("Per-layer device placement (hf_device_map):")
    for k, v in model.hf_device_map.items():
        print(f"  {k:45s} -> {v}")
    print()

# ============================================================================
# 2. Load WikiText-2, tokenize once, slice into non-overlapping windows.
#    Eval windows come from the TEST split; calibration windows come from a
#    small slice of the TRAIN split, so the Hessian is never built from data
#    the perplexity number is measured on.
# ============================================================================
def load_wikitext2(split):
    # `datasets`' wikitext loader has changed its trust_remote_code
    # requirement across versions; try the plain call first and only fall
    # back if that's actually the problem, rather than guessing up front.
    try:
        return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            return load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
        raise


print("Loading WikiText-2...")
test_ds = load_wikitext2("test")
test_text = "\n\n".join(test_ds["text"])
test_ids = tokenizer(test_text, return_tensors="pt").input_ids[0]

train_ds = load_wikitext2("train")
calib_text = "\n\n".join(train_ds["text"][:2000])
calib_ids = tokenizer(calib_text, return_tensors="pt").input_ids[0]


def get_windows(ids, seqlen, n_windows):
    n_windows = min(n_windows, ids.numel() // seqlen)
    return [ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0) for i in range(n_windows)]


eval_windows = get_windows(test_ids, SEQLEN, NUM_EVAL_WINDOWS)
calib_windows = get_windows(calib_ids, SEQLEN, NUM_CALIB_WINDOWS)
print(f"Eval windows (test split): {len(eval_windows)}   "
      f"Calibration windows (train split): {len(calib_windows)}\n")


# ============================================================================
# 3. Perplexity helper - standard non-overlapping-window protocol.
#    HF's CausalLM `loss` (given labels=input_ids) is the MEAN cross-entropy
#    over the seqlen-1 shifted-target predictions in that window. Multiplying
#    back by (numel-1) recovers the window's TOTAL NLL, so summing across
#    windows and dividing by the true total token count (not window count)
#    gives the correct corpus-level perplexity - validated against a fully
#    manual (no "mean loss") token-level computation before shipping this.
# ============================================================================
@torch.no_grad()
def compute_perplexity(model, windows):
    total_nll = 0.0
    total_tokens = 0
    in_device = _input_device(model)
    for w in windows:
        w = w.to(in_device)
        out = model(w, labels=w)
        n_targets = w.numel() - 1
        total_nll += out.loss.float().item() * n_targets
        total_tokens += n_targets
    return float(np.exp(total_nll / total_tokens))


print("Computing BASELINE (unquantized FP16) perplexity...")
t0 = time.time()
ppl_baseline = compute_perplexity(model, eval_windows)
print(f"Baseline PPL: {ppl_baseline:.4f}  ({time.time()-t0:.1f}s)\n")

# ============================================================================
# 4. Identify target Linear layers (everything except retained ones)
# ============================================================================
def is_retained(name):
    return any(s in name for s in RETAIN_FP16_SUBSTRINGS)


targets = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and not is_retained(name):
        if module.in_features % HESS_BLOCK == 0:
            targets.append((name, module))
        else:
            print(f"  skipping {name}: in_features={module.in_features} "
                  f"not divisible by HESS_BLOCK={HESS_BLOCK}")

print(f"Quantizing {len(targets)} linear layers "
      f"(retaining FP16 for names matching {RETAIN_FP16_SUBSTRINGS})\n")

# Snapshot original weights so we can (a) restore for the naive-rounding
# ablation below and (b) keep every config's run on identical starting
# weights for a fair comparison. NOTE: this is still one extra full-size
# copy of most of the model's weights, resident for the whole run - not
# addressed by the streaming calibration fix below, since it serves a
# different purpose (restoring between configs, not calibration). At 7-13B
# this is a real but manageable cost; past that it would need to become a
# CPU-resident or disk-backed snapshot instead of an in-memory clone.
original_weights = {name: module.weight.data.clone() for name, module in targets}

# ============================================================================
# 5. Register hooks to capture REAL calibration input activations per layer
# ============================================================================
captured = {name: [] for name, _ in targets}
hooks = []


def make_hook(name):
    def hook(module, inputs, output):
        x = inputs[0].detach()
        captured[name].append(x.reshape(-1, x.shape[-1]).float().cpu())
    return hook


for name, module in targets:
    hooks.append(module.register_forward_hook(make_hook(name)))

print("Running calibration forward passes to capture real activations...")
calib_input_device = _input_device(model)
with torch.no_grad():
    for w in calib_windows:
        model(w.to(calib_input_device))

for h in hooks:
    h.remove()
print("Calibration activations captured.\n")

# ============================================================================
# 6. E2M1 dequantize (torch, GPU) - mirrors reference.py's already-validated
#    NumPy dequant math (dequant_hessian_gemv_ref), used here only to
#    reconstruct a dense weight tensor from the kernel's packed codes for
#    fake-quantization. This is plain tensor indexing (gather), not a new
#    CUDA kernel - the actual quantization COMPUTE (the Hessian solve) is
#    100% the real kernel; this step just unpacks its output.
# ============================================================================
E2M1_CODEBOOKS_T = torch.tensor([
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
], dtype=torch.float32, device=KERNEL_DEVICE)


def unpack_codes_torch(packed):
    lo = (packed & 0xF).to(torch.uint8)
    hi = ((packed >> 4) & 0xF).to(torch.uint8)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequantize_e2m1(W_codes, W_alpha, W_bias, block_size):
    M = W_codes.shape[0]
    codes = unpack_codes_torch(W_codes).long()               # [M, K], values 0-15
    n_blocks = codes.shape[1] // block_size
    codes = codes.view(M, n_blocks, block_size)
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0).float()
    mag_table = E2M1_CODEBOOKS_T[W_bias.long()]               # [M, n_blocks, 8]
    idx = (codes & 0x7)                                       # [M, n_blocks, block_size], values 0-7
    mag = torch.gather(mag_table, 2, idx)                     # [M, n_blocks, block_size]
    W_hat = (sign * mag * W_alpha.float().unsqueeze(-1)).reshape(M, n_blocks * block_size)
    return W_hat


# ============================================================================
# 7. Quantize every target layer ONCE for BOTH bases - raw (feeds W4A16) and
#    Hadamard-rotated (feeds W4A4 and the W4A16-Hadamard ablation) - and free
#    each layer's captured calibration activations immediately afterward.
#
#    This is the memory-critical step for larger models: the previous
#    structure captured every layer's activations, then consumed them in TWO
#    separate later passes (raw-basis solve, then much later the rotated-
#    basis solve), so ALL layers' calibration activations stayed resident in
#    CPU RAM for the entire script. Doing both solves back-to-back per layer,
#    right here, means only ONE layer's activations are ever alive at once -
#    the difference between O(all layers) and O(one layer) CPU RAM, which is
#    a few hundred MB for OPT-125M but tens of GB at 7B+ with the default
#    calibration window count.
#
#    Results are stored compactly (packed 4-bit codes + small alpha/bias
#    arrays, moved to CPU) rather than as dense per-layer weight copies, so
#    idle state between the four evaluation passes below stays small; each
#    pass dequantizes on demand right before it needs the weights.
# ============================================================================
print("Quantizing weights with the real Hessian-weighted CUDA kernels")
print("(raw basis for W4A16, Hadamard-rotated basis for W4A4/W4A16-Hadamard,")
print("computed together per layer so each layer's calibration activations")
print("can be freed immediately)...")

raw_quant_state = {}   # name -> dict(codes, alpha, bias) on CPU - raw (un-rotated) basis
rot_quant_state = {}   # name -> dict(...) on CPU - Hadamard-rotated basis
per_layer_relerr = []
t_start = time.time()

for name, module in targets:
    K = module.in_features
    N = module.out_features

    X_raw = torch.cat(captured[name], dim=0).to(KERNEL_DEVICE).float().contiguous()
    del captured[name]   # free this layer's CPU-resident calibration activations NOW
    W_fp32 = module.weight.data.float().to(KERNEL_DEVICE).contiguous()

    # --- raw-basis Hessian weight-quant (feeds W4A16) ---
    H = ext.hessian_accumulate(X_raw, HESS_BLOCK)                      # REAL kernel, REAL data
    H_damped = ext.hessian_damp_blocks(H, KAPPA, POWER_ITERS)          # REAL kernel
    W_codes, W_alpha, W_bias = ext.hessian_weight_solve(W_fp32, H_damped)  # REAL kernel

    W_hat = dequantize_e2m1(W_codes, W_alpha, W_bias, HESS_BLOCK)
    relerr = (W_hat - W_fp32).norm() / (W_fp32.norm() + 1e-8)
    per_layer_relerr.append((name, relerr.item()))
    raw_quant_state[name] = dict(codes=W_codes.cpu(), alpha=W_alpha.cpu(), bias=W_bias.cpu())

    # --- Hadamard-rotated basis (feeds W4A4 and the W4A16-Hadamard ablation) ---
    # Same per-layer random sign vector applied to both the weight's columns
    # and every activation tensor at that layer - the invariant that makes
    # the rotation reversible/consistent (QuaRot-style). Seeded from a
    # deterministic hash of the layer name (zlib.crc32), NOT Python's
    # built-in hash() - str hashing is randomized per-process by default
    # (PYTHONHASHSEED), so hash(name) gives a different sign vector, and
    # therefore a slightly different PPL, on every single run. crc32 is
    # stable across runs/processes/machines.
    seed = zlib.crc32(name.encode("utf-8")) % (2 ** 31)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d_sign = (torch.randint(0, 2, (K,), generator=gen).float() * 2 - 1).to(KERNEL_DEVICE)

    X_had = ext.hadamard_fwht(X_raw, ACT_HAD_BLOCK, d_sign)          # REAL kernel: rotate calibration activations
    W_had = ext.hadamard_fwht(W_fp32, ACT_HAD_BLOCK, d_sign)         # REAL kernel: rotate weight columns identically

    H_r = ext.hessian_accumulate(X_had, HESS_BLOCK)                  # REAL kernel, on ROTATED activations
    H_r_damped = ext.hessian_damp_blocks(H_r, KAPPA, POWER_ITERS)    # REAL kernel
    Wc_r, Wa_r, Wb_r = ext.hessian_weight_solve(W_had, H_r_damped)   # REAL kernel, on ROTATED weight
    W_had_hat = dequantize_e2m1(Wc_r, Wa_r, Wb_r, HESS_BLOCK)        # dense reconstructed rotated weight

    mu = X_had.mean(dim=0).contiguous()                              # [K], real per-channel mean, rotated basis
    bias_correction = (W_had_hat @ mu).contiguous()                  # [N] = W_had_q @ mu

    # Clip-ratio calibration search: real gf4_encode(mu=...)/gf4_decode
    # round trip per candidate, exactly like bench.py section 5d, but on
    # this layer's real (rotated) calibration activations.
    X_had_flat = X_had.reshape(-1).contiguous()
    n_blocks_cal = X_had_flat.numel() // GF4_BLOCK
    best_alpha, best_mse = CLIP_RATIO_CANDIDATES[0], float("inf")
    for alpha in CLIP_RATIO_CANDIDATES:
        codes_c, scales_c = ext.gf4_encode(X_had_flat, alpha, True, mu)
        x_dec_c = ext.gf4_decode(codes_c, scales_c, n_blocks_cal)
        # gf4_decode doesn't re-add mu (by design - see gf4_decode_kernel),
        # so compare against the mean-centered activations it's meant to
        # reconstruct, matching clip_ratio_search_ref's convention.
        mu_tiled = mu.repeat(X_had.shape[0])
        mse_c = float(((X_had_flat - mu_tiled - x_dec_c) ** 2).mean().item())
        if mse_c < best_mse:
            best_mse, best_alpha = mse_c, alpha

    rot_quant_state[name] = dict(
        K=K, N=N,
        weight_had_q_half=W_had_hat.half().cpu().contiguous(),
        bias_correction=bias_correction.float().cpu(),
        orig_bias=(module.bias.data.float().cpu().clone() if module.bias is not None else None),
        d_sign=d_sign.cpu(), had_block=ACT_HAD_BLOCK, mu=mu.cpu(), clip_ratio=best_alpha,
    )

    del (X_raw, X_had, X_had_flat, W_fp32, W_had, H, H_damped, H_r, H_r_damped,
         W_codes, W_alpha, W_bias, W_hat, Wc_r, Wa_r, Wb_r, W_had_hat)
    torch.cuda.empty_cache()

t_quant = time.time() - t_start
print(f"Computed both quantization bases for {len(targets)} layers in {t_quant:.1f}s "
      f"using the real CUDA kernels.\n")
print("Per-layer weight reconstruction relative error, raw basis (Frobenius norm):")
for name, err in per_layer_relerr:
    print(f"  {name:45s} {err:.4f}")
print()
print("Per-layer calibrated clip_ratio (alpha*), Hadamard-rotated basis:")
for name in rot_quant_state:
    print(f"  {name:45s} alpha*={rot_quant_state[name]['clip_ratio']}")
print()

# ============================================================================
# 8. Apply the raw-basis weights and compute W4A16 perplexity.
# ============================================================================
print("Applying raw-basis Hessian-weighted W4A16 weights...")
for name, module in targets:
    st = raw_quant_state[name]
    W_hat = dequantize_e2m1(
        st["codes"].to(KERNEL_DEVICE), st["alpha"].to(KERNEL_DEVICE), st["bias"].to(KERNEL_DEVICE), HESS_BLOCK)
    module.weight.data.copy_(W_hat.to(module.weight.device).to(module.weight.dtype))
    del W_hat

print("Computing QUANTIZED (real CUDA kernel, fake-quant) perplexity...")
t0 = time.time()
ppl_quantized = compute_perplexity(model, eval_windows)
print(f"Quantized PPL: {ppl_quantized:.4f}  ({time.time()-t0:.1f}s)\n")

# ============================================================================
# 9. Control: naive round-to-nearest quantization, SAME 4-bit E2M1 format and
#    block size, NO Hessian weighting. This isolates whether the Hessian
#    solve is actually earning its complexity - if it doesn't meaningfully
#    beat plain rounding, that's a real problem; if it does, the remaining
#    gap to the deck's Table 1 number is explained by pipeline stages this
#    script deliberately doesn't include (activation quant, residual passes,
#    Hadamard rotation), not a bug in the kernel.
# ============================================================================
print("Restoring original weights, then running a naive round-to-nearest")
print("ablation (same format/block size, NO Hessian weighting) for a")
print("same-conditions comparison...")
for name, module in targets:
    module.weight.data.copy_(original_weights[name])


def quantize_scale_e4m3_torch(alpha, e_bits=4, m_bits=3):
    """Torch mirror of reference.py's quantize_scale_e4m3_ref - the same
    E4M3 scale-quantization step the real kernel applies, so the naive
    ablation below isn't unfairly given a higher-precision scale than the
    Hessian-weighted kernel path gets."""
    e_min = -(2 ** (e_bits - 1))
    e_max = (2 ** (e_bits - 1)) - 1
    a = alpha.clamp_min(1e-8)
    e = torch.floor(torch.log2(a)).clamp(e_min, e_max)
    base = torch.pow(2.0, e)
    levels = 2 ** m_bits
    frac = a / base - 1.0
    frac_q = torch.round(frac * levels) / levels
    return base * (1.0 + frac_q)


def quantize_e2m1_naive_torch(W_fp32, block_size, bias=1):
    """Per-block max-abs scale + nearest-codebook rounding - NO Hessian
    weighting. Pure PyTorch, no CUDA kernel needed for this control.
    Codes are assigned using the raw (unquantized) scale, then the scale
    itself is E4M3-quantized before use in reconstruction - matching the
    order of operations in reference.py's quantize_weight_e2m1_fast_ref, so
    this is a fair same-precision comparison against the CUDA kernel path,
    not just "no Hessian weighting plus a free higher-precision scale."""
    N, M = W_fp32.shape
    n_blocks = M // block_size
    Wb = W_fp32.view(N, n_blocks, block_size)
    w_abs = Wb.abs()
    cb = E2M1_CODEBOOKS_T[bias]
    alpha_raw = (w_abs.amax(dim=2) / cb.max()).clamp_min(1e-8)      # [N, n_blocks]
    xn = w_abs / alpha_raw.unsqueeze(-1)
    dist = (xn.unsqueeze(-1) - cb.view(1, 1, 1, -1)).abs()          # [N, n_blocks, block_size, 8]
    codes_mag = dist.argmin(dim=-1)
    alpha_q = quantize_scale_e4m3_torch(alpha_raw)
    sign = torch.where(Wb < 0, -1.0, 1.0)
    W_hat = sign * cb[codes_mag] * alpha_q.unsqueeze(-1)
    return W_hat.reshape(N, M)


t0 = time.time()
for name, module in targets:
    W_fp32 = module.weight.data.float().to(KERNEL_DEVICE).contiguous()
    W_hat = quantize_e2m1_naive_torch(W_fp32, HESS_BLOCK)
    module.weight.data.copy_(W_hat.to(module.weight.device).to(module.weight.dtype))
    del W_fp32, W_hat
print(f"Naive-rounding quantization done in {time.time()-t0:.1f}s.\n")

ppl_naive = compute_perplexity(model, eval_windows)
print(f"Naive round-to-nearest PPL: {ppl_naive:.4f}\n")

# Restore original (un-rotated) weights before the rotated-basis passes below
# read module.bias / module.weight.device for their own state, and so
# module.weight.data is back to a clean baseline in case anything else
# inspects it between here and the next config.
for name, module in targets:
    module.weight.data.copy_(original_weights[name])

# ============================================================================
# 10. W4A4: weights AND activations quantized, using the real Hadamard,
#     Hessian, GF4-encode, and GF4-decode CUDA kernels. This is the
#     configuration that actually exercises mean-centering (mu),
#     bias-correction, and the clip-ratio calibration search on a real
#     model - the W4A16 run above never touches them, since there's no
#     activation quantization step for them to hook into.
# ============================================================================
print("=" * 72)
print("W4A4: weight + activation quantization (real Hadamard/Hessian/GF4 kernels)")
print("=" * 72)


def make_w4a4_forward(st, ext_ref, kernel_device):
    """Closure holding one layer's precomputed W4A4 state. Every call
    re-derives the activation quantization from scratch using the real
    hadamard_fwht + gf4_encode(mu=...) + gf4_decode kernels - nothing here
    is cached across tokens/windows, so this is a faithful (if slow)
    forward pass, not a shortcut.

    Moves x to kernel_device on entry and the result back to x's original
    device on exit, so this works whether x arrives on kernel_device
    directly (no --device-map) or on whatever device accelerate placed this
    layer's input on (--device-map, possibly a different GPU or CPU)."""
    K, N = st["K"], st["N"]
    W_half = st["weight_had_q_half"].to(kernel_device)
    bias_correction = st["bias_correction"].to(kernel_device)
    orig_bias = st["orig_bias"].to(kernel_device) if st["orig_bias"] is not None else None
    d_sign = st["d_sign"].to(kernel_device)
    had_block = st["had_block"]
    mu = st["mu"].to(kernel_device)
    clip_ratio = st["clip_ratio"]

    def forward(x):
        orig_dtype = x.dtype
        orig_device = x.device
        orig_shape = x.shape
        x_flat = x.reshape(-1, K).float().to(kernel_device).contiguous()
        x_had = ext_ref.hadamard_fwht(x_flat, had_block, d_sign)          # REAL kernel, every forward call
        n_blocks = x_had.numel() // GF4_BLOCK
        codes, scales = ext_ref.gf4_encode(x_had.reshape(-1).contiguous(), clip_ratio, True, mu)  # REAL kernel
        x_q = ext_ref.gf4_decode(codes, scales, n_blocks).reshape(x_had.shape)                     # REAL kernel
        y = x_q.half() @ W_half.t()
        y = y.float() + bias_correction.unsqueeze(0)
        if orig_bias is not None:
            y = y + orig_bias.unsqueeze(0)
        return y.to(orig_dtype).to(orig_device).reshape(*orig_shape[:-1], N)

    return forward


print("Patching forward pass for each target layer (real kernels run on every token)...")
orig_forwards = {}
for name, module in targets:
    orig_forwards[name] = module.forward
    module.forward = make_w4a4_forward(rot_quant_state[name], ext, KERNEL_DEVICE)

print("Computing W4A4 (real Hadamard + Hessian + GF4 encode/decode) perplexity...")
t0 = time.time()
ppl_w4a4 = compute_perplexity(model, eval_windows)
print(f"W4A4 PPL: {ppl_w4a4:.4f}  ({time.time()-t0:.1f}s)\n")

for name, module in targets:
    module.forward = orig_forwards[name]

# ============================================================================
# 11. Isolation ablation: Hadamard-rotated Hessian weight-quant, activations
#     left FULL PRECISION (no GF4 encode/decode, no mu, no bias_correction).
#
# The W4A16 run (section 8) solves the Hessian weight-quant in the RAW,
# un-rotated basis. The W4A4 run (section 10) solves it in the ROTATED basis
# AND quantizes activations. This section separates those two changes: it
# reuses the exact same rotated, Hessian-quantized weight
# (rot_quant_state[name]) and the exact same per-layer d_sign, but the
# activation path only does the REAL hadamard_fwht kernel (a lossless,
# invertible rotation - no information lost, unlike GF4 encode/decode) and
# skips activation quantization entirely. If this number lands close to
# W4A4, the rotation is doing essentially all the work and activation
# quantization is nearly free here; if it lands close to FP16 baseline,
# activation quantization is the one earning its keep; if it lands between,
# both matter and roughly by how much is now visible.
# ============================================================================
print("=" * 72)
print("W4A16-HADAMARD: Hadamard-rotated weight quant, activations left FP16")
print("(isolates whether the W4A16 -> W4A4 improvement came from rotating")
print("the weights before quantizing them, from quantizing activations, or both)")
print("=" * 72)


def make_w4a16_hadamard_forward(st, ext_ref, kernel_device):
    """Same rotated+quantized weight as the W4A4 forward above, but the
    activation side only runs the real hadamard_fwht kernel (lossless
    rotation) - no gf4_encode/gf4_decode, no mu, no bias_correction, since
    none of those are needed when activations aren't being quantized."""
    K, N = st["K"], st["N"]
    W_half = st["weight_had_q_half"].to(kernel_device)
    orig_bias = st["orig_bias"].to(kernel_device) if st["orig_bias"] is not None else None
    d_sign = st["d_sign"].to(kernel_device)
    had_block = st["had_block"]

    def forward(x):
        orig_dtype = x.dtype
        orig_device = x.device
        orig_shape = x.shape
        x_flat = x.reshape(-1, K).float().to(kernel_device).contiguous()
        x_had = ext_ref.hadamard_fwht(x_flat, had_block, d_sign)   # REAL kernel, lossless rotation only
        y = (x_had.half() @ W_half.t()).float()
        if orig_bias is not None:
            y = y + orig_bias.unsqueeze(0)
        return y.to(orig_dtype).to(orig_device).reshape(*orig_shape[:-1], N)

    return forward


print("Patching forward pass (Hadamard-rotated weights, FP16 activations)...")
for name, module in targets:
    orig_forwards[name] = module.forward
    module.forward = make_w4a16_hadamard_forward(rot_quant_state[name], ext, KERNEL_DEVICE)

print("Computing W4A16-Hadamard (rotated weight quant, no activation quant) perplexity...")
t0 = time.time()
ppl_w4a16_had = compute_perplexity(model, eval_windows)
print(f"W4A16-Hadamard PPL: {ppl_w4a16_had:.4f}  ({time.time()-t0:.1f}s)\n")

for name, module in targets:
    module.forward = orig_forwards[name]
    module.weight.data.copy_(original_weights[name])

print("=" * 72)
print(f"{MODEL_NAME}  -  WikiText-2, {len(eval_windows)} non-overlapping {SEQLEN}-token windows")
print("=" * 72)
print(f"  FP16 baseline PPL:                            {ppl_baseline:.4f}")
print(f"  Naive round-to-nearest (no Hessian):           {ppl_naive:.4f}   (delta {ppl_naive - ppl_baseline:+.4f})")
print(f"  Hessian-weighted W4A16, raw basis:              {ppl_quantized:.4f}   (delta {ppl_quantized - ppl_baseline:+.4f})")
print(f"  Hessian-weighted W4A16, Hadamard-rotated basis: {ppl_w4a16_had:.4f}   (delta {ppl_w4a16_had - ppl_baseline:+.4f})")
print(f"  W4A4 (Hadamard-rotated + GF4-quantized acts):   {ppl_w4a4:.4f}   (delta {ppl_w4a4 - ppl_baseline:+.4f})")
print()
if ppl_quantized < ppl_naive:
    print(f"Hessian weighting beats naive rounding by {ppl_naive - ppl_quantized:.4f} PPL at identical "
          f"bit width/block size - the kernel's weighting is doing real work.")
else:
    print(f"Hessian weighting did NOT beat naive rounding in the raw basis (worse by "
          f"{ppl_quantized - ppl_naive:.4f} PPL).")
rotation_gain = ppl_quantized - ppl_w4a16_had     # positive = rotation helped
actquant_cost = ppl_w4a16_had - ppl_w4a4          # positive = activation quant cost extra PPL (should be small/negative if act quant is nearly free here)
print(f"Rotating the weights before the Hessian solve changed PPL by {-rotation_gain:+.4f} "
      f"(raw-basis {ppl_quantized:.4f} -> rotated-basis {ppl_w4a16_had:.4f}) - this isolates the "
      f"rotation's effect with activations held at FP16 the whole time.")
print(f"Adding GF4 activation quantization (mu-centering + bias-correction + calibrated "
      f"clip_ratio) on top of the rotated weights changed PPL by {ppl_w4a4 - ppl_w4a16_had:+.4f} "
      f"(rotated-basis-FP16-acts {ppl_w4a16_had:.4f} -> W4A4 {ppl_w4a4:.4f}).")
print()
print("Remember: the deck's Table 1 numbers are the FULL pipeline (Hadamard-")
print("rotated activations + GF4 residual passes + outlier retention +")
print("Hessian weight quant together, with clip-ratio calibration and")
print("mean-centering baked into every layer). The three ablations above now")
print("separate that pipeline's components on a real model - a remaining gap")
print("vs. Table 1 is most likely GF4 residual passes and outlier retention,")
print("which this script still does not implement, not a bug in the")
print("Hadamard/Hessian/GF4 kernels themselves.")
PYFILE_EOF_MARKER_9f3a

echo "llm_quant_eval.py updated."