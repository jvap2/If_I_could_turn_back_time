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
  (unavoidable - one forward pass fires every layer's hook, so ALL layers'
  activations are simultaneously resident in CPU RAM during the capture
  pass itself - no per-layer freeing trick helps with that part). What IS
  fixed: once captured, activations are QUANTIZED (both bases) and FREED
  immediately per layer, instead of being held in CPU RAM for the entire
  rest of the script - the difference between O(one layer's activations)
  and O(all layers') for everything AFTER capture. Captured activations are
  also stored as fp16, not fp32 (halving capture-time memory for free,
  since the model only computes them at fp16 precision in the first place -
  cast back to fp32 right before use). Even with both of these, the CAPTURE
  pass itself can be tens of GB at 7B+ scale with more than a couple of
  calibration windows - keep `--num-calib-windows` small (1-4) for large
  models; it's the one setting this rewrite couldn't make scale-free.
- `--device-map` loads the model with HF/accelerate's device_map="auto",
  spreading layers across available GPUs and, with `--max-memory`, spilling
  to CPU RAM when GPU memory isn't enough. Every CUDA kernel call in this
  script explicitly stages its inputs on a fixed KERNEL_DEVICE and writes
  results back to wherever that specific layer's weight actually lives
  (module.weight.device), so it works correctly no matter how accelerate
  has placed things across GPU(s)/CPU - the kernels themselves have no
  multi-GPU dispatch logic of their own.

By default, accelerate's dispatch hooks (attached at the "no-split" block
granularity, e.g. one whole decoder layer, not per-Linear-submodule) leave
every offloaded (cpu- or disk-tier) parameter as a meta placeholder between
forward() calls - materializing the real values just-in-time before each
call and DISCARDING them right after. That's transparent for plain
inference, but incompatible with fake-quantizing a layer (read the weight,
quantize it, write the result back into module.weight.data, expect it to
still be there several forward passes later): left alone, the very next
forward() call would silently discard the quantized weight and restore the
hook's own original copy. Right after target layers are identified, this
script walks up to each target's nearest hooked ancestor, materializes it
for real, snapshots the values, writes them back as ordinary resident
tensors, and disables just that hook's weight-offload behavior (not the
hook itself - it stays attached so it keeps moving activations across
device boundaries correctly, which is what actually makes a GPU/CPU-split
model work at layer boundaries). This makes `--offload-folder` (disk
offload) usable for target layers too, at a cost: a disk-tier block gets
pulled fully into memory (onto wherever accelerate would have executed it,
normally CPU RAM) and kept there for the rest of the run, trading away
disk's RAM savings for that block in exchange for correctness. If that
trade doesn't fit your machine, a smaller --model or a larger
--max-memory cpu= budget is the better lever than leaning on disk offload
for layers this script actually quantizes.

NOT addressed by the above: `original_weights` (a full clone of every
target layer's original weights, kept so the naive/raw/rotated ablations
can each start from a clean slate) is still one extra full-size copy of
most of the model, resident in CPU RAM for the whole run. At 7B-13B this is
a manageable, if real, RAM cost; past that it would need to become a
disk-backed snapshot restored on demand instead of an in-memory clone -
flagged here rather than silently left as a surprise.

I don't have GPU access in this environment, so none of the device_map
code path has been run - treat the first attempt at a large model exactly
like every other kernel change in this project: a debug session, not a
guarantee.

Requires:  pip install transformers datasets accelerate
Run from the same directory as bindings.cpp / *.cu (reuses the exact
compiled `gf4_kernels` extension bench.py builds/caches - if you've already
run bench.py successfully, this will NOT need to recompile).

    python3 llm_quant_eval.py
    python3 llm_quant_eval.py --model meta-llama/Llama-2-7b-hf --device-map
    python3 llm_quant_eval.py --model meta-llama/Llama-2-13b-hf --device-map \
        --max-memory "0=20GiB,cpu=64GiB" --num-calib-windows 4

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
_cli.add_argument("--config", default="all",
                   choices=["all", "baseline", "naive", "w4a16_raw", "w4a16_hadamard",
                            "w4a4", "rotated"],
                   help="Which config(s) to run. 'all' (default) = the legacy single-process "
                        "run of every config (keeps the ~model-sized fp16 original_weights clone, "
                        "which caps it near 7B). Any SINGLE config runs in isolation and needs NO "
                        "clone (process exit frees everything, nothing to restore), so big models "
                        "fit in just model + that config's calibration. 'rotated' = w4a16_hadamard "
                        "+ w4a4 together (they share the rotated calibration). Run one config per "
                        "process and each appends its PPL to --results-csv; assemble the table from "
                        "the CSV. This is how you get every perplexity on a 30B model without the "
                        "clone OOM.")
_cli.add_argument("--results-csv", default="eval_results.csv",
                   help="CSV that each run appends its (model, config, ppl, ...) row to, so "
                        "per-config runs in separate processes accumulate into one table.")
_cli.add_argument("--device-map", action="store_true",
                   help="Load the model with HF/accelerate's device_map='auto' instead of a "
                        "single .to(cuda) call - spreads layers across all visible GPUs and, "
                        "combined with --max-memory, spills to CPU RAM when the model doesn't "
                        "fit in GPU memory alone. Requires `pip install accelerate`. All CUDA "
                        "kernel calls in this script stage their own inputs on a fixed device "
                        "and write results back to wherever each layer actually lives, so this "
                        "is safe to combine with CPU offloading.")
_cli.add_argument("--max-memory", default=None,
                   help="Per-device memory cap for --device-map, e.g. '0=20GiB,cpu=64GiB' "
                        "(GPU 0 gets up to 20GiB, everything else spills to CPU RAM up to "
                        "64GiB, and then to --offload-folder if that's also exceeded).")
_cli.add_argument("--offload-folder", default=None,
                   help="Disk folder for accelerate to offload layers to when GPU+CPU memory "
                        "(per --max-memory) both run out. Created if it doesn't exist. Usable "
                        "even for layers this script quantizes - accelerate's disk-offload hooks "
                        "would normally reload a layer's original weight from disk before every "
                        "forward() call and discard in-memory edits afterward (silently undoing "
                        "fake-quantization), but this script detects and materializes any "
                        "disk-tier block containing a target layer into permanently-resident CPU "
                        "RAM right after loading, specifically to avoid that. That trades away "
                        "disk's RAM savings for those blocks in exchange for correctness - if "
                        "your machine can't absorb that, raise --max-memory's cpu= budget or use "
                        "a smaller --model instead of leaning on disk for target layers.")
_cli.add_argument("--fixed-clip", type=float, default=None,
                   help="Use a single fixed activation clip ratio (e.g. 2.5) instead of the "
                        "per-layer adaptive clip search. Weight-side Hessian reconstruction is "
                        "unchanged, so this gives the fixed-clip + Hessian form of GF4.")
_cli.add_argument("--per-block-clip", action="store_true",
                   help="PER-BLOCK adaptive activation clip: each 32-elem block picks the clip "
                        "in the candidate grid that minimizes its own reconstruction MSE, online, "
                        "every forward (no per-layer calibration). Matches the original FP_Quant "
                        "experiments' quantize_activations_gf4_adaptive (bit_split.py). This is the "
                        "'alpha per block' iso-configuration. Mutually exclusive with --fixed-clip.")
_cli.add_argument("--hess-block", type=int, default=32,
                   help="Block size for the E2M1/Hessian weight reconstruction (must divide "
                        "each target layer's in_features and be <= 32, the kernel max). Set 16 "
                        "to match the deployed 'pure GF4' pipeline's block size.")
_args = _cli.parse_args()


from huggingface_hub import login
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)
# ============================================================================
# Config
# ============================================================================
MODEL_NAME = _args.model
SEQLEN = _args.seqlen          # matches the deck's stated "non-overlapping 2048-token" protocol
NUM_EVAL_WINDOWS = _args.num_eval_windows      # raise for a more rigorous PPL number if time allows
NUM_CALIB_WINDOWS = _args.num_calib_windows    # separate windows (from TRAIN split) used only to build per-layer Hessians

# --- Which configs this process runs (see --config). A SINGLE config needs no
#     original_weights clone: nothing precedes it in this process to restore. ---
CONFIG          = _args.config
WANT_BASELINE   = CONFIG in ("all", "baseline")
WANT_NAIVE      = CONFIG in ("all", "naive")
WANT_W4A16_RAW  = CONFIG in ("all", "w4a16_raw")
WANT_W4A16_HAD  = CONFIG in ("all", "w4a16_hadamard", "rotated")
WANT_W4A4       = CONFIG in ("all", "w4a4", "rotated")
NEED_RAW_CALIB  = WANT_W4A16_RAW                       # raw-basis Hessian solve
NEED_ROT_CALIB  = WANT_W4A16_HAD or WANT_W4A4          # rotated-basis Hessian solve
NEED_CALIB      = NEED_RAW_CALIB or NEED_ROT_CALIB     # baseline/naive need no calibration at all
NEED_CLONE      = CONFIG == "all"                      # only the legacy all-in-one run restores between configs


def _append_result(config, ppl):
    """Append one (model, config, ppl) row to --results-csv so per-config runs
    in separate processes accumulate into a single table. Written immediately
    after each PPL so an OOM in a later config never loses an earlier result."""
    import csv as _csv
    path = _args.results_csv
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = _csv.writer(f)
        if new:
            w.writerow(["timestamp", "model", "config", "ppl", "seqlen",
                        "eval_windows", "calib_windows"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), MODEL_NAME, config,
                    f"{ppl:.4f}", SEQLEN, NUM_EVAL_WINDOWS, NUM_CALIB_WINDOWS])
    print(f"  [recorded {config} -> {ppl:.4f} in {path}]")
HESS_BLOCK = _args.hess_block  # E2M1/Hessian weight block; must divide in_features and be <= 32 (kernel max)
assert HESS_BLOCK <= 32, "hess-block must be <= 32 (HESSIAN_MAX_BLOCK in the kernel)"
KAPPA = 100.0
POWER_ITERS = 30
# --- W4A4 (activation quantization) config ---
ACT_HAD_BLOCK = 32              # blockwise Hadamard block size for the activation-side rotation, same
                                 # convention as HESS_BLOCK/GF4_BLOCK elsewhere in this codebase; simpler
                                 # than trying to match a specific block size from the notebook, and still
                                 # a real randomized-Hadamard rotation, not a placeholder
CLIP_RATIO_CANDIDATES = (1.5, 2.0, 2.5, 3.0, 4.0)  # matches calibrate_model_gf4's real search grid
PER_BLOCK_CLIP = _args.per_block_clip               # per-block online adaptive clip (bit_split parity)
if _args.per_block_clip and _args.fixed_clip is not None:
    raise SystemExit("--per-block-clip and --fixed-clip are mutually exclusive: per-block searches "
                     "the grid per block, fixed-clip pins one value.")
if _args.fixed_clip is not None:                    # fixed-clip + Hessian form (no per-layer search)
    CLIP_RATIO_CANDIDATES = (_args.fixed_clip,)
    print(f"[fixed-clip] activation clip ratio pinned to {_args.fixed_clip} (adaptive search disabled)")
if PER_BLOCK_CLIP:                                  # each block picks its own clip from this grid, online
    print(f"[per-block-clip] online per-block adaptive clip over {CLIP_RATIO_CANDIDATES} "
          f"(no per-layer calibration; alpha travels with each block's data)")
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
        "gf4_fused_gemv_kernel.cu",     # defines launch_gf4_fused/lattice/naive + dense_fp16 (referenced by bindings.cpp)
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

# Confirm what actually got loaded, rather than trusting torch_dtype=float16
# silently took effect - some transformers/accelerate version combinations
# have sized the automatic device map (or even loaded weights) in fp32
# despite an explicit torch_dtype request, which roughly doubles real memory
# use vs. what --max-memory was budgeted for and is a common, confusing
# cause of "why did accelerate push so much to disk/cpu when this should
# easily fit."
_actual_dtype = next(model.parameters()).dtype
print(f"Loaded weight dtype: {_actual_dtype}")
if _actual_dtype != torch.float16:
    _ratio = torch.tensor([], dtype=_actual_dtype).element_size() / torch.tensor([], dtype=torch.float16).element_size()
    print(f"  WARNING: requested torch.float16 but parameters are {_actual_dtype} - actual memory "
          f"use is roughly {_ratio:.0f}x what --max-memory was sized for assuming fp16. This is "
          f"the most likely explanation if more layers than expected landed on cpu/disk.")

if _args.device_map and hasattr(model, "hf_device_map"):
    from collections import Counter
    tally = Counter(model.hf_device_map.values())
    print(f"Device placement tally: {dict(tally)}")
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
    # Newer datasets/huggingface_hub reject the legacy bare name "wikitext"
    # (HfUriError: "Repository id must be 'namespace/name'"). The parquet-hosted
    # Salesforce/wikitext mirror has no loading script and works on both old and
    # new library versions.
    last = None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            return load_dataset(repo, "wikitext-2-raw-v1", split=split)
        except Exception as e:
            last = e
    raise RuntimeError(f"could not load wikitext-2-raw-v1 (try: pip install -U datasets): {last}")


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


ppl_baseline = None
if WANT_BASELINE:
    print("Computing BASELINE (unquantized FP16) perplexity...")
    t0 = time.time()
    ppl_baseline = compute_perplexity(model, eval_windows)
    print(f"Baseline PPL: {ppl_baseline:.4f}  ({time.time()-t0:.1f}s)\n")
    _append_result("baseline", ppl_baseline)

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

# --device-map spreads the model across GPU(s)/CPU(/disk) using accelerate's
# AlignDevicesHook, attached at the "no-split" block granularity (e.g. one
# whole decoder layer), not per-Linear-submodule - a target Linear layer's
# OWN module typically has no _hf_hook of its own even when its weight sits
# at the meta device; the hook lives on some ANCESTOR module, and it
# materializes/releases every parameter under that ancestor together, once
# per forward() call. That's transparent for plain inference (forward passes
# already go through it correctly - the calibration pass and
# compute_perplexity above and below don't need to know or care), but it's
# fundamentally incompatible with what this section does next: reading
# module.weight.data directly outside of forward() to quantize it, then
# writing the quantized result back and expecting it to stick across many
# later forward() calls. Left alone, the very next forward() would
# re-materialize the hook's own stored original weight and silently discard
# our quantized one - worse for a disk-tier layer (reloaded from an on-disk
# checkpoint) than a cpu-tier one, but the same underlying problem either
# way: the hook, not module.weight, is the actual source of truth.
#
# Fix: for every target layer, walk up to its nearest hooked ancestor,
# materialize that ancestor's parameters for real (the same
# pre_forward()/post_forward() cycle a real forward() call would trigger),
# snapshot the values, write them back as ordinary resident tensors (not
# meta placeholders), and set hook.offload = False so future pre_forward
# calls stop re-materializing-then-discarding weights.
#
# Deliberately NOT calling accelerate.hooks.remove_hook_from_module here:
# the hook's OTHER job - moving each incoming activation tensor onto
# whatever device this block's parameters live on (hook.execution_device),
# unconditionally, every forward() call - is exactly what makes a model
# split across GPU and CPU work at all; a decoder layer on GPU 0 feeding
# into the next one on CPU only works because that next layer's hook moves
# the incoming activation from GPU to CPU before running its own forward.
# Removing the hook entirely would silently break that at every such
# boundary (a device-mismatch RuntimeError the next time forward() crosses
# it) - setting hook.offload = False keeps the activation-movement behavior
# intact and only disables the weight materialize/discard behavior we don't
# want.
#
# For a cpu-tier ancestor this costs nothing extra (its data was going to
# live in CPU RAM regardless). For a disk-tier ancestor this pulls that
# block fully into memory (onto hook.execution_device, normally CPU) and
# keeps it there, trading away disk offload's RAM savings for that specific
# block in exchange for this script's quantize-then-restore approach
# actually being correct - if that trade doesn't fit your machine, a
# smaller --model or a larger --max-memory cpu= budget (so accelerate never
# reaches for disk in the first place) is the better lever than trying to
# keep disk offload AND per-layer weight surgery at the same time.
if _args.device_map:
    _module_by_name = dict(model.named_modules())

    def _hooked_ancestor(name):
        """Nearest module carrying its own _hf_hook, searching from `name`
        itself (a target Linear leaf can carry its own hook directly, if
        accelerate's no-split granularity happens to be per-Linear rather
        than per-block) up through every parent, all the way to the root
        model."""
        parts = name.split(".")
        for i in range(len(parts), -1, -1):
            candidate = ".".join(parts[:i])
            mod = model if candidate == "" else _module_by_name.get(candidate)
            if mod is not None and getattr(mod, "_hf_hook", None) is not None:
                return mod
        return None

    def _iter_align_hooks(hook):
        """Yields every hook-with-an-.offload-attribute reachable from
        `hook`. accelerate sometimes chains multiple hooks on one module via
        a SequentialHook (e.g. an AlignDevicesHook plus a tied-weights hook)
        instead of attaching a single AlignDevicesHook directly - a
        SequentialHook itself has no .offload attribute of its own, only its
        children (in .hooks) do, so checking hasattr(anc._hf_hook, 'offload')
        on the outer object can silently be False even though a real,
        active offload hook is nested one level down. Calling
        hook.pre_forward()/post_forward() on the outer object still works
        either way (SequentialHook delegates to its children), but disabling
        offload has to reach every actual AlignDevicesHook, wherever it is."""
        if hook is None:
            return
        if hasattr(hook, "offload"):
            yield hook
        for child in getattr(hook, "hooks", []):
            yield from _iter_align_hooks(child)

    _ancestors_to_fix = {}
    for name, module in targets:
        anc = _hooked_ancestor(name)
        if anc is not None and any(p.device.type == "meta" for p in anc.parameters(recurse=True)):
            _ancestors_to_fix[id(anc)] = anc

    if _ancestors_to_fix:
        from accelerate.utils import set_module_tensor_to_device

        print(f"Materializing {len(_ancestors_to_fix)} offloaded block(s) containing target "
              f"layers into permanently-resident tensors (the block's activation-movement hook "
              f"stays attached and active - only its weight offload/discard behavior is turned "
              f"off - so cross-device forward passes still work correctly at every layer "
              f"boundary)...")
        for anc in _ancestors_to_fix.values():
            hook = anc._hf_hook
            hook.pre_forward(anc)                      # materializes real values (wherever accelerate executes this block)
            materialized = {n: p.detach().clone() for n, p in anc.named_parameters(recurse=True)}
            hook.post_forward(anc, None)                # resets to meta; harmless, already captured above
            # A plain `p.data = real_tensor` assignment fails here
            # ("incompatible tensor type") because `p` was originally a meta
            # tensor - PyTorch's `.data` setter can't swap a meta parameter
            # for a real one in place. set_module_tensor_to_device is
            # accelerate's own sanctioned way to do this correctly (it's
            # literally what pre_forward calls internally to materialize a
            # parameter in the first place), handling nested attribute
            # names and constructing a fresh real Parameter under the hood.
            for n, val in materialized.items():
                set_module_tensor_to_device(anc, n, val.device, value=val)
            align_hooks = list(_iter_align_hooks(hook))
            for h in align_hooks:
                h.offload = False                       # stop future pre_forward calls from re-materializing/discarding
            if not align_hooks:
                print(f"  WARNING: found no .offload-bearing hook under {type(hook).__name__} for "
                      f"one block - its weights were materialized once (safe for this run) but may "
                      f"get reset to meta again on a later forward() call if accelerate's hook "
                      f"structure doesn't match what this script expects.")
        print("Done.\n")

    # Verify the fix actually reached every target rather than assuming it
    # did - accelerate's hook nesting/granularity varies by version and
    # model, and the ancestor search above is a best-effort match against
    # that structure, not a guarantee. Fail here, loudly and with the exact
    # layer name(s) involved, instead of crashing several lines later with
    # a generic PyTorch error that gives no clue which layer or why.
    _still_meta = [name for name, module in targets if module.weight.device.type == "meta"]
    if _still_meta:
        print(f"\n{len(_still_meta)} target layer(s) are STILL on the meta device after the "
              f"materialization step above - the ancestor/hook search didn't fully match this "
              f"model's actual accelerate hook structure. Diagnostic info for the first few:")
        for name in _still_meta[:5]:
            own_hook = getattr(_module_by_name.get(name), "_hf_hook", None)
            anc = _hooked_ancestor(name)
            anc_name = next((k for k, v in _module_by_name.items() if v is anc), "?") if anc is not None else None
            print(f"  {name}: own _hf_hook={type(own_hook).__name__ if own_hook is not None else None}, "
                  f"nearest hooked ancestor={anc_name} ({type(anc).__name__ if anc is not None else None})")
        raise SystemExit(
            "\nCannot proceed with meta-device target weights (see diagnostics above) - please "
            "share this output so the ancestor-detection logic can be adjusted for your "
            "installed accelerate/transformers version. In the meantime, dropping --device-map "
            "(if the model fits without it) or trying a smaller --model will avoid this path "
            "entirely.")

# Snapshot original weights so we can (a) restore for the naive-rounding
# ablation below and (b) keep every config's run on identical starting
# weights for a fair comparison. Cloned to CPU (not left on module.weight's
# own device) deliberately: on a --device-map run, target layers can already
# be sitting on the GPU right up against --max-memory's cap, and a
# same-device clone would try to duplicate that many bytes on an already-
# full GPU - which is exactly what OOMs here on a 16GB card with a 7B model
# before any quantization work even starts. CPU RAM is the much roomier
# budget in that situation. This is still one extra full-size copy of most
# of the model's weights, resident in CPU RAM for the whole run - fine at
# 7-13B, a real limitation past that, where it would need to become a
# disk-backed snapshot restored on demand instead.
# Only the legacy all-in-one run (--config all) needs this clone, to restore
# originals between the several weight-modifying configs it runs back-to-back.
# A single-config run modifies weights at most once and then the process exits,
# so there is nothing to restore -> no clone -> the ~model-sized fp16 CPU copy
# that caps this script near 7B simply doesn't exist.
original_weights = ({name: module.weight.data.detach().to("cpu").clone() for name, module in targets}
                    if NEED_CLONE else None)

# ============================================================================
# 5. Register hooks to capture REAL calibration input activations per layer
# ============================================================================
captured = {name: [] for name, _ in targets}
hooks = []


def make_hook(name):
    def hook(module, inputs, output):
        x = inputs[0].detach()
        # Stored as fp16, not fp32: one shared forward pass fires every
        # target layer's hook at once, so ALL layers' calibration
        # activations are simultaneously resident in CPU RAM during capture
        # (the per-layer "free right after use" optimization elsewhere in
        # this script only helps AFTER capture finishes, not during it -
        # capturing is inherently O(all layers), not O(one layer), since a
        # single forward pass is what fires every hook). At 7B+ hidden
        # sizes this is tens of GB even for a handful of calibration
        # windows; halving it via fp16 storage costs nothing real, since
        # the model itself only computes these activations at fp16
        # precision in the first place - upcasting to fp32 on capture never
        # recovered any precision that wasn't already lost. Cast back to
        # fp32 right before use (see the quantization loop below).
        captured[name].append(x.reshape(-1, x.shape[-1]).half().cpu())
    return hook


if NEED_CALIB:   # baseline / naive need no calibration at all
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
    # Kept in uint8 (1 byte/element) as long as possible rather than casting
    # to int64 up front - torch.gather requires an int64 index, but that's
    # only actually needed for `idx`, right at the gather call, not for the
    # sign computation. For a 14336x4096 layer (Mistral's gate_proj/up_proj)
    # that's the difference between an ~470MB and an ~59MB buffer for the
    # unpacked codes - on a GPU where resident model weights (from
    # --device-map) already fill most of the card, that difference is what
    # separates fitting from an OOM crash. In-place multiplies below avoid
    # two more full-size float32 temporaries beyond what's unavoidable.
    M = W_codes.shape[0]
    codes_u8 = unpack_codes_torch(W_codes)                    # uint8, [M, K], values 0-15
    n_blocks = codes_u8.shape[1] // block_size
    codes_u8 = codes_u8.view(M, n_blocks, block_size)
    sign = torch.where((codes_u8 & 0x8) != 0, -1.0, 1.0)      # float32, cheap uint8 compare
    mag_table = E2M1_CODEBOOKS_T[W_bias.long()]               # [M, n_blocks, 8]
    idx = (codes_u8 & 0x7).long()                             # int64 only right at gather time
    del codes_u8
    mag = torch.gather(mag_table, 2, idx)                     # [M, n_blocks, block_size]
    del idx, mag_table
    W_hat = sign
    W_hat.mul_(mag).mul_(W_alpha.float().unsqueeze(-1))       # in-place - no extra full-size temporaries
    return W_hat.reshape(M, n_blocks * block_size)


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
# Always defined so later config sections can reference them unconditionally.
raw_quant_state = {}   # name -> dict(codes, alpha, bias) on CPU - raw (un-rotated) basis
rot_quant_state = {}   # name -> dict(...) on CPU - Hadamard-rotated basis
per_layer_relerr = []

if NEED_CALIB:
    _bases = ("raw+rotated" if (NEED_RAW_CALIB and NEED_ROT_CALIB)
              else "raw" if NEED_RAW_CALIB else "Hadamard-rotated")
    print("Quantizing weights with the real Hessian-weighted CUDA kernels")
    print(f"({_bases} basis, per --config {CONFIG}; each layer's calibration")
    print("activations are freed immediately after its solve)...")
    t_start = time.time()

    for name, module in targets:
        K = module.in_features
        N = module.out_features

        X_raw = torch.cat(captured[name], dim=0).to(KERNEL_DEVICE).float().contiguous()
        del captured[name]   # free this layer's CPU-resident calibration activations NOW
        W_fp32 = module.weight.data.float().to(KERNEL_DEVICE).contiguous()

        if NEED_RAW_CALIB:
            # --- raw-basis Hessian weight-quant (feeds W4A16) ---
            H = ext.hessian_accumulate(X_raw, HESS_BLOCK)                      # REAL kernel, REAL data
            H_damped = ext.hessian_damp_blocks(H, KAPPA, POWER_ITERS)          # REAL kernel
            W_codes, W_alpha, W_bias = ext.hessian_weight_solve(W_fp32, H_damped)  # REAL kernel
            W_hat = dequantize_e2m1(W_codes, W_alpha, W_bias, HESS_BLOCK)
            relerr = (W_hat - W_fp32).norm() / (W_fp32.norm() + 1e-8)
            per_layer_relerr.append((name, relerr.item()))
            raw_quant_state[name] = dict(codes=W_codes.cpu(), alpha=W_alpha.cpu(), bias=W_bias.cpu())
            del H, H_damped, W_codes, W_alpha, W_bias, W_hat

        if NEED_ROT_CALIB:
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
            # this layer's real (rotated) calibration activations. SKIPPED under
            # --per-block-clip: there the clip is chosen per block, online, from
            # each forward's own activations, so there is nothing to calibrate
            # per layer (best_alpha is unused by the per-block forward path).
            X_had_flat = None
            if PER_BLOCK_CLIP:
                best_alpha = None
            else:
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
            del X_had, X_had_flat, W_had, H_r, H_r_damped, Wc_r, Wa_r, Wb_r, W_had_hat

        del X_raw, W_fp32
        torch.cuda.empty_cache()

    t_quant = time.time() - t_start
    print(f"Computed {_bases} quantization for {len(targets)} layers in {t_quant:.1f}s "
          f"using the real CUDA kernels.\n")
if per_layer_relerr:
    print("Per-layer weight reconstruction relative error, raw basis (Frobenius norm):")
    for name, err in per_layer_relerr:
        print(f"  {name:45s} {err:.4f}")
    print()
if rot_quant_state and PER_BLOCK_CLIP:
    print(f"Activation clip: PER-BLOCK adaptive over {CLIP_RATIO_CANDIDATES} (chosen online per "
          f"32-elem block every forward; no per-layer alpha* to report).\n")
elif rot_quant_state:
    print("Per-layer calibrated clip_ratio (alpha*), Hadamard-rotated basis:")
    for name in rot_quant_state:
        print(f"  {name:45s} alpha*={rot_quant_state[name]['clip_ratio']}")
    # DIAGNOSTIC: distribution of selected alphas. If this is a single value for
    # every layer, adaptive == fixed by construction (suspicious - see below).
    from collections import Counter as _Counter
    _alpha_hist = _Counter(rot_quant_state[n]["clip_ratio"] for n in rot_quant_state)
    print(f"  [alpha* histogram] {dict(sorted(_alpha_hist.items()))}   "
          f"(candidates were {CLIP_RATIO_CANDIDATES})")
    if len(_alpha_hist) == 1:
        print(f"  [WARN] every layer selected the SAME clip {next(iter(_alpha_hist))} -- "
              f"adaptive will tie a fixed run at that clip; verify this is real, not a "
              f"degenerate/no-op search.")
    print()

# ============================================================================
# 8. Apply the raw-basis weights and compute W4A16 perplexity.
# ============================================================================
ppl_quantized = None
if WANT_W4A16_RAW:
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
    _append_result("w4a16_raw", ppl_quantized)

# ============================================================================
# 9. Control: naive round-to-nearest quantization, SAME 4-bit E2M1 format and
#    block size, NO Hessian weighting. This isolates whether the Hessian
#    solve is actually earning its complexity - if it doesn't meaningfully
#    beat plain rounding, that's a real problem; if it does, the remaining
#    gap to the deck's Table 1 number is explained by pipeline stages this
#    script deliberately doesn't include (activation quant, residual passes,
#    Hadamard rotation), not a bug in the kernel.
# ============================================================================
ppl_naive = None
if WANT_NAIVE and NEED_CLONE:
    # 'all' mode only: a prior config overwrote the weights in this same process,
    # so restore the originals before the naive ablation. A single --config naive
    # run has untouched weights (and no clone), so it skips straight to quantizing.
    print("Restoring original weights before the naive round-to-nearest ablation...")
    for name, module in targets:
        module.weight.data.copy_(original_weights[name].to(module.weight.device).to(module.weight.dtype))


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
    xn = w_abs / alpha_raw.unsqueeze(-1)                            # [N, n_blocks, block_size]
    # Nearest codebook entry WITHOUT materializing the [N,n_blocks,block_size,8]
    # distance tensor (that 8x broadcast OOMs the GPU at 6.7B+). Loop over the 8
    # levels tracking a running best; strict '<' keeps the first-min index on
    # ties, matching argmin's default. Peak memory is O(1x) instead of O(8x).
    best_dist = torch.full_like(xn, float("inf"))
    codes_mag = torch.zeros(xn.shape, dtype=torch.long, device=xn.device)
    for i in range(cb.numel()):
        d = (xn - cb[i]).abs()
        better = d < best_dist
        best_dist = torch.where(better, d, best_dist)
        codes_mag = torch.where(better, i, codes_mag)
    alpha_q = quantize_scale_e4m3_torch(alpha_raw)
    sign = torch.where(Wb < 0, -1.0, 1.0)
    W_hat = sign * cb[codes_mag] * alpha_q.unsqueeze(-1)
    return W_hat.reshape(N, M)


if WANT_NAIVE:
    print("Naive round-to-nearest (same 4-bit E2M1 format/block, NO Hessian weighting)...")
    t0 = time.time()
    for name, module in targets:
        W_fp32 = module.weight.data.float().to(KERNEL_DEVICE).contiguous()
        W_hat = quantize_e2m1_naive_torch(W_fp32, HESS_BLOCK)
        module.weight.data.copy_(W_hat.to(module.weight.device).to(module.weight.dtype))
        del W_fp32, W_hat
    print(f"Naive-rounding quantization done in {time.time()-t0:.1f}s.")
    ppl_naive = compute_perplexity(model, eval_windows)
    print(f"Naive round-to-nearest PPL: {ppl_naive:.4f}\n")
    _append_result("naive", ppl_naive)

if NEED_CLONE:
    # 'all' mode: restore the un-rotated originals before the rotated-basis passes
    # below inspect module state. Single-config runs never overwrote the weights
    # this way (the patched-forward configs don't touch module.weight at all).
    for name, module in targets:
        module.weight.data.copy_(original_weights[name].to(module.weight.device).to(module.weight.dtype))

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
    # Keep the big quantized weight on CPU and stream it to the GPU per-call (below).
    # Pre-moving it here pins a full second copy of the whole model on the GPU across
    # ALL patched layers, which OOMs a 16GB card at 6.7B+; device_map streams the base
    # model the same way, so this just mirrors that for the quantized weights.
    W_half_cpu = st["weight_had_q_half"]
    bias_correction = st["bias_correction"].to(kernel_device)
    orig_bias = st["orig_bias"].to(kernel_device) if st["orig_bias"] is not None else None
    d_sign = st["d_sign"].to(kernel_device)
    had_block = st["had_block"]
    mu = st["mu"].to(kernel_device)
    clip_ratio = st["clip_ratio"]
    # Per-block adaptive clip: candidate grid lives on the kernel device; each
    # block picks its own clip online (see gf4_encode_adaptive). Built once per
    # layer closure, not per forward call.
    clip_candidates_t = (torch.tensor(CLIP_RATIO_CANDIDATES, dtype=torch.float32, device=kernel_device)
                         if PER_BLOCK_CLIP else None)

    def forward(x):
        orig_dtype = x.dtype
        orig_device = x.device
        orig_shape = x.shape
        x_flat = x.reshape(-1, K).float().to(kernel_device).contiguous()
        x_had = ext_ref.hadamard_fwht(x_flat, had_block, d_sign)          # REAL kernel, every forward call
        n_blocks = x_had.numel() // GF4_BLOCK
        if PER_BLOCK_CLIP:
            # each block searches the grid for its own MSE-optimal clip, online
            codes, scales = ext_ref.gf4_encode_adaptive(x_had.reshape(-1).contiguous(), clip_candidates_t, mu)
        else:
            codes, scales = ext_ref.gf4_encode(x_had.reshape(-1).contiguous(), clip_ratio, True, mu)  # REAL kernel
        x_q = ext_ref.gf4_decode(codes, scales, n_blocks).reshape(x_had.shape)                     # REAL kernel
        W_half = W_half_cpu.to(kernel_device, non_blocking=True)          # stream this layer's weight in
        y = x_q.half() @ W_half.t()
        del W_half                                                        # free before the next layer streams its own
        y = y.float() + bias_correction.unsqueeze(0)
        if orig_bias is not None:
            y = y + orig_bias.unsqueeze(0)
        return y.to(orig_dtype).to(orig_device).reshape(*orig_shape[:-1], N)

    return forward


orig_forwards = {}
ppl_w4a4 = None
if WANT_W4A4:
    print("Patching forward pass for each target layer (real kernels run on every token)...")
    for name, module in targets:
        orig_forwards[name] = module.forward
        module.forward = make_w4a4_forward(rot_quant_state[name], ext, KERNEL_DEVICE)

    print("Computing W4A4 (real Hadamard + Hessian + GF4 encode/decode) perplexity...")
    t0 = time.time()
    ppl_w4a4 = compute_perplexity(model, eval_windows)
    print(f"W4A4 PPL: {ppl_w4a4:.4f}  ({time.time()-t0:.1f}s)\n")
    _append_result("w4a4", ppl_w4a4)

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
    W_half_cpu = st["weight_had_q_half"]   # streamed per-call (see make_w4a4_forward note) to avoid pinning a full model copy on GPU
    orig_bias = st["orig_bias"].to(kernel_device) if st["orig_bias"] is not None else None
    d_sign = st["d_sign"].to(kernel_device)
    had_block = st["had_block"]

    def forward(x):
        orig_dtype = x.dtype
        orig_device = x.device
        orig_shape = x.shape
        x_flat = x.reshape(-1, K).float().to(kernel_device).contiguous()
        x_had = ext_ref.hadamard_fwht(x_flat, had_block, d_sign)   # REAL kernel, lossless rotation only
        W_half = W_half_cpu.to(kernel_device, non_blocking=True)   # stream this layer's weight in
        y = (x_had.half() @ W_half.t()).float()
        del W_half
        if orig_bias is not None:
            y = y + orig_bias.unsqueeze(0)
        return y.to(orig_dtype).to(orig_device).reshape(*orig_shape[:-1], N)

    return forward


ppl_w4a16_had = None
if WANT_W4A16_HAD:
    print("Patching forward pass (Hadamard-rotated weights, FP16 activations)...")
    for name, module in targets:
        orig_forwards[name] = module.forward
        module.forward = make_w4a16_hadamard_forward(rot_quant_state[name], ext, KERNEL_DEVICE)

    print("Computing W4A16-Hadamard (rotated weight quant, no activation quant) perplexity...")
    t0 = time.time()
    ppl_w4a16_had = compute_perplexity(model, eval_windows)
    print(f"W4A16-Hadamard PPL: {ppl_w4a16_had:.4f}  ({time.time()-t0:.1f}s)\n")
    _append_result("w4a16_hadamard", ppl_w4a16_had)

    for name, module in targets:
        module.forward = orig_forwards[name]
    if NEED_CLONE:
        for name, module in targets:
            module.weight.data.copy_(original_weights[name].to(module.weight.device).to(module.weight.dtype))

print("=" * 72)
print(f"{MODEL_NAME}  -  WikiText-2, {len(eval_windows)} non-overlapping {SEQLEN}-token windows"
      f"  (--config {CONFIG})")
print("=" * 72)


def _summary_line(label, ppl, is_baseline=False):
    if ppl is None:
        return
    if ppl_baseline is not None and not is_baseline:
        print(f"  {label:46s} {ppl:.4f}   (delta {ppl - ppl_baseline:+.4f})")
    else:
        print(f"  {label:46s} {ppl:.4f}")


_summary_line("FP16 baseline PPL:", ppl_baseline, is_baseline=True)
_summary_line("Naive round-to-nearest (no Hessian):", ppl_naive)
_summary_line("Hessian-weighted W4A16, raw basis:", ppl_quantized)
_summary_line("Hessian-weighted W4A16, Hadamard-rotated basis:", ppl_w4a16_had)
_summary_line("W4A4 (Hadamard-rotated + GF4-quantized acts):", ppl_w4a4)
print()

# Commentary only where BOTH operands were computed in this same process.
if ppl_quantized is not None and ppl_naive is not None:
    if ppl_quantized < ppl_naive:
        print(f"Hessian weighting beats naive rounding by {ppl_naive - ppl_quantized:.4f} PPL at identical "
              f"bit width/block size - the kernel's weighting is doing real work.")
    else:
        print(f"Hessian weighting did NOT beat naive rounding in the raw basis (worse by "
              f"{ppl_quantized - ppl_naive:.4f} PPL).")
if ppl_quantized is not None and ppl_w4a16_had is not None:
    print(f"Rotating the weights before the Hessian solve changed PPL by {ppl_w4a16_had - ppl_quantized:+.4f} "
          f"(raw-basis {ppl_quantized:.4f} -> rotated-basis {ppl_w4a16_had:.4f}).")
if ppl_w4a16_had is not None and ppl_w4a4 is not None:
    print(f"Adding GF4 activation quantization on top of the rotated weights changed PPL by "
          f"{ppl_w4a4 - ppl_w4a16_had:+.4f} (rotated-basis-FP16-acts {ppl_w4a16_had:.4f} -> W4A4 {ppl_w4a4:.4f}).")
print()
if CONFIG != "all":
    print(f"[--config {CONFIG}] only the config(s) above ran in this process (no clone; memory")
    print(f"freed on exit). Run the other --config values in separate processes - every row")
    print(f"accumulates in {_args.results_csv}; read it back for the full comparison table.")
    print()
print("Remember: the deck's Table 1 numbers are the FULL pipeline (Hadamard-")
print("rotated activations + GF4 residual passes + outlier retention +")
print("Hessian weight quant together, with clip-ratio calibration and")
print("mean-centering baked into every layer). The three ablations above now")
print("separate that pipeline's components on a real model - a remaining gap")
print("vs. Table 1 is most likely GF4 residual passes and outlier retention,")
print("which this script still does not implement, not a bug in the")
print("Hadamard/Hessian/GF4 kernels themselves.")
