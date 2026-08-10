"""
Confirm the opt-13b W4A16 collapse hypothesis: does the pad-to-next-pow2 Hadamard
FAIL to decorrelate opt-13b's severe emergent activation outliers?

For every residual-stream-reading linear we capture the input activation x, apply
the SAME rotation the quant pipeline uses  ( x_had = FWHT( pad(x) * D ) ,
had_block_size="auto" => P = next_pow2(in_features), random ±1 signs D seeded per
layer ), and compare the OUTLIER heaviness before vs after the rotation:

    ratio  = max|·| / mean|·|      (scale-invariant; high = spiky/outlier-heavy)
    kurt   = excess kurtosis        (heavy-tail measure)
    smear  = ratio_raw / ratio_rot  (how much the Hadamard flattened outliers;
                                      >>1 = worked, ~1 = outliers SURVIVED)

Expected if the mechanism is right:
    opt-6.7b (0% pad) and Llama-2-13b (mild outliers): rotated ratio LOW, smear HIGH.
    opt-13b (60% pad + severe outliers):               rotated ratio still HIGH, smear ~1.

Usage (one or more HF ids):
    python3 diag_hadamard_outliers.py facebook/opt-6.7b facebook/opt-13b meta-llama/Llama-2-13b-hf
Env: NSEQ (default 2), SEQLEN (default 512).  Big models load via device_map="auto".
"""
import os, sys, math, hashlib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

NSEQ   = int(os.environ.get("NSEQ", 2))
SEQLEN = int(os.environ.get("SEQLEN", 256))
WTOK   = int(os.environ.get("WTOK", 128))   # tokens used for the weight output-error matmul

# residual-stream reads (post-LayerNorm) — where OPT's emergent outliers live
IN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "fc1", "gate_proj", "up_proj")

# GF4_POS dequant levels (positive half) — from cuda_helpers.GF4_LEVELS
GF4 = torch.tensor([0.0, 0.0796082, 0.1737177, 0.2828685,
                    0.3952704, 0.5250730, 0.6961928, 1.0])
# E2M1 (FP4/NVFP4) positive levels {0,.5,1,1.5,2,3,4,6} normalized to max=1 — the
# codebook the pipeline's v5 solver actually uses for WEIGHTS.
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]) / 6.0


def cb_quant_blocks(W, levels, bs=16, clip=2.5):
    """Per-16-block 4-bit quantize along the last dim (rms*clip scale) using the
    given magnitude codebook + a sign. IDENTICAL scaling for every codebook and
    model, so the comparison isolates the CODEBOOK (GF4 vs E2M1)."""
    N, P = W.shape
    pad = (bs - P % bs) % bs
    if pad:
        W = F.pad(W, (0, pad))
    Wb    = W.view(N, -1, bs)
    scale = (Wb.pow(2).mean(-1, keepdim=True).sqrt() * clip).clamp_min(1e-8)
    x     = (Wb / scale).clamp(-1, 1)
    lv    = levels.to(W.device)
    idx   = (x.abs().unsqueeze(-1) - lv).abs().argmin(-1)   # nearest level
    q     = lv[idx] * x.sign()
    return (q * scale).view(N, -1)[:, :P]


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _stable_signs(name, P, device):
    """Deterministic ±1 sign vector, same recipe as generate_random_signs+_stable_seed."""
    seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, (P,), generator=g).float() * 2 - 1).to(device)


def fwht(a):
    """Normalized fast Walsh-Hadamard transform over the last dim (power of 2)."""
    P = a.shape[-1]
    a = a.clone()
    flat = a.view(-1, P)
    h = 1
    while h < P:
        flat = flat.view(flat.shape[0], -1, 2 * h)
        x = flat[:, :, :h].clone()
        y = flat[:, :, h:2 * h].clone()
        flat[:, :, :h] = x + y
        flat[:, :, h:2 * h] = x - y
        flat = flat.view(flat.shape[0], P)
        h *= 2
    return (flat / math.sqrt(P)).view(*a.shape)


def _stats(v):
    """abs-max/mean ratio and excess kurtosis of a flattened tensor (fp32)."""
    v = v.reshape(-1).float()
    av = v.abs()
    ratio = (av.max() / av.mean().clamp_min(1e-12)).item()
    m = v.mean(); s = v.std().clamp_min(1e-12)
    kurt = (((v - m) / s) ** 4).mean().item() - 3.0
    return ratio, kurt


def run_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    model.eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt",
              add_special_tokens=False).input_ids[0]

    # suffix -> [raw_ratio, raw_kurt, rot_ratio, rot_kurt, n_act, gf4_orel, e2m1_orel, n_w]
    acc = {}
    skipped_meta = [0]
    handles = []

    def make_hook(name, suffix):
        def hook(mod, args):                          # forward-PRE hook: weight is live
            x = args[0].detach()
            M = x.shape[-1]
            P = _next_pow2(M)
            dev = x.device
            xf = x.reshape(-1, M).float()
            if P != M:
                xf = F.pad(xf, (0, P - M))
            D = _stable_signs(name, P, dev)
            xr = fwht(xf * D)
            rr, rk = _stats(x)                        # raw activation
            tr, tk = _stats(xr)                       # rotated activation
            W = mod.weight
            is_meta = (W.device.type == "meta")       # offloaded & not materialized
            if is_meta:
                skipped_meta[0] += 1
                gf4_o = e2m1_o = float("nan")
            else:
                # weight metric ENTIRELY on CPU — never competes with the ~22GB
                # device_map forward on the GPU (that OOMed before). Quantize the
                # SAME rotated weight two ways (GF4 vs the pipeline's E2M1) with
                # identical scaling, and measure how much each corrupts the real
                # layer output x_had @ W_had.T.
                Wc = W.detach().float().cpu()          # [N, M]
                Dc = _stable_signs(name, P, "cpu")
                Wp = F.pad(Wc, (0, P - M)) if P != M else Wc
                W_had = fwht(Wp * Dc)                  # rotate weight the SAME way
                xc  = xr[:WTOK].detach().float().cpu()
                ref = xc @ W_had.T                    # == x @ W.T (true pre-activation)
                rn  = ref.norm().clamp_min(1e-9)
                gf4_o  = ((xc @ cb_quant_blocks(W_had, GF4).T  - ref).norm() / rn).item()
                e2m1_o = ((xc @ cb_quant_blocks(W_had, E2M1).T - ref).norm() / rn).item()
                del Wc, Wp, W_had, xc, ref
            # [rr, rk, tr, tk, n_act, gf4_orel, e2m1_orel, n_w]
            a = acc.setdefault(suffix, [0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0])
            a[0] += rr; a[1] += rk; a[2] += tr; a[3] += tk; a[4] += 1
            if not is_meta:
                a[5] += gf4_o; a[6] += e2m1_o; a[7] += 1
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(IN_SUFFIXES):
            handles.append(mod.register_forward_pre_hook(make_hook(name, name.split(".")[-1])))

    dev0 = model.get_input_embeddings().weight.device
    n = ids.size(0) // SEQLEN
    step = max(n // max(NSEQ, 1), 1)
    with torch.inference_mode():
        for i in range(NSEQ):
            s = (i * step) * SEQLEN
            chunk = ids[s:s + SEQLEN].unsqueeze(0).to(dev0)
            model(chunk, use_cache=False)
    for h in handles:
        h.remove()

    hid = getattr(model.config, "hidden_size", None)
    P = _next_pow2(hid) if hid else None
    pad = (1 - hid / P) * 100 if P else float("nan")
    print(f"\n===== {model_id}   hidden={hid} -> pad {P} ({pad:.0f}% zero-pad) =====")
    print(f"  {'layer':10} {'rot max/mean':>13} {'rot kurt':>9} "
          f"{'GF4 out relerr':>15} {'E2M1 out relerr':>16}")
    gf4_rels, e2m1_rels = [], []
    for suf in [s for s in ("q_proj", "k_proj", "v_proj", "fc1", "gate_proj", "up_proj")
                if s in acc]:
        rr, rk, tr, tk, na, gs, es, nw = acc[suf]
        tr, tk = tr / na, tk / na
        g = gs / nw if nw else float("nan")
        e = es / nw if nw else float("nan")
        if nw:
            gf4_rels.append(g); e2m1_rels.append(e)
        print(f"  {suf:10} {tr:13.1f} {tk:9.1f} {g:15.3f} {e:16.3f}")
    if gf4_rels:
        gmed = sorted(gf4_rels)[len(gf4_rels) // 2]
        emed = sorted(e2m1_rels)[len(e2m1_rels) // 2]
        print(f"  --> median OUTPUT relerr:  GF4={gmed:.3f}   E2M1={emed:.3f}   "
              f"(E2M1 >> GF4 on opt-13b only => E2M1 weight codebook is the collapse)")
    if skipped_meta[0]:
        print(f"  (note: {skipped_meta[0]} layer-calls had offloaded weights; "
              f"weight metrics use the materialized ones)")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "facebook/opt-6.7b", "facebook/opt-13b", "meta-llama/Llama-2-13b-hf"]
    for m in models:
        try:
            run_model(m)
        except Exception as e:
            print(f"\n[skip] {m}: {type(e).__name__}: {e}")
