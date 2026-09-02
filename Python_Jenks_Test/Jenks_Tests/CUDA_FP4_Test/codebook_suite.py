"""
codebook_suite.py -- evaluate the candidate 4-bit codebooks across a SUITE of
OPT and Llama models before committing to one.

Compares four codebooks (all A-only GF4 under a blockwise Hadamard, weights fp16):
  exact_GF4   : continuous Gaussian-quantile levels (the hand-tuned format)
  round_Q1.4  : GF4 rounded to the 1/16 grid  {0,1,3,5,6,8,11,16}/16
  opt_Q1.4    : constrained-optimal on the 1/16 grid {0,2,4,6,8,10,13,16}/16
  opt_pop2    : + shift-and-add constraint (popcount<=2) {0,2,4,6,8,10,12,16}/16

Reuses the tested machinery in gf4_regularity_colab.py (rotation, quant, PPL) so
the numbers are directly comparable to the single-model study. Frees each model
before the next so a suite fits sequentially; device_map="auto" spreads big
models across GPU+CPU. WikiText-2 perplexity is the metric.

Usage:
  python3 codebook_suite.py --models facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b
  python3 codebook_suite.py --models meta-llama/Llama-2-7b-hf --load-8bit   # to FIT
"""
import argparse, csv, gc, sys
import numpy as np
import torch
import gf4_regularity_colab as G

CODEBOOKS = {
    "exact_GF4":  G.GF4_LEVEL,
    "round_Q1.4": np.array([0, 1, 3, 5, 6, 8, 11, 16]) / 16,
    "opt_Q1.4":   np.array([0, 2, 4, 6, 8, 10, 13, 16]) / 16,
    "opt_pop2":   np.array([0, 2, 4, 6, 8, 10, 12, 16]) / 16,
}
COLS = list(CODEBOOKS) + ["solved_opt", "solved_pop2"]   # + the per-model solved codebooks


def eval_model(name, windows_n, seqlen, scale_mode, load_8bit, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=False)
    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if load_8bit:
        kw.update(load_in_8bit=True); kw.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    windows = G.load_windows(tok, seqlen, windows_n)
    dev = next(model.parameters()).device

    G._SCALE_MODE = scale_mode
    ppl_fp16 = G.perplexity(model, windows)
    G.install_hooks(model, seed)
    # SOLVE the constrained-optimal codebook on THIS model's own activations
    mags = G.collect_mags(model, windows)
    s_lv, s_ks = G.solve_constrained_codebook(mags)
    s_lv2, s_ks2 = G.solve_constrained_codebook(mags, popcount=2)
    cbs = dict(CODEBOOKS); cbs["solved_opt"] = s_lv; cbs["solved_pop2"] = s_lv2
    out = {"fp16": ppl_fp16, "solved_ks": str(list(s_ks)), "solved_pop2_ks": str(list(s_ks2))}
    for cb, levels in cbs.items():
        G.set_codebook(levels, dev)
        out[cb] = G.perplexity(model, windows)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out, len(windows)


def purge_hf_cache(name):
    """Delete one model's HF cache so peak disk = the largest single model, not
    the sum (Colab disk fills otherwise). Only removes the given repo."""
    import shutil
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        base = HF_HUB_CACHE
    except Exception:
        base = os.path.expanduser("~/.cache/huggingface/hub")
    d = os.path.join(base, "models--" + name.replace("/", "--"))
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"  [purged cache: {d}]")


def disk_free_gb(path="/"):
    import shutil
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("nan")


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fp16"] + COLS + ["solved_ks", "solved_pop2_ks"])
        for name, base, res in rows:
            w.writerow([name, f"{base:.4f}"] + [f"{res[c]:.4f}" for c in COLS]
                       + [res.get("solved_ks", ""), res.get("solved_pop2_ks", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b")
    ap.add_argument("--eval-windows", type=int, default=10000,
                    help="Non-overlapping 2048-tok windows; default caps to the FULL WikiText-2 "
                         "test set (load_windows uses min(n, tokens//seqlen)). Set small only for "
                         "a quick smoke test -- a subset gives optimistic PPL not comparable to the "
                         "full-set big-table / literature numbers.")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--scale-mode", default="fp16", choices=["fp16", "e4m3"])
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--weights", default="nvfp4", choices=["nvfp4", "fp16"],
                    help="nvfp4 = W4A4 (NVFP4/E2M1 weights + codebook activations); fp16 = A-only")
    ap.add_argument("--purge-cache", action="store_true",
                    help="delete each model's HF cache after eval (keeps Colab disk from filling)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="codebook_suite_results.csv")
    args = ap.parse_args()
    G.QUANT_WEIGHTS_NVFP4 = (args.weights == "nvfp4")
    print(f"weights: {args.weights}  ({'W4A4' if G.QUANT_WEIGHTS_NVFP4 else 'A-only, W16'})")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for name in models:
        print(f"\n===== {name} =====  (disk free {disk_free_gb():.0f} GB)", flush=True)
        try:
            res, nw = eval_model(name, args.eval_windows, args.seqlen,
                                 args.scale_mode, args.load_8bit, args.seed)
        except Exception as e:
            print(f"  SKIP ({type(e).__name__}: {str(e)[:120]})", flush=True)
            if args.purge_cache:
                purge_hf_cache(name)              # reclaim a partial download too
            continue
        base = res["fp16"]; ex = res["exact_GF4"]
        print(f"  fp16 {base:.4f}  ({nw} windows)   SOLVED opt={res['solved_ks']} pop2={res['solved_pop2_ks']}")
        for cb in COLS:
            print(f"    {cb:12s} PPL {res[cb]:8.4f}   dPPL(fp16) {res[cb]-base:+7.3f}   "
                  f"dPPL(exact) {res[cb]-ex:+7.3f}")
        rows.append((name, base, res))
        # write CSV incrementally so results survive an OOM/disk-full crash later
        _write_csv(args.out, rows)
        if args.purge_cache:
            purge_hf_cache(name)                  # free disk before the next (bigger) model

    print(f"\nwrote {args.out}")

    if rows:
        print("\n=== dPPL vs exact_GF4 (negative = beats hand-tuned GF4) ===")
        print("model".ljust(26) + "".join(c.rjust(13) for c in COLS))
        for name, base, res in rows:
            print(name.ljust(26) + "".join(f"{res[c]-res['exact_GF4']:+13.3f}" for c in COLS))
        print("\n=== solved-per-model codebook (matches global {0,2,4,6,8,10,13,16}?) ===")
        for name, base, res in rows:
            print(f"  {name:26s} solved_opt={res['solved_ks']:28s} solved_pop2={res['solved_pop2_ks']}")


if __name__ == "__main__":
    main()
