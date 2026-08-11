"""Export each harvested compact model (*_HARVESTED_compact.pth) as a
timeloop_gf4/models/<name>_hwsim.json for gen_problems.py, in conv-as-GEMM form
with per-layer residual density (for Sparseloop):
    Timeloop C = Cin*R*S,  K = Cout,  N(batch) = B*P*Q,  density = nonzero fraction.
Run from the repo's Jenks_Tests dir:  python3 timeloop_gf4/export_harvested_hwsim.py
"""
import torch, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # Jenks_Tests
BATCH = 1
VGG_CFG = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']

def vgg_spatial(hw):
    res=[]; r=hw
    for v in VGG_CFG:
        if v=='M': r//=2
        else: res.append(r)
    return res

def clean_name(ckpt):
    rel = ckpt.split("Best_Results_HPO/")[1]
    parts = [p for p in os.path.dirname(rel).split("/")]
    # dedupe and normalize: e.g. VGG-19/CIFAR-100/98_sparsity -> vgg19_cifar100_98_sparsity
    s = "_".join(parts).replace("-", "").lower()
    return s

def export_vgg(ckpt):
    d = torch.load(ckpt, map_location='cpu', weights_only=False); sd = d["state_dict"]
    hw = 64 if "tiny" in ckpt.lower() else 32
    convs = [sd[k] for k in sd if k.startswith('feature.') and k.endswith('.weight') and sd[k].dim()==4]
    res = vgg_spatial(hw)
    shapes=[]
    for i,w in enumerate(convs):
        Cout,Ci,R,S = w.shape; P = res[i]; density = 1-(w==0).float().mean().item()
        shapes.append(dict(name=f"conv{i}", K=Ci*R*S, N=Cout, batchN=BATCH*P*P,
                           count=1, density=round(density,4), macs=Ci*R*S*Cout*P*P))
    return clean_name(ckpt), shapes, hw

def main():
    os.chdir(ROOT)
    os.makedirs(os.path.join(HERE,"models"), exist_ok=True)
    # drop stale/duplicate harvested exports so only clean ones remain
    for f in set(glob.glob(os.path.join(HERE,"models","*vgg19*_hwsim.json")) +
                 glob.glob(os.path.join(HERE,"models","*harvest*_hwsim.json"))):
        os.remove(f)
    for ckpt in sorted(glob.glob("Best_Results_HPO/**/*vgg19*_HARVESTED_compact.pth", recursive=True)):
        name, shapes, hw = export_vgg(ckpt)
        exp = dict(model=name, arch="vgg19", input_hw=hw,
                   note="conv-as-GEMM: C=Cin*R*S, K=Cout, N=B*P*Q; density=nonzero frac (Sparseloop)",
                   shapes=shapes, total_macs=sum(s["macs"] for s in shapes),
                   effective_macs_sparse=sum(int(s["macs"]*s["density"]) for s in shapes))
        out = os.path.join(HERE,"models",f"{name}_hwsim.json")
        json.dump(exp, open(out,"w"), indent=2)
        tm, es = exp["total_macs"], exp["effective_macs_sparse"]
        print(f"{name:30s} {len(shapes)} convs  dense {tm/1e6:6.1f}M  sparse-eff {es/1e6:5.1f}M "
              f"({tm/max(es,1):4.0f}x)  -> models/{name}_hwsim.json")

if __name__ == "__main__":
    main()
