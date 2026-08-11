"""Emit timeloop_gf4/models/<key>_hwsim.json (conv/linear as GEMM + per-layer density)
for EVERY Best_Results_HPO model, via forward hooks (robust to any architecture):
  VGG/LeNet -> the HARVESTED compact model (residual sparsity inside kept channels).
  ResNet/DenseNet -> the ORIGINAL sparse model (they can't be harvested; #3 on their
                     unstructured sparsity is their only accelerator sparsity story).
Timeloop mapping: C = Cin*R*S,  K = Cout,  N(batch) = P*Q  (a conv = GEMM).
Run from Jenks_Tests:  python3 timeloop_gf4/export_all_hwsim.py
"""
import torch, torch.nn as nn, json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
import harvest_all as H   # reuse the exact harvest functions + model builders

def hooked_shapes(model, hw, in_ch=3):
    recs=[]; hooks=[]
    def mk(mod):
        def h(m,inp,out):
            x=inp[0]
            if isinstance(m,nn.Conv2d):
                Cin,R,S=m.in_channels,m.kernel_size[0],m.kernel_size[1]
                P,Q=out.shape[-2],out.shape[-1]; K=m.out_channels
                dens=1-(m.weight.data==0).float().mean().item()
                recs.append(dict(name=f"L{len(recs)}", K=Cin*R*S, N=K, batchN=int(P*Q), count=1,
                                 density=round(dens,4), macs=int(Cin*R*S*K*P*Q)))
            elif isinstance(m,nn.Linear):
                dens=1-(m.weight.data==0).float().mean().item()
                recs.append(dict(name=f"L{len(recs)}", K=m.in_features, N=m.out_features, batchN=1, count=1,
                                 density=round(dens,4), macs=int(m.in_features*m.out_features)))
        return h
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)): hooks.append(m.register_forward_hook(mk(m)))
    model.eval()
    with torch.no_grad(): model(torch.randn(1,in_ch,hw,hw))
    for h in hooks: h.remove()
    return recs

def key(tag): return tag.replace("/","_").replace("-","").lower()

def main():
    os.makedirs(os.path.join(HERE,"models"), exist_ok=True)
    done=0
    for tag, kind, extra in H.JOBS:
        import glob
        cks=[c for c in glob.glob(f"Best_Results_HPO/{tag}/*.pth") if "HARVEST" not in c]
        if not cks: continue
        ck=cks[0]
        try:
            if kind=="vgg":  full,comp,shape=H.harvest_vgg(ck, 64 if "tiny" in ck.lower() else 32); mdl=comp; inh=3; hw=shape[-1]
            elif kind=="l300": full,comp,shape=H.harvest_l300(ck); mdl=comp; inh=1; hw=28
            elif kind=="l5":   full,comp,shape=H.harvest_l5(ck); mdl=comp; inh=1; hw=28
            elif kind=="tp":
                build,nc,hw=extra
                full,comp,shape,npr=H.harvest_tp(lambda: build(nc), ck, nc, hw)
                mdl=full; inh=3        # ResNet/DenseNet: not harvested -> ORIGINAL sparse model
            recs=hooked_shapes(mdl, hw, inh)
            k=key(tag)
            exp=dict(model=k, source=("harvested" if kind in("vgg","l300","l5") else "original_sparse"),
                     input_hw=hw, note="conv/linear as GEMM: C=Cin*R*S, K=Cout, N=P*Q; density=nonzero frac",
                     shapes=recs, total_macs=sum(r["macs"] for r in recs),
                     effective_macs_sparse=sum(int(r["macs"]*r["density"]) for r in recs))
            json.dump(exp, open(os.path.join(HERE,"models",f"{k}_hwsim.json"),"w"), indent=2)
            tm,es=exp["total_macs"],exp["effective_macs_sparse"]
            print(f"{k:34s} {len(recs):3d} layers  dense {tm/1e6:7.1f}M  sparse-eff {es/1e6:6.1f}M ({tm/max(es,1):4.0f}x)  [{exp['source']}]")
            done+=1
        except Exception as e:
            print(f"{tag:34s} EXPORT ERR: {repr(e)[:70]}")
    print(f"\nwrote {done} model hwsim JSONs to timeloop_gf4/models/")

if __name__=="__main__":
    main()
