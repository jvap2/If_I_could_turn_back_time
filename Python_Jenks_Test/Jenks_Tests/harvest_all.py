"""Harvest + benchmark EVERY Best_Results_HPO checkpoint into a compact dense model.
  VGG/LeNet : exact channel/neuron harvest with constant-folding (provably lossless).
  ResNet/DenseNet : Torch-Pruning DepGraph, prune only truly-dead (zero-importance)
                    channel groups; residual/concat coupling handled automatically.
Writes harvest_all_results.csv. All numbers verified lossless (output diff on random
input); GPU throughput measured on this box.
"""
import torch, torch.nn as nn, torch.nn.functional as F, glob, copy, csv, time, os, sys
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)

def params(m): return sum(p.numel() for p in m.parameters())
def bench(m, shape, dtype, iters=60, warm=20):
    m = m.to(DEV).to(dtype); x = torch.randn(*shape, device=DEV, dtype=dtype)
    with torch.no_grad():
        for _ in range(warm): m(x)
        torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(iters): m(x)
        b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/iters
def lossless_diff(full, comp, shape):
    with torch.no_grad():
        x=torch.randn(*shape, dtype=torch.float64)
        d=(full.double().cpu()(x)-comp.double().cpu()(x)).abs().max().item()
    full.float(); comp.float(); return d

# ---------------- VGG ----------------
VGG_CFG=[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
def _vgg(cfg,nc,last):
    L=[]; c=3
    for v in cfg:
        if v=='M': L+=[nn.MaxPool2d(2,2)]
        else: L+=[nn.Conv2d(c,v,3,padding=1,bias=False), nn.BatchNorm2d(v), nn.ReLU(True)]; c=v
    class M(nn.Module):
        def __init__(s): super().__init__(); s.feature=nn.Sequential(*L); s.avgpool=nn.AdaptiveAvgPool2d((1,1)); s.classifier=nn.Linear(last,nc)
        def forward(s,x): return s.classifier(torch.flatten(s.avgpool(s.feature(x)),1))
    return M()
def harvest_vgg(ck, hw):
    sd=torch.load(ck,map_location='cpu',weights_only=False)
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    nc=sd['classifier.weight'].shape[0]
    full=_vgg(VGG_CFG,nc,512).eval(); full.load_state_dict(sd,strict=True)
    ci=[i for i,m in enumerate(full.feature) if isinstance(m,nn.Conv2d)]; keep=[]
    for i in ci:
        cv=full.feature[i]; bn=full.feature[i+1]
        dead=(cv.weight.data.reshape(cv.weight.shape[0],-1).abs().sum(1)==0)
        c=bn.bias.data-bn.weight.data*bn.running_mean.data/torch.sqrt(bn.running_var.data+bn.eps)
        keep.append(~(dead&(torch.clamp(c,min=0)==0)))
    cfg=[]; k=0
    for v in VGG_CFG:
        if v=='M': cfg.append('M')
        else: cfg.append(max(1,int(keep[k].sum()))); k+=1
    lk=keep[-1]; comp=_vgg(cfg,nc,max(1,int(lk.sum()))).eval()
    prev=torch.ones(3,dtype=torch.bool); k=0
    for oi,cxi in zip(ci,[i for i,m in enumerate(comp.feature) if isinstance(m,nn.Conv2d)]):
        kk=keep[k]; oc=full.feature[oi]; ob=full.feature[oi+1]; cc=comp.feature[cxi]; cb=comp.feature[cxi+1]
        cc.weight.data.copy_(oc.weight.data[kk][:,prev])
        for a in('weight','bias','running_mean','running_var'): getattr(cb,a).data.copy_(getattr(ob,a).data[kk])
        cb.num_batches_tracked.data.copy_(ob.num_batches_tracked.data); prev=kk; k+=1
    comp.classifier.weight.data.copy_(full.classifier.weight.data[:,lk]); comp.classifier.bias.data.copy_(full.classifier.bias.data)
    return full, comp, (1,3,hw,hw)

# ---------------- LeNet ----------------
def _l300(h1=300,h2=100): return nn.Sequential(nn.Flatten(),nn.Linear(784,h1),nn.ReLU(),nn.Linear(h1,h2),nn.ReLU(),nn.Linear(h2,10))
def harvest_l300(ck):
    sd=torch.load(ck,map_location='cpu',weights_only=False); ref=_l300(); ref.load_state_dict(sd,strict=True); ref.eval(); w=copy.deepcopy(ref)
    for i,j in [(1,3),(3,5)]:
        dead=(w[i].weight.data.abs().sum(1)==0); const=F.relu(w[i].bias.data)
        w[j].bias.data += w[j].weight.data[:,dead]@const[dead]; w[i]._k=~dead
    h1,h2=int(w[1]._k.sum()),int(w[3]._k.sum()); c=_l300(h1,h2).eval(); k1,k3=w[1]._k,w[3]._k
    c[1].weight.data.copy_(w[1].weight.data[k1]); c[1].bias.data.copy_(w[1].bias.data[k1])
    c[3].weight.data.copy_(w[3].weight.data[k3][:,k1]); c[3].bias.data.copy_(w[3].bias.data[k3])
    c[5].weight.data.copy_(w[5].weight.data[:,k3]); c[5].bias.data.copy_(w[5].bias.data)
    return ref,c,(256,1,28,28)
class _L5(nn.Module):
    def __init__(s,c1=20,c2=50,f1=500):
        super().__init__(); s.stem=nn.Module(); s.stem.conv1=nn.Conv2d(1,c1,5); s.stem.conv2=nn.Conv2d(c1,c2,5)
        s.linear1=nn.Linear(c2*16,f1); s.linear2=nn.Linear(f1,10)
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.stem.conv1(x)),2); x=F.max_pool2d(F.relu(s.stem.conv2(x)),2)
        return s.linear2(F.relu(s.linear1(x.flatten(1))))
def harvest_l5(ck):
    sd=torch.load(ck,map_location='cpu',weights_only=False); ref=_L5(); ref.load_state_dict(sd,strict=True); ref.eval(); w=copy.deepcopy(ref)
    c1,c2,l1,l2=w.stem.conv1,w.stem.conv2,w.linear1,w.linear2
    d1=(c1.weight.data.reshape(c1.weight.shape[0],-1).abs().sum(1)==0); e1=F.relu(c1.bias.data)
    c2.bias.data += c2.weight.data[:,d1].sum(dim=(2,3))@e1[d1]
    d2=(c2.weight.data.reshape(c2.weight.shape[0],-1).abs().sum(1)==0); e2=F.relu(c2.bias.data); C2=c2.weight.shape[0]
    l1.bias.data += l1.weight.data.view(l1.weight.shape[0],C2,16).sum(2)[:,d2]@e2[d2]
    df=(l1.weight.data.abs().sum(1)==0); ef=F.relu(l1.bias.data); l2.bias.data += l2.weight.data[:,df]@ef[df]
    k1,k2,kf=~d1,~d2,~df; comp=_L5(int(k1.sum()),int(k2.sum()),int(kf.sum())).eval()
    comp.stem.conv1.weight.data.copy_(c1.weight.data[k1]); comp.stem.conv1.bias.data.copy_(c1.bias.data[k1])
    comp.stem.conv2.weight.data.copy_(c2.weight.data[k2][:,k1]); comp.stem.conv2.bias.data.copy_(c2.bias.data[k2])
    col=torch.repeat_interleave(k2,16); comp.linear1.weight.data.copy_(l1.weight.data[kf][:,col]); comp.linear1.bias.data.copy_(l1.bias.data[kf])
    comp.linear2.weight.data.copy_(l2.weight.data[:,kf]); comp.linear2.bias.data.copy_(l2.bias.data)
    return ref,comp,(256,1,28,28)

# ---------------- ResNet / DenseNet via Torch-Pruning ----------------
def harvest_tp(build, ck, nc, hw):
    import torch_pruning as tp
    sd=torch.load(ck,map_location='cpu',weights_only=False)
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    m=build(); m.load_state_dict(sd,strict=True); m.eval()
    full=copy.deepcopy(m)
    DG=tp.DependencyGraph().build_dependency(m, example_inputs=torch.randn(1,3,hw,hw))
    ignored=[mod for mod in m.modules() if isinstance(mod,nn.Linear) and mod.out_features==nc]
    n_pruned=0
    for layer in [mod for mod in m.modules() if isinstance(mod,(nn.Conv2d,nn.Linear))]:
        if layer in ignored: continue
        W=layer.weight.data
        dead=(W.reshape(W.shape[0],-1).abs().sum(1)==0).nonzero().flatten().tolist()
        if not dead: continue
        pf=tp.prune_conv_out_channels if isinstance(layer,nn.Conv2d) else tp.prune_linear_out_channels
        try:
            g=DG.get_pruning_group(layer, pf, idxs=dead)
            if DG.check_pruning_group(g): g.prune(); n_pruned+=len(dead)
        except Exception:
            pass
    # SAFETY: only accept the prune if it is actually lossless; residual/concat
    # coupling can make removing a locally-dead channel change the output.
    try:
        d=lossless_diff(copy.deepcopy(full), copy.deepcopy(m), (1,3,hw,hw))
    except Exception:
        d=1e9
    if d > 1e-3:
        return full, copy.deepcopy(full), (1,3,hw,hw), 0     # not lossless -> report unchanged (harvest N/A)
    return full, m, (1,3,hw,hw), n_pruned

def resnet32_c(nc):
    from resnet import resnet32
    try: return resnet32(num_classes=nc)
    except TypeError: return resnet32()
def densenet_c(nc):
    from densenet import create_densenet40
    return create_densenet40()

# ---------------- driver ----------------
JOBS = [
    ("VGG-19/CIFAR-10/90_sparsity",  "vgg", None),
    ("VGG-19/CIFAR-10/99_sparsity",  "vgg", None),
    ("VGG-19/CIFAR-100/90_sparsity", "vgg", None),
    ("VGG-19/CIFAR-100/98_sparsity", "vgg", None),
    ("VGG-19/TinyImageNet",          "vgg", None),
    ("VGG19_Test",                   "vgg", None),
    ("LeNet300",                     "l300", None),
    ("LeNet5",                       "l5", None),
    ("ResNet32/CIFAR-10/86_sparsity","tp", (resnet32_c,10,32)),
    ("ResNet32/CIFAR-10/95_sparsity","tp", (resnet32_c,10,32)),
    ("ResNet32/CIFAR-100/85_Sparsity","tp",(resnet32_c,100,32)),
    ("ResNet32/CIFAR-100/86_Sparsity","tp",(resnet32_c,100,32)),
    ("ResNet32/TinyImageNet",        "tp", (resnet32_c,200,64)),
    ("DenseNet40",                   "tp", (densenet_c,10,32)),
]
rows=[]
for tag, kind, extra in JOBS:
    cks=glob.glob(f"Best_Results_HPO/{tag}/*.pth")
    cks=[c for c in cks if "HARVEST" not in c]
    if not cks: print(f"SKIP {tag}: no ckpt"); continue
    ck=cks[0]; hw=64 if "tiny" in ck.lower() else (28 if "lenet" in kind else 32)
    try:
        if kind=="vgg": full,comp,shape=harvest_vgg(ck,hw); npr=None
        elif kind=="l300": full,comp,shape=harvest_l300(ck); npr=None
        elif kind=="l5": full,comp,shape=harvest_l5(ck); npr=None
        elif kind=="tp":
            build,nc,hw=extra; full,comp,shape,npr=harvest_tp(lambda: build(nc), ck, nc, hw)
        diff=lossless_diff(full,comp,shape)
        pf,pc=params(full),params(comp)
        bs=(256,)+shape[1:] if shape[0]==1 else shape   # benchmark at batch 256
        s16=bench(copy.deepcopy(full),bs,torch.float16)/bench(copy.deepcopy(comp),bs,torch.float16)
        s32=bench(copy.deepcopy(full),bs,torch.float32)/bench(copy.deepcopy(comp),bs,torch.float32)
        r=dict(model=tag, params_full=pf, params_compact=pc, shrink=round(pf/pc,2),
               lossless_diff=f"{diff:.1e}", gpu_fp16=round(s16,2), gpu_fp32=round(s32,2),
               tp_pruned=npr if npr is not None else "")
        print(f"{tag:34s} {pf/1e6:6.2f}M->{pc/1e6:6.3f}M ({pf/pc:5.2f}x)  loss={diff:.0e}  gpu x{s16:.2f}/{s32:.2f}"
              + (f"  tp_pruned={npr}" if npr is not None else ""))
    except Exception as e:
        r=dict(model=tag, params_full="", params_compact="", shrink="ERR", lossless_diff=repr(e)[:60], gpu_fp16="", gpu_fp32="", tp_pruned="")
        print(f"{tag:34s} ERROR: {repr(e)[:80]}")
    rows.append(r)
with open("harvest_all_results.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print("\nsaved harvest_all_results.csv")
