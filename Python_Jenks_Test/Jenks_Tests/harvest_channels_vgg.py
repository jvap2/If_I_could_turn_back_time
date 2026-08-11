"""#1 generalize: harvest emergent dead channels from every sparse VGG-19 checkpoint
into a compact DENSE model, prove lossless, benchmark GPU throughput. Handles any
num_classes / input resolution (inferred from the checkpoint + path)."""
import torch, torch.nn as nn, time, copy, os, glob, csv

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']

def make(channel_cfg, in_ch=3):
    layers=[]; c=in_ch
    for v in channel_cfg:
        if v=='M': layers+=[nn.MaxPool2d(2,2)]
        else: layers+=[nn.Conv2d(c,v,3,padding=1,bias=False), nn.BatchNorm2d(v), nn.ReLU(True)]; c=v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(s, channel_cfg, nc, last_ch):
        super().__init__(); s.feature=make(channel_cfg); s.avgpool=nn.AdaptiveAvgPool2d((1,1)); s.classifier=nn.Linear(last_ch,nc)
    def forward(s,x): return s.classifier(torch.flatten(s.avgpool(s.feature(x)),1))

def harvest(ckpt, hw):
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    nc = sd['classifier.weight'].shape[0]
    full = VGG(CFG, nc, 512).eval(); full.load_state_dict(sd, strict=True)

    conv_idx=[i for i,m in enumerate(full.feature) if isinstance(m,nn.Conv2d)]
    keep=[]
    for i in conv_idx:
        conv=full.feature[i]; bn=full.feature[i+1]
        dead=(conv.weight.data.reshape(conv.weight.shape[0],-1).abs().sum(1)==0)
        c=bn.bias.data - bn.weight.data*bn.running_mean.data/torch.sqrt(bn.running_var.data+bn.eps)
        truly0 = dead & (torch.clamp(c,min=0)==0)
        keep.append(~truly0)
    compact_cfg=[]; ci=0
    for v in CFG:
        if v=='M': compact_cfg.append('M')
        else: compact_cfg.append(max(1,int(keep[ci].sum()))); ci+=1
    last_keep=keep[-1]
    compact=VGG(compact_cfg, nc, max(1,int(last_keep.sum()))).eval()
    prev=torch.ones(3,dtype=torch.bool); ci=0
    for oi,cix in zip(conv_idx,[i for i,m in enumerate(compact.feature) if isinstance(m,nn.Conv2d)]):
        k=keep[ci]; oconv=full.feature[oi]; obn=full.feature[oi+1]
        cconv=compact.feature[cix]; cbn=compact.feature[cix+1]
        cconv.weight.data.copy_(oconv.weight.data[k][:,prev])
        for a in ('weight','bias','running_mean','running_var'): getattr(cbn,a).data.copy_(getattr(obn,a).data[k])
        cbn.num_batches_tracked.data.copy_(obn.num_batches_tracked.data); prev=k; ci+=1
    compact.classifier.weight.data.copy_(full.classifier.weight.data[:,last_keep])
    compact.classifier.bias.data.copy_(full.classifier.bias.data)

    # lossless check in float64 on CPU
    with torch.no_grad():
        x=torch.randn(4,3,hw,hw,dtype=torch.float64)
        diff=(full.double()(x)-compact.double()(x)).abs().max().item()
    full.float(); compact.float()
    return full, compact, nc, diff, compact_cfg

def params(m): return sum(p.numel() for p in m.parameters())
def bench(m, bs, hw, dtype, iters=60, warm=20):
    m=m.to(DEV).to(dtype); x=torch.randn(bs,3,hw,hw,device=DEV,dtype=dtype)
    with torch.no_grad():
        for _ in range(warm): m(x)
        torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(iters): m(x)
        b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/iters

torch.backends.cudnn.benchmark=True
rows=[]
for ckpt in sorted(glob.glob("Best_Results_HPO/**/*vgg19*.pth", recursive=True)) + \
            sorted(glob.glob("Best_Results_HPO/**/*vgg19*.pth".replace("vgg19","VGG"), recursive=True)):
    if "HARVESTED" in ckpt: continue
    hw = 64 if "tiny" in ckpt.lower() else 32
    try:
        full, compact, nc, diff, ccfg = harvest(ckpt, hw)
    except Exception as e:
        print(f"SKIP {ckpt}: {e}"); continue
    pf, pc = params(full), params(compact)
    sp16 = bench(copy.deepcopy(full),256,hw,torch.float16)/bench(copy.deepcopy(compact),256,hw,torch.float16)
    sp32 = bench(copy.deepcopy(full),256,hw,torch.float32)/bench(copy.deepcopy(compact),256,hw,torch.float32)
    tag="/".join(ckpt.split("Best_Results_HPO/")[1].split("/")[:-1])
    rows.append(dict(net=tag, nc=nc, hw=hw, params_full_M=round(pf/1e6,2), params_compact_M=round(pc/1e6,3),
                     shrink=round(pf/pc,1), lossless_diff=f"{diff:.1e}", sp_fp16_bs256=round(sp16,2), sp_fp32_bs256=round(sp32,2)))
    out=ckpt.replace(".pth","_HARVESTED_compact.pth")
    torch.save({"state_dict":compact.state_dict(),"channel_cfg":ccfg,"num_classes":nc}, out)
    print(f"{tag:38s} nc={nc:3d} hw={hw}  {pf/1e6:5.1f}M->{pc/1e6:5.2f}M ({pf/pc:4.1f}x)  "
          f"lossless={diff:.0e}  GPU x{sp16:.2f}(fp16)/x{sp32:.2f}(fp32)")

with open("harvest_results.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
print("\nsaved harvest_results.csv")
