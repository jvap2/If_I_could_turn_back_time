"""#1 harvest for the two LeNets (MNIST), WITH constant-folding so ALL dead-weight
units are removed losslessly: a dead unit emits const=ReLU(bias); we fold that
constant into the next layer's bias, then delete the unit + its downstream fan-in."""
import torch, torch.nn as nn, torch.nn.functional as F, glob, copy

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def lenet300(h1=300, h2=100):
    return nn.Sequential(nn.Flatten(), nn.Linear(784,h1), nn.ReLU(),
                         nn.Linear(h1,h2), nn.ReLU(), nn.Linear(h2,10))

def harvest_mlp(ckpt):
    sd=torch.load(ckpt,map_location='cpu',weights_only=False)
    ref=lenet300(); ref.load_state_dict(sd,strict=True); ref.eval()
    w=copy.deepcopy(ref)                                   # mutate the working copy, keep ref pristine
    # hidden layer i -> next linear j: fold dead-unit constants into j.bias, drop them
    for i,j in [(1,3),(3,5)]:
        dead=(w[i].weight.data.abs().sum(1)==0)
        const=F.relu(w[i].bias.data)                       # constant a dead unit emits
        w[j].bias.data += w[j].weight.data[:,dead] @ const[dead]   # fold
        w[i]._keep = ~dead
    h1,h2=int(w[1]._keep.sum()),int(w[3]._keep.sum())
    c=lenet300(h1,h2).eval(); k1,k3=w[1]._keep,w[3]._keep
    c[1].weight.data.copy_(w[1].weight.data[k1]);           c[1].bias.data.copy_(w[1].bias.data[k1])
    c[3].weight.data.copy_(w[3].weight.data[k3][:,k1]);     c[3].bias.data.copy_(w[3].bias.data[k3])
    c[5].weight.data.copy_(w[5].weight.data[:,k3]);         c[5].bias.data.copy_(w[5].bias.data)
    return ref,c

class LeNet5(nn.Module):
    def __init__(s,c1=20,c2=50,f1=500):
        super().__init__(); s.stem=nn.Module()
        s.stem.conv1=nn.Conv2d(1,c1,5); s.stem.conv2=nn.Conv2d(c1,c2,5)
        s.linear1=nn.Linear(c2*16,f1); s.linear2=nn.Linear(f1,10)
    def forward(s,x):
        x=F.max_pool2d(F.relu(s.stem.conv1(x)),2)
        x=F.max_pool2d(F.relu(s.stem.conv2(x)),2)
        return s.linear2(F.relu(s.linear1(x.flatten(1))))

def harvest_lenet5(ckpt):
    sd=torch.load(ckpt,map_location='cpu',weights_only=False)
    ref=LeNet5(); ref.load_state_dict(sd,strict=True); ref.eval()
    w=copy.deepcopy(ref)
    c1w,c2w,l1,l2=w.stem.conv1,w.stem.conv2,w.linear1,w.linear2
    # conv1 -> conv2: dead conv1 ch emits const map; fold into conv2.bias
    d1=(c1w.weight.data.reshape(c1w.weight.shape[0],-1).abs().sum(1)==0); e1=F.relu(c1w.bias.data)
    c2w.bias.data += c2w.weight.data[:,d1].sum(dim=(2,3)) @ e1[d1]
    # conv2 -> linear1: dead conv2 ch -> pooled constant -> 16 flat cols; fold into linear1.bias
    d2=(c2w.weight.data.reshape(c2w.weight.shape[0],-1).abs().sum(1)==0); e2=F.relu(c2w.bias.data)
    C2=c2w.weight.shape[0]
    l1.bias.data += l1.weight.data.view(l1.weight.shape[0],C2,16).sum(2)[:,d2] @ e2[d2]
    # linear1 -> linear2: dead linear1 unit -> const; fold into linear2.bias
    df=(l1.weight.data.abs().sum(1)==0); ef=F.relu(l1.bias.data)
    l2.bias.data += l2.weight.data[:,df] @ ef[df]
    k1,k2,kf=~d1,~d2,~df
    nc1,nc2,nf=int(k1.sum()),int(k2.sum()),int(kf.sum())
    c=LeNet5(nc1,nc2,nf).eval()
    c.stem.conv1.weight.data.copy_(c1w.weight.data[k1]);        c.stem.conv1.bias.data.copy_(c1w.bias.data[k1])
    c.stem.conv2.weight.data.copy_(c2w.weight.data[k2][:,k1]);  c.stem.conv2.bias.data.copy_(c2w.bias.data[k2])
    col=torch.repeat_interleave(k2,16)
    c.linear1.weight.data.copy_(l1.weight.data[kf][:,col]);     c.linear1.bias.data.copy_(l1.bias.data[kf])
    c.linear2.weight.data.copy_(l2.weight.data[:,kf]);          c.linear2.bias.data.copy_(l2.bias.data)
    return ref,c

def params(m): return sum(p.numel() for p in m.parameters())
def bench(m,bs,iters=100,warm=30):
    m=m.to(DEV); x=torch.randn(bs,1,28,28,device=DEV)
    with torch.no_grad():
        for _ in range(warm): m(x)
        torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
        for _ in range(iters): m(x)
        b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)/iters
torch.backends.cudnn.benchmark=True

for name,fn,pat in [("LeNet300",harvest_mlp,"Best_Results_HPO/LeNet300/*.pth"),
                    ("LeNet5",harvest_lenet5,"Best_Results_HPO/LeNet5/*.pth")]:
    ck=glob.glob(pat)[0]; m,c=fn(ck)
    with torch.no_grad():
        x=torch.randn(8,1,28,28,dtype=torch.float64); diff=(m.double()(x)-c.double()(x)).abs().max().item()
    m.float(); c.float()
    sp=bench(copy.deepcopy(m),1024)/bench(copy.deepcopy(c),1024)
    print(f"{name:9s} params {params(m)/1e3:5.0f}K->{params(c)/1e3:5.0f}K ({params(m)/params(c):4.1f}x)  "
          f"lossless={diff:.0e}  GPU throughput x{sp:.2f} @bs1024")
    torch.save({"state_dict":c.state_dict()}, ck.replace(".pth","_HARVESTED_compact.pth"))
