import torch
from collections import defaultdict

def collect_class_means(model, dataloader, num_classes, device):
    model.eval()
    layer_sums = defaultdict(lambda: torch.zeros(0, device=device))
    layer_counts = torch.zeros(num_classes, device=device)
    
    hooks = []
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    # Register hooks on layers of interest
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model(x)
            
            for name, act in activations.items():
                if layer_sums[name].numel() == 0:
                    layer_sums[name] = torch.zeros(
                        num_classes, *act.shape[1:], device=device
                    )
                
                for c in range(num_classes):
                    mask = (y == c)
                    if mask.any():
                        layer_sums[name][c] += act[mask].sum(dim=0)
            
            for c in range(num_classes):
                layer_counts[c] += (y == c).sum()

    for h in hooks:
        h.remove()

    layer_means = {}
    for name in layer_sums:
        layer_means[name] = layer_sums[name] / layer_counts.view(-1, *([1] * (layer_sums[name].dim() - 1)))

    return layer_means


#############################################
# Utility Functions
#############################################

