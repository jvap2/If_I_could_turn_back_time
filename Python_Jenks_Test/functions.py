import time
import torch

from backpack import backpack, extend
from backpack.extensions import HMP, DiagHessian
from backpack.hessianfree.hvp import hessian_vector_product
from backpack.utils.examples import load_one_batch_mnist

BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)


def rademacher(shape, dtype=torch.float32, device=DEVICE):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)


def hutchinson_trace_hmp(V, V_batch=1):
    """Hessian trace estimate using BackPACK's HMP extension.

    Perform `V_batch` Hessian multiplications at a time.
    """
    V_count = 0
    trace = 0

    while V_count < V:
        V_missing = V - V_count
        V_next = min(V_batch, V_missing)

        for param in model.parameters():
            v = rademacher((V_next, *param.shape))
            Hv = param.hmp(v).detach()
            vHv = torch.einsum("i,i->", v.flatten(), Hv.flatten().detach())
            trace += vHv / V

        V_count += V_next

    return trace