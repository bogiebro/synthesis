from collections import defaultdict
from itertools import chain
from enum import Enum
import itertools
import torch
import pyro
import pyro.distributions as dist
import gpytorch as gp
from gpytorch.kernels import RBFKernel, PeriodicKernel, ScaleKernel, ProductKernel, \
    AdditiveKernel, LinearKernel, MaternKernel
from typing import Optional, Callable

# TODO: quantify degree of sharing
# Compare training losses for REINFORCE and gflownet.
# Show top k choices.
# Try a neural parameterization
# Compare with a forward prior. 
# do we pick paths in proportion to reward? what exactly is the model
# we're capturing?

def kerns(a):
    return [b.__name__ for b in a]

class Prog:
    def __init__(self):
        self.builder = set()
        self.adds = set()
        self.done = False
    def __str__(self):
        return f"Prog({self.done} {kerns(self.builder)} {[kerns(a) for a in self.adds]})"

def hash_prog(p: Prog) -> int:
    return hash((p.done, frozenset(p.adds), frozenset(p.builder)))

Action = Enum('Action', 'Done Add Periodic RBF Linear Matern')

def notseen(ty, p) -> bool:
    return ty not in p.builder and \
        frozenset(chain([ty], p.builder)) not in p.adds

def valid_actions(p: Prog) -> torch.Tensor:
    return torch.tensor([
        len(p.builder) > 0 and frozenset(p.builder) not in p.adds,
        len(p.builder) > 0,
        notseen(PeriodicKernel, p),
        notseen(RBFKernel, p),
        notseen(LinearKernel, p),
        notseen(MaternKernel, p)])

def step(p: Prog, a: Action):
    if a == Action.Done:
        p.done = True
    elif a == Action.Add:
        p.adds.add(frozenset(p.builder))
        p.builder = set()
    elif a == Action.Periodic:
        p.builder.add(PeriodicKernel)
    elif a == Action.RBF:
        p.builder.add(RBFKernel)
    elif a == Action.Linear:
        p.builder.add(LinearKernel)
    elif a == Action.Matern:
        p.builder.add(MaternKernel)
    else:
        assert False

def to_kernel(p: Prog) -> gp.kernels.Kernel:
    if len(p.builder) > 0:
        p.adds.add(frozenset(p.builder))
    assert len(p.adds) > 0
    for a in p.adds:
        assert len(a) > 0
    return AdditiveKernel(*[ScaleKernel(
        ProductKernel(*[a() for a in mul])) for mul in p.adds]) 

def init_params(p: Prog) -> Callable[[], torch.Tensor]:
    def inner():
        vals = torch.zeros(len(Action)) + (-1/ valid_actions(p)) + 1
        vals[0] += 1
        if (vals == -torch.inf).all():
            raise DoneException()
        return vals
    return inner

def next_prog(capture, logp: torch.Tensor, prog: Prog):
    old_prog = hash_prog(prog)
    logits = pyro.param("param " + str(old_prog), init_params(prog))
    ix = pyro.sample(old_prog, dist.Categorical(logits=logits))
    action = Action(int(ix.item()) + 1)
    step(prog, action)
    # print(prog)
    new_prog = hash_prog(prog)
    if new_prog in capture.trace.nodes:
        raise LoopException()
    logp -= torch.log(torch.tensor(num_parents(prog)))
        
def path_guide(capture, kerns):
    prog = Prog()
    logp = torch.tensor(0.0)
    try:
        while not prog.done:
            next_prog(capture, logp, prog)
    except DoneException:
        pass
    newkern = False
    prog_hash = hash_prog(prog)
    if prog_hash not in kerns:
        kerns[prog_hash] = to_kernel(prog)
        newkern = True
    return prog, logp, kerns[prog_hash], newkern

def gp_model(prog, mean, kern, x, y):
    pyro.module(f"kernel {hash_prog(prog)}", kern)
    pyro.module(f"mu", mean)
    covar = kern(x)
    covar = covar.add_diag(torch.tensor(0.01))
    d = gp.distributions.MultivariateNormal(mean(x), covar)
    return pyro.sample("y", d, obs=y)

def num_parents(prog):
    if prog.done or len(prog.builder) == 0:
        return 1
    else:
        return len(prog.builder)

class LoopException(Exception):
    pass

class DoneException(Exception):
    pass