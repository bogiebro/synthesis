from collections import defaultdict
from itertools import chain
from enum import Enum
import itertools
from re import I
import torch
import pyro.poutine as poutine
import pyro
import pyro.distributions as dist
import gpytorch as gp
from gpytorch.kernels import RBFKernel, PeriodicKernel, ScaleKernel, ProductKernel, \
    AdditiveKernel, LinearKernel, MaternKernel, RQKernel
from typing import Optional, Callable
from gpytorch.priors import GammaPrior, NormalPrior
from queue import Queue

# Compare training losses for REINFORCE and gflownet.
# Show top k choices.
# Try a neural parameterization
# Compare with a forward prior. 
# do we pick paths in proportion to reward? what exactly is the model
# we're capturing?

# Add a prior to the kernel params.

# Should compare to greedy 
# Use Pyrsistent, and replace hashes with entities

# Validation loss could also use the top 1. 
# Or average the loss over the top percentile. 

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
        len(p.builder) == 0 and len(p.adds) > 0,
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

def base_kernel(t):
    if t is RBFKernel:
        return RBFKernel(lengthscale_prior=GammaPrior(1, 1))
    if t is MaternKernel:
        return RBFKernel(lengthscale_prior=GammaPrior(1, 1))
    if t is LinearKernel:
        return LinearKernel(
            offset_prior=NormalPrior(0, 5),
            variance_prior=GammaPrior(1,1))        
    if t is PeriodicKernel:
        return PeriodicKernel(
            lengthscale_prior=GammaPrior(1,1),
            period_length_prior=GammaPrior(1,1))
    if t is RQKernel:
        return RQKernel(
            lengthscale_prior=GammaPrior(1,1),
            alpha_prior=GammaPrior(1,1))

def to_kernel(p: Prog) -> gp.kernels.Kernel:
    if len(p.builder) > 0:
        p.adds.add(frozenset(p.builder))
    assert len(p.adds) > 0
    for a in p.adds:
        assert len(a) > 0
    return AdditiveKernel(*[ScaleKernel(
        ProductKernel(*[base_kernel(a) for a in mul]),
        outputscale_prior=GammaPrior(1,1)) for mul in p.adds])

def init_params(p: Prog) -> Callable[[], torch.Tensor]:
    def inner():
        vals = torch.zeros(len(Action)) + (-1/ valid_actions(p)) + 1
        vals[0:2] += 2
        if (vals == -torch.inf).all():
            raise DoneException()
        return vals
    return inner

def next_prog(logp: torch.Tensor, prog: Prog):
    old_prog = hash_prog(prog)
    logits = pyro.param("param " + str(old_prog), init_params(prog))
    ix = pyro.sample(old_prog, Categorical(logits=logits))
    action = Action(int(ix.item()) + 1)
    step(prog, action)
    logp -= torch.log(torch.tensor(num_parents(prog)))
        
def path_guide(kerns):
    prog = Prog()
    logp = torch.tensor(0.0)
    try:
        while not prog.done:
            next_prog(logp, prog)
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
    covar = covar.add_diag(torch.tensor(0.001))
    d = gp.distributions.MultivariateNormal(mean(x), covar)
    return pyro.sample("y", d, obs=y)

def num_parents(prog):
    if prog.done or len(prog.builder) == 0:
        return 1
    else:
        return len(prog.builder)

class DoneException(Exception):
    pass

def prior_logprob(kern):
    res = torch.tensor(0.)
    for name, module, prior, closure, _ in kern.named_priors():
        res.add_(prior.log_prob(closure(module)).sum())
    return res

class Categorical(dist.Categorical):
    def enumerate_support(self, expand=True):
        result = super().enumerate_support(expand)
        probs = self.log_prob(result)
        mask = probs > -torch.inf
        return result[mask]

# Problem: this isn't right. 
# We don't just want to take the top k at every site. 
# We want the top k overall. 
# Which means we want something like the queue messenger,
# But we actually want a Priority queue.
# we should also return the log probs. 

class TopKMessenger(poutine.messenger.Messenger):
    def __init__(self, k):
        self.k = k

    def _pyro_sample(self, msg):
        if type(msg['fn']) == Categorical:
            n = msg['fn'].logits.shape[0]
            _, indices = msg['fn'].logits.topk(n - self.k, largest=False)
            msg['fn'].logits[indices] = -torch.inf

# Ideally, we just want to run the model, using a handler
# that picks the top k. There's queues in pyro for enumeration too. 
def topk(fn, k):
    q = Queue()
    q.put(poutine.Trace())
    enum_model = poutine.queue(TopKMessenger(k)(fn), queue=q)
    samples = []
    while not q.empty():
        trace = poutine.trace(enum_model).get_trace()
        samples.append(trace.nodes['a']["value"])
    return samples

# TODO: finish this.
def greedy_search(p):
    v = valid_actions(p)
    actions = [Action(i+1) for (i,b) in enumerate(v) if b]
    for a in actions:
        step(p, a)
    # Optimize all the valid actions (not including Done or Add)
    # Pick the best scoring one. 
    # Then, either Add or do another multiply
    # Repeat with the best of these. 
    # The greedy search doesn't look at Done. 


# for _ in range(10):
#     prog, logp, kern, newkern = a.path_guide(KERNS)
#     print(prog)