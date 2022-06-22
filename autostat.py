from collections import defaultdict, namedtuple
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
from pyrsistent import pset
import functools

# Show top k choices.
# Try a neural parameterization
# Compare with a forward prior. 
# Add a prior to the kernel params.
# Validation loss could also use the top 1. 
# Or average the loss over the top percentile. 

MEAN = gp.means.ZeroMean()

Prog = namedtuple('Prog', 'builder adds done')

def new_prog():
    return Prog(pset([]), pset([]), False)

Action = Enum('Action', 'Done Add Periodic RBF Linear') # Matern RQ')

def notseen(ty, p) -> bool:
    return ty not in p.builder and \
        p.builder.add(ty) not in p.adds

def valid_actions(p: Prog) -> torch.Tensor:
    return torch.tensor([
        len(p.builder) == 0 and len(p.adds) > 0,
        len(p.builder) > 0,
        notseen(Action.Periodic, p),
        notseen(Action.RBF, p),
        notseen(Action.Linear, p)])
        # notseen(Action.Matern, p),
        # notseen(Action.RQ, p)])

def step(p: Prog, a: Action):
    if a == Action.Done:
        return p._replace(done = True)
    elif a == Action.Add:
        return Prog(pset([]), p.adds.add(p.builder), False)
    else:
        return p._replace(builder=p.builder.add(a))

def base_kernel(t):
    if t is Action.Periodic:
        return PeriodicKernel(
            lengthscale_prior=GammaPrior(1,1),
            period_length_prior=GammaPrior(1,1))
    if t is Action.RBF:
        return RBFKernel(lengthscale_prior=GammaPrior(1, 1))
    if t is Action.Linear:
        return LinearKernel(
            offset_prior=NormalPrior(0, 5),
            variance_prior=GammaPrior(1,1))  
    if t is Action.Matern:
        return MaternKernel(lengthscale_prior=GammaPrior(1, 1))
    if t is Action.RQ:
        return RQKernel(
            lengthscale_prior=GammaPrior(1,1),
            alpha_prior=GammaPrior(1,1))

def to_kernel(p: Prog) -> gp.kernels.Kernel:
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
    logits = pyro.param(str(hash(prog)), init_params(prog))
    ix = pyro.sample(prog, Categorical(logits=logits))
    action = Action(int(ix.item()) + 1)
    prog2 = step(prog, action)
    logp -= torch.log(torch.tensor(num_parents(prog2)))
    return prog2
        
def path_guide(kerns):
    prog = new_prog()
    logp = torch.tensor(0.0)
    try:
        while not prog.done:
            prog = next_prog(logp, prog)
    except DoneException:
        pass
    newkern = False
    if prog not in kerns:
        kerns[prog] = to_kernel(prog)
        newkern = True
    return prog, logp, kerns[prog], newkern

def gp_model(prog, kern, x, y):
    pyro.module(f"kernel{hash(prog)}", kern)
    covar = kern(x)
    covar = covar.add_diag(torch.tensor(0.001))
    d = gp.distributions.MultivariateNormal(MEAN(x), covar)
    return pyro.sample("y", d, obs=y)

def num_parents(prog):
    if prog.done or len(prog.builder) == 0:
        return 1
    else:
        return len(prog.builder)

class DoneException(Exception):
    pass

class RejectException(poutine.NonlocalExit):
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

class RejectionMessenger(poutine.messenger.Messenger):
    def __init__(self, escape_fn):
        super().__init__()
        self.escape_fn = escape_fn
    def _pyro_sample(self, msg):
        if self.escape_fn(msg):
            msg['done'] = True
            msg['stop'] = True
            def cont(m):
                raise RejectException(m)
            msg['continuation'] = cont

class QuantileMessenger(poutine.messenger.Messenger):
    def __init__(self, thresh):
        self.thresh = thresh

    def _pyro_sample(self, msg):
        if type(msg['fn']) == Categorical:
            sorted_probs = msg['fn'].probs.sort(descending=True)
            quantiles = torch.cumsum(sorted_probs.values, 0)
            ix = (quantiles < self.thresh).sum()
            msg['fn'].logits[sorted_probs.indices[ix+1:]] = -torch.inf

class MAPMessenger(poutine.messenger.Messenger):
    def _pyro_sample(self, msg):
        if type(msg['fn']) == Categorical:
            ix = torch.argmax(msg['fn'].probs)
            msg['fn'].logits[ix+1:] = -torch.inf
            msg['fn'].logits[:ix] = -torch.inf

def deep_escape(depth, msg):
    return (type(msg['name']) is Prog and
        len(msg['name'].adds) >= depth and
        len(msg['name'].builder) > 0)

# Could we combine greedy_search with all_in_quantile?
# So that we don't just keep 1 from the previous level, but
# as many as are in the appropriate quantile?
# It may take inordinately long, but worth trying

def all_in_quantile(fn, k):
    q = Queue()
    q.put(poutine.Trace())
    enum_model = poutine.queue(QuantileMessenger(k)(fn), queue=q)
    samples = []
    while not q.empty():
        samples.append(enum_model())
    return samples