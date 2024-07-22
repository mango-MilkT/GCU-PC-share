import torch
from d2l import torch as d2l
import torch_gcu

def run(x):
    return [x.mm(x) for _ in range(50)]

x1 = torch.rand(size=(4000,4000),device='xla:0')
x2 = torch.rand(size=(4000,4000),device='xla:1')

run(x1)
run(x2)
# torch.cuda.synchronize('xla:0')
# torch.cuda.synchronize('xla:1')

with d2l.Benchmark('GCU0 time'):
    run(x1)
    # torch.cuda.synchronize('xla:0')

with d2l.Benchmark('GCU1 time'):
    run(x2)
    # torch.cuda.synchronize('xla:1')

with d2l.Benchmark('GCU0 & GCU1 time'):
    run(x1)
    run(x2)
    # torch.cuda.synchronize()