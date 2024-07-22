import torch
import torch_gcu

torch_gcu.memory_allocated(device) # balance.profile.py
torch_gcu.get_rng_state(device) # checkpoint.py
torch_gcu.get_exe_default_stream
torch_gcu.synchronize()

torch.cuda.stream
torch.Stream
