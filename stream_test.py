import torch
import torch_gcu

device = 'xla:0'

print(torch_gcu.get_exe_default_stream(device=device))