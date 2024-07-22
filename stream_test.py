import torch
import torch_gcu

device = 'xla:0'

print('******************Test result******************')
print(torch_gcu.get_exe_default_stream(device=device))