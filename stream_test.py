import torch
import torch_gcu

device = 'xla:0'

stream = torch_gcu.get_exe_default_stream(device=device)

print('******************Test result******************')
print(stream)