import torch
import torch_gcu

device = 'xla:0'
default_stream = torch.cuda.current_stream()
with torch.cuda.stream(default_stream):
    pass
stream = torch_gcu.get_exe_default_stream(device=device)

print('******************Test result******************')
print(stream)