import torch

print(torch.backends.mps.is_available())
print(torch.backends.cpu.get_cpu_capability())