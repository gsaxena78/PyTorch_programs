import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.mps.is_available()) # Check for Apple Silicon acceleration

