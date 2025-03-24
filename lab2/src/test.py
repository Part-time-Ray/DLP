import torch
print(torch.__version__)  # 應顯示 2.6.0+cu124
print(torch.cuda.is_available())  # 應顯示 True
print(torch.version.cuda)  # 應顯示 12.4
print(torch.cuda.current_device())  # 應顯示 0（你的 GPU 設備編號）
print(torch.cuda.get_device_name(0))  # 應顯示 "NVIDIA GeForce RTX 3060 Ti"