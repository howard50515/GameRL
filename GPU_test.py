import torch
print(torch.cuda.is_available())  # 應該返回 True
print(torch.cuda.device_count())  # 應該返回 GPU 數量
