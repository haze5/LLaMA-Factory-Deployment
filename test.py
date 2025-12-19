import torch
# 1. 查看PyTorch版本和它认为的CUDA版本 (应为12.1)
print(f"PyTorch版本: {torch.__version__}")
print(f"PyTorch编译时的CUDA版本: {torch.version.cuda}")

# 2. 这是最重要的：检查GPU是否可用
print(f"GPU是否可用 (True为成功): {torch.cuda.is_available()}")

# 3. 打印你的GPU信息，确认是24GB的A10
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    