# from huggingface_hub import snapshot_download

# local_dir = "/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B"
# repo_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# # 下载模型，参数与你的 CLI 命令基本一致
# snapshot_download(
#     repo_id=repo_id,
#     local_dir=local_dir,
#     resume_download=True,       # 对应 --resume-download
#     local_dir_use_symlinks=False, # 避免使用符号链接，直接下载文件
# )
# print(f"模型已下载到：{local_dir}")
print(f"CUDA版本: {torch.version.cuda}")