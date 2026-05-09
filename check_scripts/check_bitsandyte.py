import bitsandbytes as bnb
print(f"bitsandbytes 版本: {bnb.__version__}")
print(f"支持的 GPU: {bnb.cuda.is_available()}")