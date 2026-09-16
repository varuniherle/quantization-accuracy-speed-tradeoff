import sys
print("Python:", sys.version)

for pkg in ["torch", "transformers", "numpy", "psutil"]:
    try:
        mod = __import__(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'unknown version')}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CPU threads torch will use:", torch.get_num_threads())
except Exception as e:
    print("torch check failed:", e)