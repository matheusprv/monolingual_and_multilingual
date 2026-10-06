import gc
import torch

def clean_memory():
    """Force RAM and VRAM to clean"""
    gc.collect()                # Clean not used python objects
    torch.cuda.empty_cache()    # Clean Pytorch alocation cache
    print("Memory cleaned 🧹!")