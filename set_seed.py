import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Set the random seed for reproducibility
    
    Args:
        seed: Integer seed for random number generators
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # PyTorch backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to use deterministic algorithms where possible
    # This may not work for all operations but helps with reproducibility
    try:
        torch.use_deterministic_algorithms(True)
    except:
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seed set to {seed} for reproducibility")
    
def enable_reproducible_inference():
    """
    Enable settings specifically for reproducible model inference
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except:
        torch.use_deterministic_algorithms(True, warn_only=True)
