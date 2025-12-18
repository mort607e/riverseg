import torch

def get_device(verbose:bool = True):
    """Returns the available device ('cuda' if available, else 'cpu')."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Device: {device}")
    return device