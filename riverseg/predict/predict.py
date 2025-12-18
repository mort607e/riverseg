import torch
from torch.nn import Module
from riverseg.models.unet import UNet


def loadmodel(weight_path: str, device: str = "cpu", in_channels: int = 10, out_channels: int = 1) -> Module:
    """
    Loads a UNet model with the specified parameters and weights.

    Returns:
    - model: torch.nn.Module, the loaded and initialized model.
    """ 
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    
    state_dict = torch.load(
                    weight_path,
                    map_location=torch.device(device),
                    weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    weight_path = "weights\\weights.pth"

    model = loadmodel(weight_path, "cpu", 9, 1)
    print("Model loaded successfully!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {next(model.parameters()).device}")