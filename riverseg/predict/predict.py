import torch
import numpy as np
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

def predict_river_extent(preprocessed_bands:np.ndarray, model: Module) -> np.ndarray:
    """
    Predicts the river extent from the preprocessed bands using the provided model.

    Returns:
    - numpy array with predicted river extent.
    """ 
    input_tensor = torch.from_numpy(preprocessed_bands).unsqueeze(0).float()
    with torch.no_grad():
        output = model(input_tensor)
    predicted_mask = output.squeeze(0).cpu().numpy()
    return predicted_mask

if __name__ == "__main__":
    # Initializing model
    weight_path = "weights\\weights.pth"
    model = loadmodel(weight_path, "cpu", 9, 1)
    print("Model loaded successfully!")
    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {next(model.parameters()).device}")

    # Prediction Example
    preprocessed_bands = np.random.random((9,256,256)) # <- Relpace with the actual preprocessed bands.
    predicted_mask = predict_river_extent(preprocessed_bands, model)
    print("")
    print("Prediction Example:")
    print(f"Input bands shape: {preprocessed_bands.shape}")
    print(f"Prediction shape: {predicted_mask.shape}")