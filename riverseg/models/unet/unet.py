import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """UNet architecture for image segmentation."""
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        # Encoder Layers
        self.enc1 = self._conv_block(self.in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)

        # Decoder Layers
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)

        # Output Layer
        self.output = nn.Conv2d(64, self.out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        # Upsample and concatenate with skip connection
        upconv4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((upconv4, enc4), dim=1))# Check the dimension of the tensor

        upconv3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((upconv3, enc3), dim=1))

        upconv2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((upconv2, enc2), dim=1))

        upconv1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((upconv1, enc1), dim=1))

        # Output Layer
        output = self.output(dec1) 

        return output


    @staticmethod
    def _conv_block(in_channels, out_channels):
        """Convolutional block for the UNet.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        Returns:
            nn.Sequential: Convolutional block of the UNet."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    


if __name__ == "__main__":
    in_channels = 3
    out_channels = 1
    model = UNet(in_channels, out_channels)

    x = torch.randn((1, in_channels, 256, 256)) # Batch size, channels, height, width
    prediction = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {prediction.shape}")

