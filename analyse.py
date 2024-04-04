import torch
import torch.nn as nn

# Create a random tensor to represent your input
input = torch.rand([1, 1536, 24, 42])

# Define the layer
layer = nn.ModuleList([
    # nn.ConvTranspose2d(in_channels=1536, out_channels=1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.Conv2d(1536,256 , kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2), # x2
            
            nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2), # x2
            
            # nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2), # x2
            
            # nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3, bias=False),
            
            nn.Conv2d(64, 1536, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
            # nn.ConvTranspose2d(64, 1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(64, 1536, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingNearest2d(scale_factor=4),
            ])

# Pass the input through the layer
output = input
for fc in layer:
    output = fc(output)

# Print the output size
print(output.shape)
# data = np.load('/home/coisini/data/nuscenes_fcclip_features/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.npy')
# print(data)
# print(data.shape)
