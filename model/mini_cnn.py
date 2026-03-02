import torch.nn as nn


padding = 1
kernel_size = 3
img_size = 224
num_class = 50
                                                                                                                        

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → ReLU → Conv → ReLU
    Used for classification
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,padding=padding)
        #self.batchNorm2 = nn.BatchNorm2d(out_channels)
        #self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        #x = self.conv2(x)
        #x = self.batchNorm2(x)
        #x = self.relu2(x)
        return x


  
class MiniCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # level 1: 3*224*224
        self.conv1 = ConvBlock(3,16)
        self.pool1 = nn.MaxPool2d(2) # → 16*112×112

        # level 2: 16*112*112
        self.conv2 = ConvBlock(16,32)
        self.pool2 = nn.MaxPool2d(2) # → 32*56×56

        # level 3: 32*56*56
        self.conv3 = ConvBlock(32,64)
        self.pool3 = nn.MaxPool2d(2) # → 64*28×28

        # level 4: 64*28*28
        self.conv4 = ConvBlock(64,128)
        self.pool4 = nn.MaxPool2d(2) # → 128*14×14

        # level 5 : 128*14×14
        self.conv5 = ConvBlock(128,256)
        self.pool5 = nn.MaxPool2d(2) # → 256*7×7

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        x = x.view(x.size(0), 3, img_size, img_size)

        x1 = self.conv1(x)
        x = self.pool1(x1)

        x2 = self.conv2(x)
        x = self.pool2(x2)

        x3 = self.conv3(x)
        x = self.pool3(x3)

        x4 = self.conv4(x)
        x = self.pool4(x4)

        x5 = self.conv5(x)
        x = self.pool5(x5)

        return self.head(x)