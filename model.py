import torch
from torch import nn

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.list = [256, 300, 512]
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(150, self.list[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.list[2], affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.list[2], self.list[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[2], affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.list[2], self.list[2], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.list[2], affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(self.list[2], self.list[1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.list[1], affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(self.list[1], self.list[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.list[0], affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(self.list[0], 150, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(150, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(150, 150, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(150, affine=True),
                nn.ReLU(inplace=True),
        )
        initialize_weights(self)

    def forward(self, x):
        out1 = self.encoder_conv1(x)
        out2 = self.encoder_conv2(out1)
        out3 = self.encoder_conv3(out2)
        encoder = self.encoder_conv4(out3 + out1)
        decoder = self.decoder(encoder)

        return encoder, decoder

class SCAE_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.list = [256, 300, 512]
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(150, self.list[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.list[0], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.list[0], self.list[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[1], affine=True),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv4 = nn.Sequential(
            nn.Conv2d(self.list[1], self.list[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.list[2], affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.list[2], self.list[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(self.list[2], affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.list[2] * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        initialize_weights(self)

    def forward(self, x):
        batch_size = x.size(0)
        out1 = self.encoder_conv1(x)
        out2 = self.encoder_conv2(out1)
        out3 = self.encoder_conv3(out2)
        out4 = self.encoder_conv4(out3 + out1)
        out = out4.view(batch_size, -1)
        out = self.fc(out)

        return out







