import torch
from torch import nn

class Hybrid_CNN(nn.Module):
    def __init__(self, seq_len=200, scales=4, channels=4):
        super(Hybrid_CNN, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.scales = scales

        # High-resolution branch (single scale)
        self.high_res = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.AdaptiveMaxPool1d(10)
        )

        # Multi-resolution branch (multi scale)
        self.multiscale_conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.multiscale_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.multiscale_conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.AdapMaxPool1d10 = nn.AdaptiveMaxPool1d(10)
        self.fc = nn.Sequential(
            nn.Linear(2 * 128 * 10, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        # High-res branch (first scale only)
        x_high = self.high_res(x[:, :, :, 0]).squeeze(-1)
        
        # Multi-scale branch
        x_multi = self.multiscale_conv1(x)
        x_multi = self.multiscale_conv2(x_multi).squeeze(-1)
        x_multi = self.multiscale_conv3(x_multi)
        x_multi = self.AdapMaxPool1d10(x_multi)
        
        # Concat features
        x = torch.cat([x_high, x_multi], dim=1)
        x = x.view(x.size(0), -1)
        
        # feature fusion
        x = self.fc(x)
        out = self.classifier(x)
        return out