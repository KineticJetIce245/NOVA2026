import torch
import torch.nn as nn

CLASSES = 2
CHANNEL_NUM = 12
SAMPLE_POINTS = 1024


class EEGWaveNetCNN(nn.Module):
    """
    EEGWaveNetCNN: model architecture from https://ieeexplore.ieee.org/document/9645336

    EEGWaveNetCNN is a multiscale CNN-based model for EEG seizure detection.
    It extracts both spatial and temporal features from raw multichannel EEG
    signals through the following stages:

    1. Input scaling: the raw EEG input (12 channels x samples) is scaled.
    2. Temporal feature extraction: parallel 1D convolutional branches with
       different kernel sizes capture multiscale temporal patterns from each
       channel.
    3. Spatial feature extraction: the per-channel temporal features are
       concatenated and fused by a 1x1 (pointwise) convolution to capture
       cross-channel interactions.
    4. Deep feature extraction: repeated 1D convolutions with residual (skip)
       connections further refine high-level features.
    5. Classification: features are flattened and passed through a fully
       connected layer followed by a softmax (or sigmoid) to output the
       seizure/non-seizure prediction.
    """

    def __init__(self, chn=CHANNEL_NUM, spps=SAMPLE_POINTS, classes=CLASSES):
        super().__init__()
        # Temporal convolution
        self.temp_conv_1 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)
        self.temp_conv_2 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)
        self.temp_conv_3 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)
        self.temp_conv_4 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)
        self.temp_conv_5 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)
        self.temp_conv_6 = nn.Conv1d(chn, chn, kernel_size=2, stride=2, groups=chn)

        # Total of five individual piplines:
        self.pipeline_1 = nn.Sequential(
            # (chn, (1, spps))
            # (12, (1, spps)) -> (32, (1, spps-3))
            nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
            # Normalization the resulting feature maps
            nn.BatchNorm1d(32),
            # Activation function
            nn.LeakyReLU(0.01),
            # (32, (1, spps-3)) -> (32, (1, spps-6))
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            # (32, (1, spps-6)) -> (32, (1, spps-9))
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )
        self.pipeline_2 = nn.Sequential(
            nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )
        self.pipeline_3 = nn.Sequential(
            nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )
        self.pipeline_4 = nn.Sequential(
            nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )
        self.pipeline_5 = nn.Sequential(
            nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
        )
        self.classfier = nn.Sequential(
            nn.Linear(160, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, classes),
        )

    def forward(self, x):
        temp_x = self.temp_conv_1(x)
        temp_w1 = self.temp_conv_2(temp_x)
        temp_w2 = self.temp_conv_3(temp_w1)
        temp_w3 = self.temp_conv_4(temp_w2)
        temp_w4 = self.temp_conv_5(temp_w3)
        temp_w5 = self.temp_conv_6(temp_w4)

        w1 = self.pipeline_1(temp_w1)
        w2 = self.pipeline_2(temp_w2)
        w3 = self.pipeline_3(temp_w3)
        w4 = self.pipeline_4(temp_w4)
        w5 = self.pipeline_5(temp_w5)

        concat_vector = torch.cat((w1, w2, w3, w4, w5), dim=1)  # pyright: ignore
        classes = nn.functional.log_softmax(self.classifier(concat_vector), dim=1)  # pyright: ignore
        return classes
