import torch
import torch.nn as nn

CLASSES = 2
CHANNEL_NUM = 12
SAMPLE_POINTS = 1024

TEMPORAL_FILTER = 8  # number of temporal filters
DEPTHWISE_FILTER = 2  # number of spatial filters
DROP_OUT_RATE = 0.5  # dropout rate
SAMPLE_RATE = 128  # EEG sample rate in Hz


class EEGWaveNet(nn.Module):
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
        # self.pipeline_5 = nn.Sequential(
        #     nn.Conv1d(in_channels=chn, out_channels=32, kernel_size=4, groups=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(0.01),
        #     nn.Conv1d(32, 32, kernel_size=4, groups=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(0.01),
        #     nn.Conv1d(32, 32, kernel_size=4, groups=1),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(0.01),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            # nn.Linear(160, 64),
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
        # temp_w5 = self.temp_conv_6(temp_w4)

        w1 = self.pipeline_1(temp_w1).mean(dim=-1)
        w2 = self.pipeline_2(temp_w2).mean(dim=-1)
        w3 = self.pipeline_3(temp_w3).mean(dim=-1)
        w4 = self.pipeline_4(temp_w4).mean(dim=-1)
        # w5 = self.pipeline_5(temp_w5).mean(dim=-1)

        concat_vector = torch.cat((w1, w2, w3, w4), dim=1)
        classes = nn.functional.log_softmax(self.classifier(concat_vector), dim=1)
        return classes


class EEGNet(nn.Module):
    """EEGNet: compact CNN architecture for EEG-based BCIs from Lawhern et al. (2018)
    https://arxiv.org/abs/1611.08024

    EEGNet is a compact convolutional neural network for EEG-based
    brain-computer interfaces. It processes raw multichannel EEG signals
    through the following stages:

    1. Input scaling: the raw EEG input (channels x samples) is scaled.
    2. Temporal convolution: a standard convolution over the time axis
       learns frequency-specific temporal filters.
    3. Depthwise convolution: a per-channel (depthwise) spatial
       convolution combines the EEG channels to learn frequency-specific
       spatial patterns.
    4. Separable convolution: a depthwise convolution followed by a 1x1
       (pointwise) convolution summarizes the temporal dimension into a
       compact feature map.
    5. Classification: feature maps are pooled, flattened, and passed
       through a fully connected layer followed by softmax to output the
       class prediction.
    """

    def __init__(
        self,
        f1=TEMPORAL_FILTER,
        d=DEPTHWISE_FILTER,
        chn=CHANNEL_NUM,
        classes=CLASSES,
        p=DROP_OUT_RATE,
        fs=SAMPLE_RATE,
    ):
        super().__init__()

        kernLength = int(fs * 0.5)  # 64
        sep_kernLength = int(fs * 0.125)  # 16

        # Block 1 ===================
        # (\, 1, (C, T)) -> (\, F1, (C, T))
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=f1,
            kernel_size=(1, kernLength),
            padding="same",
            bias=False,
        )
        # (\, F1, (C, T)) -> (\, F1 * D, (1, T))
        self.depthwise_conv = nn.Conv2d(
            in_channels=f1,
            out_channels=f1 * d,
            kernel_size=(chn, 1),
            groups=f1,
            padding="valid",
            bias=False,
        )
        self.batchnorm1 = nn.BatchNorm2d(f1)
        self.batchnorm2 = nn.BatchNorm2d(f1 * d)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(p)

        # Block 2 ===================
        self.sep_depthwise_conv = nn.Conv2d(
            in_channels=f1 * d,
            out_channels=f1 * d,
            kernel_size=(1, sep_kernLength),
            groups=f1 * d,
            padding="same",
            bias=False,
        )
        self.sep_pointwise_conv = nn.Conv2d(
            in_channels=f1 * d, out_channels=f1 * d, kernel_size=(1, 1), bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(f1 * d)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8))

        self.classifier = nn.LazyLinear(classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (B, 1, (C, T))

        x = self.temporal_conv(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)

        x = self.avgpool1(x)
        x = self.dropout(x)

        x = self.sep_depthwise_conv(x)
        x = self.sep_pointwise_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)

        x = self.avgpool2(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=1)  # Flatten: (B, F1 * D * T)
        x = self.classifier(x)  # Fully connected layer: (B, classes)

        return x
