import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise separable conv block for faster long-sequence processing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.depthwise_bn(self.depthwise(x)))
        x = self.relu(self.pointwise_bn(self.pointwise(x)))
        return x


class ResidualConvBlock(nn.Module):
    """Two-layer residual block with optional channel projection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_depthwise: bool = False,
    ):
        super().__init__()
        block_cls = DepthwiseSeparableConvBlock if use_depthwise else ConvBlock
        self.conv1 = block_cls(in_channels, out_channels, kernel_size=kernel_size)
        self.conv2 = block_cls(out_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return self.relu(x)


class EncoderBlock(nn.Module):
    """Encoder block with two conv blocks and optional downsampling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        use_residual: bool = False,
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        block_cls = DepthwiseSeparableConvBlock if use_depthwise else ConvBlock
        if use_residual:
            self.block = ResidualConvBlock(in_channels, out_channels, use_depthwise=use_depthwise)
        else:
            self.conv1 = block_cls(in_channels, out_channels)
            self.conv2 = block_cls(out_channels, out_channels)
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_residual:
            x = self.block(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        before_pool = x
        if self.downsample:
            x = self.pool(x)
        return x, before_pool


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, concatenation, and conv blocks."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = False,
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_residual = use_residual
        block_cls = DepthwiseSeparableConvBlock if use_depthwise else ConvBlock
        if use_residual:
            self.block = ResidualConvBlock(out_channels * 2, out_channels, use_depthwise=use_depthwise)
        else:
            self.conv1 = block_cls(out_channels * 2, out_channels)
            self.conv2 = block_cls(out_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        

        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([x, skip], dim=1)
        if self.use_residual:
            x = self.block(x)
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x


class AttentionBlock(nn.Module):
    """Channel attention mechanism."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM for temporal modeling."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, f, t = x.size()
        

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)
        

        x, _ = self.lstm(x)
        x = self.linear(x)
        

        x = x.view(b, t, c, f)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x


class MaskNet(nn.Module):
    """U-Net mask estimator for speech enhancement."""
    def __init__(
        self, 
        in_channels: int = 1, 
        base_channels: int = 32,
        use_lstm: bool = True,
        use_attention: bool = True,
        use_residual: bool = False,
        use_depthwise: bool = False,
        input_freq_bins: int = 257,
        output_channels: int = 1,
        output_activation: str = "sigmoid",
        output_scale: float = 1.0,
        mask_type: str = "magnitude",
        complex_mask_clip: float = 5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_depthwise = use_depthwise
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.mask_type = mask_type
        self.complex_mask_clip = complex_mask_clip

        block_cls = DepthwiseSeparableConvBlock if use_depthwise else ConvBlock
        

        self.enc1 = EncoderBlock(in_channels, base_channels, downsample=True, use_residual=use_residual, use_depthwise=use_depthwise)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, downsample=True, use_residual=use_residual, use_depthwise=use_depthwise)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, downsample=True, use_residual=use_residual, use_depthwise=use_depthwise)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, downsample=True, use_residual=use_residual, use_depthwise=use_depthwise)
        

        if use_residual:
            self.bottleneck = nn.Sequential(
                ResidualConvBlock(base_channels * 8, base_channels * 16, use_depthwise=use_depthwise),
                ResidualConvBlock(base_channels * 16, base_channels * 16, use_depthwise=use_depthwise)
            )
        else:
            self.bottleneck = nn.Sequential(
                block_cls(base_channels * 8, base_channels * 16),
                block_cls(base_channels * 16, base_channels * 16)
            )
        

        if self.use_lstm:
            # Four stride-2 downsamples.
            freq_after_downsample = input_freq_bins // 16
            lstm_input_size = (base_channels * 16) * freq_after_downsample
            self.lstm = BiLSTMBlock(lstm_input_size, base_channels * 8, num_layers=2)
        

        if self.use_attention:
            self.att4 = AttentionBlock(base_channels * 8)
            self.att3 = AttentionBlock(base_channels * 4)
            self.att2 = AttentionBlock(base_channels * 2)
            self.att1 = AttentionBlock(base_channels)
        

        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, use_residual=use_residual, use_depthwise=use_depthwise)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, use_residual=use_residual, use_depthwise=use_depthwise)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, use_residual=use_residual, use_depthwise=use_depthwise)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, use_residual=use_residual, use_depthwise=use_depthwise)
        

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, output_channels, kernel_size=1),
        )

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x) * self.output_scale
        if self.output_activation in {"identity", "linear", None}:
            return x
        raise ValueError(f"Unknown output activation: {self.output_activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a speech enhancement mask."""

        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        

        x = self.bottleneck(x4)
        

        if self.use_lstm:
            x = self.lstm(x)
        

        if self.use_attention:
            skip4 = self.att4(skip4)
            skip3 = self.att3(skip3)
            skip2 = self.att2(skip2)
            skip1 = self.att1(skip1)
        
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        

        mask = self.out_conv(x)
        return self._apply_output_activation(mask)



class SimpleMaskNet(nn.Module):
    """Simple MaskNet baseline."""
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 2)
        self.out_conv = nn.Conv2d(base_channels * 2, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.out_conv(x)
        return self.act(x)
