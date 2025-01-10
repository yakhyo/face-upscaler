import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, bias=True):
    """
    Creates a convolutional layer with adaptive padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        bias (bool, optional): Whether to include a bias term. Defaults to True.

    Returns:
        nn.Conv2d: Configured 2D convolutional layer.
    """
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3):
    """
    Creates an upsampling block using pixel shuffle for super-resolution tasks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int, optional): Upscaling factor. Defaults to 2.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.

    Returns:
        nn.Sequential: A sequential module with convolution and pixel shuffle.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    """
    A 3-layer convolution module with support for parameter updates and optional ReLU activation.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gain1 (int, optional): Multiplier for intermediate channels. Defaults to 1.
        gain2 (int, optional): Not used. Defaults to 0.
        s (int, optional): Stride for convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        relu (bool, optional): Whether to apply ReLU activation. Defaults to False.
    """

    def __init__(self, c_in: int, c_out: int, gain1=1, gain2=0, s=1, bias: bool = True, relu=False):
        super().__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )

        if not self.training:
            self.eval_conv.weight.requires_grad = False
            self.eval_conv.bias.requires_grad = False  # type: ignore
            self.update_params()

    def update_params(self):
        """
        Updates the parameters of the evaluation convolution layer by combining weights and biases
        from multiple convolution layers in the module.
        """
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()  # type: ignore
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat.contiguous()
        self.eval_conv.bias.data = self.bias_concat.contiguous()  # type: ignore

    def forward(self, x):
        """
        Forward pass for the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    """
    Spatial Attention Block (SPAB) to enhance features for super-resolution.

    Args:
        in_channels (int): Number of input channels.
        mid_channels (int, optional): Number of intermediate channels. Defaults to in_channels.
        out_channels (int, optional): Number of output channels. Defaults to in_channels.
        bias (bool, optional): Whether to include a bias term. Defaults to False.
    """

    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        Forward pass for the Spatial Attention Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor, intermediate tensor, and similarity attention.
        """
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


class SPAN(nn.Module):
    """
    Swift Parameter-free Attention Network (SPAN) for Efficient Super-Resolution.

    Args:
        num_in_ch (int): Number of input channels.
        num_out_ch (int): Number of output channels.
        feature_channels (int, optional): Number of feature channels. Defaults to 48.
        upscale (int, optional): Upscaling factor. Defaults to 4.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        norm (bool, optional): Whether to apply normalization. Defaults to True.
        img_range (float, optional): Range of image pixel values. Defaults to 255.0.
        rgb_mean (Tuple[float, float, float], optional): Mean RGB values for normalization. Defaults to (0.4488, 0.4371, 0.4040).
    """

    hyperparameters = {}

    def __init__(
        self,
        *,
        num_in_ch: int,
        num_out_ch: int,
        feature_channels=48,
        upscale=4,
        bias=True,
        norm=True,
        img_range=255.0,
        rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super().__init__()

        self.in_channels = num_in_ch
        self.out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.no_norm: torch.Tensor | None
        if not norm:
            self.register_buffer("no_norm", torch.zeros(1))
        else:
            self.no_norm = None

        self.conv_1 = Conv3XC(self.in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)

        self.upsampler = pixelshuffle_block(
            feature_channels, self.out_channels, upscale_factor=upscale
        )

    @property
    def is_norm(self):
        """
        Checks whether normalization is applied.

        Returns:
            bool: True if normalization is applied, False otherwise.
        """
        return self.no_norm is None

    def forward(self, x):
        """
        Forward pass for the Swift Parameter-free Attention Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.is_norm:
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, _att1 = self.block_1(out_feature)
        out_b2, _, _att2 = self.block_2(out_b1)
        out_b3, _, _att3 = self.block_3(out_b2)

        out_b4, _, _att4 = self.block_4(out_b3)
        out_b5, _, _att5 = self.block_5(out_b4)
        out_b6, out_b5_2, _att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output
