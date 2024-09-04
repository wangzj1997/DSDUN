import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.transforms import fft2, ifft2, complex_abs, fftshift, ifftshift,fft2c, ifft2c


class IUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 40,
        num_pool_layers: int = 3,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.bottom_FFC_conv = FFCResnetBlock(ch * 2, 'reflect')
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_FFC_conv = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_FFC_conv.append(FFCResnetBlock(ch, 'reflect'))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
        self.up_FFC_conv.append(FFCResnetBlock(ch, 'reflect'))
        self.lastconv = nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1)


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        output = self.bottom_FFC_conv(output)
        # apply up-sampling layers
        for transpose_conv, conv, ffc in zip(self.up_transpose_conv, self.up_conv, self.up_FFC_conv):
        # for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
            output = ffc(output)
        output = self.lastconv(output)
        return output
    

# class FFCSE_block(nn.Module):
    
#     def __init__(self, channels, ratio_g):
#         super(FFCSE_block, self).__init__()
#         in_cg = int(channels * ratio_g)
#         in_cl = channels - in_cg
#         r = 16

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv1 = nn.Conv2d(channels, channels // r,
#                                kernel_size=1, bias=True)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
#             channels // r, in_cl, kernel_size=1, bias=True)
#         self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
#             channels // r, in_cg, kernel_size=1, bias=True)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = x if type(x) is tuple else (x, 0)
#         id_l, id_g = x

#         x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
#         x = self.avgpool(x)
#         x = self.relu1(self.conv1(x))

#         x_l = 0 if self.conv_a2l is None else id_l * \
#             self.sigmoid(self.conv_a2l(x))
#         x_g = 0 if self.conv_a2g is None else id_g * \
#             self.sigmoid(self.conv_a2g(x))
#         return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    
    def forward(self, x):
        batch = x.shape[0]


        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        x = self.complex_to_chan_dim(fft2c(self.chan_complex_to_last_dim(x)))
        x = self.conv_layer(x)  # (batch, c*2, h, w/2+1)
        x = self.relu(x)       
        x = self.complex_to_chan_dim(ifft2c(self.chan_complex_to_last_dim(x)))



        # fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        # ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        # ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        # ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        # ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        # ffted = self.relu(ffted)

        # ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
        #     0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        # ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        # ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        # output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm

        return x


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin=0.5, ratio_gout=0.5, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # in_cg = int(in_channels * ratio_gin)
        # in_cl = in_channels - in_cg
        # out_cg = int(out_channels * ratio_gout)
        # out_cl = out_channels - out_cg



        in_cp = int(in_channels * 0.2)
        in_cl = int(in_channels * 0.4)
        in_cg = in_channels - in_cl - in_cp
        out_cp = int(out_channels * 0.2)
        out_cl = int(out_channels * 0.4)
        out_cg = out_channels - out_cl - out_cp


        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.pixel_in_num = in_cp
        self.local_in_num = in_cl


        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convp2p = module(in_cp, out_cp, 1,
                              stride, 0, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convp2l = module(in_cp, out_cl, 1,
                              stride, 0, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convp2g = module(in_cp, out_cg, 1,
                              stride, 0, dilation, groups, bias, padding_mode=padding_type)


        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2p = module(in_cl, out_cp, 3,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, 3,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, 3,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        

        module = nn.Identity if in_cg == 0 or out_cl == 0 else SpectralTransform
        self.convg2p = module(
            in_cg, out_cp, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2l = module(
            in_cg, out_cl, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)       


        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_p, x_l, x_g = x
        out_xp, out_xl, out_xg = 0, 0, 0

        # if self.gated:
        #     total_input_parts = [x_p, x_l, x_g]

        #     total_input = torch.cat(total_input_parts, dim=1)

        #     gates = torch.sigmoid(self.gate(total_input))
        #     g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        # else:
        #     g2l_gate, l2g_gate = 1, 1
        # print(self.convp2p(x_p).shape)
        # print(self.convl2p(x_l).shape)
        # print(self.convg2p(x_g).shape)
        if self.ratio_gout != 1:
            out_xp = self.convp2p(x_p) + self.convl2p(x_l) + self.convg2p(x_g)
        if self.ratio_gout != 1:
            out_xl = self.convp2l(x_p) + self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convp2g(x_p) + self.convl2g(x_l) + self.convg2g(x_g)

        return out_xp, out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=0.5, ratio_gout=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                activation_layer=nn.ReLU,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)

        global_channels = int(out_channels * ratio_gout)

        pact = nn.Identity if ratio_gout == 1 else activation_layer
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        
        self.act_p = pact(inplace=True)        
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_p, x_l, x_g = self.ffc(x)
        x_p = self.act_p(x_p)       
        x_l = self.act_l(x_l)
        x_g = self.act_g(x_g)
        return x_p, x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, activation_layer=nn.ReLU, dilation=1,
                  inline=True, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,                          
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,                                
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
      
      
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_p, x_l, x_g = x[:, :self.conv1.ffc.pixel_in_num], x[:, self.conv1.ffc.pixel_in_num:self.conv1.ffc.pixel_in_num + self.conv1.ffc.local_in_num], x[:, self.conv1.ffc.pixel_in_num + self.conv1.ffc.local_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_p, id_l, id_g = x_p, x_l, x_g

        x_p, x_l, x_g = self.conv1((x_p, x_l, x_g))
        x_p, x_l, x_g = self.conv2((x_p, x_l, x_g))

        x_p, x_l, x_g = id_p + x_p, id_l + x_l, id_g + x_g
        out = x_p, x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

    
class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)