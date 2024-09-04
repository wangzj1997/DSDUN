import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import vit_config as configs
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from data.transforms import fft2c, ifft2c, complex_abs, fftshift, ifftshift
import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def complex_to_chan_dim(x):
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

def chan_complex_to_last_dim(x):
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

class KUnet(nn.Module):
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
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        num_blocks=4,
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
        config_vit = CONFIGS['ViT-B_16']
        # self.transformer =  VisionTransformer(config_vit, img_size=32)
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=128) for i in range(num_blocks)])
        
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

         
        self.conv = ConvBlock(ch, ch * 2, drop_prob)
            

        # self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )


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
            output = self.complex_to_chan_dim(ifft2c(self.chan_complex_to_last_dim(output)))
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = self.complex_to_chan_dim(fft2c(self.chan_complex_to_last_dim(output)))

        output = self.transformer(output)
        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = self.complex_to_chan_dim(ifft2c(self.chan_complex_to_last_dim(output)))
            output = transpose_conv(output)
            output = self.complex_to_chan_dim(fft2c(self.chan_complex_to_last_dim(output)))

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

        return output

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w).contiguous()

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias=False):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim)
        self.attn = FSAS(dim, bias)

        self.norm2 = LayerNorm(dim)
        self.ffn = Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = Linear(128, 512)
        self.fc2 = Linear(512, 128)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = to_3d(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = to_4d(x, h, w)
        return x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2)

        self.patch_size = 32

    def forward(self, x):
        hidden = self.to_hidden(x)
    

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q_patch = q
        k_patch = k        
        
        q_fft = ifft2c(chan_complex_to_last_dim(q_patch))
        k_fft = ifft2c(chan_complex_to_last_dim(k_patch))
        
        q_fft = q_fft[..., 0] + 1j * q_fft[..., 1]
        k_fft = k_fft[..., 0] + 1j * k_fft[..., 1]                
                
                
        # q_fft = torch.fft.rfft2(q_patch.float())
        # k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        b, c, h, w = out.shape
        out_two = torch.empty(b, c, h, w, 2, device='cuda')
        out_two[..., 0] = out.real
        out_two[..., 1] = out.imag       
        
        out = fft2c(out_two)
        out = complex_to_chan_dim(out)
        # out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


# class Attention(nn.Module):
#     def __init__(self, config, vis):
#         super(Attention, self).__init__()
#         self.vis = vis
#         self.num_attention_heads = config.transformer["num_heads"]
#         self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = Linear(config.hidden_size, self.all_head_size)
#         self.key = Linear(config.hidden_size, self.all_head_size)
#         self.value = Linear(config.hidden_size, self.all_head_size)

#         self.out = Linear(config.hidden_size, config.hidden_size)
#         self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
#         self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

#         self.softmax = Softmax(dim=-1)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, hidden_states):
#         mixed_query_layer = self.query(hidden_states)
#         mixed_key_layer = self.key(hidden_states)
#         mixed_value_layer = self.value(hidden_states)

#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)

#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         attention_probs = self.softmax(attention_scores)
#         weights = attention_probs if self.vis else None
#         attention_probs = self.attn_dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         attention_output = self.out(context_layer)
#         attention_output = self.proj_dropout(attention_output)
#         return attention_output, weights



# class Embeddings(nn.Module):
#     """Construct the embeddings from patch, position embeddings.
#     """
#     def __init__(self, config, img_size, in_channels=128):
#         super(Embeddings, self).__init__()
#         self.hybrid = None
#         self.config = config
#         img_size = _pair(img_size)


#         patch_size = _pair(config.patches["size"])
#         n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

#         self.patch_embeddings = Conv2d(in_channels=in_channels,
#                                        out_channels=config.hidden_size,
#                                        kernel_size=patch_size,
#                                        stride=patch_size)
#         self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

#         self.dropout = Dropout(config.transformer["dropout_rate"])


#     def forward(self, x):
#         # print(x.shape)  in_channels=128
#         x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
#         x = x.flatten(2)
#         x = x.transpose(-1, -2)  # (B, n_patches, hidden)
#         # print(x.shape) torch.Size([1, 1024, 768])
#         # print(self.position_embeddings.shape)  torch.Size([1, 1024, 768])
#         embeddings = x + self.position_embeddings
#         embeddings = self.dropout(embeddings)
#         return embeddings


# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)

#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h

#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights


# class Encoder(nn.Module):
#     def __init__(self, config, vis):
#         super(Encoder, self).__init__()
#         self.vis = vis
#         self.layer = nn.ModuleList()
#         self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         for _ in range(config.transformer["num_layers"]):
#             layer = Block(config, vis)
#             self.layer.append(copy.deepcopy(layer))

#     def forward(self, hidden_states):

#         for layer_block in self.layer:
#             hidden_states, weights = layer_block(hidden_states)

#         encoded = self.encoder_norm(hidden_states)
#         return encoded


# class Transformer(nn.Module):
#     def __init__(self, config, img_size, vis):
#         super(Transformer, self).__init__()
#         self.embeddings = Embeddings(config, img_size=img_size)
#         self.encoder = Encoder(config, vis)

#     def forward(self, input_ids):
#         embedding_output = self.embeddings(input_ids)
#         encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
#         return encoded



# class VisionTransformer(nn.Module):
#     def __init__(self, config, img_size=32, vis=False):
#         super(VisionTransformer, self).__init__()

#         self.transformer = Transformer(config, img_size, vis)
#         self.config = config

#     def forward(self, x):
#         x = self.transformer(x)  # (B, n_patch, hidden)
#         B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
#         h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
#         x = x.permute(0, 2, 1)
#         x = x.contiguous().view(B, hidden, h, w)
#         return x



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
    


CONFIGS = {
'ViT-B_16': configs.get_b16_config(),
}