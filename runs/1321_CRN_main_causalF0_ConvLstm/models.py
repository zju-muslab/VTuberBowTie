import torch
import torch.nn as nn
import torch.nn.functional as F
from moudles import ResBlk, AdainResBlk, InstanceNorm2d, ConvLSTM
import copy
from munch import Munch

class CRN(nn.Module):
    """
    Conditional Residual Network (CRN) model for image synthesis.
    """

    def __init__(self, style_dim=48, F0_channel=0):
        """
        Initialize the CRN model.

        Args:
        - style_dim (int): Dimension of the style vector.
        - F0_channel (int): Number of channels in the F0 input.
        """
        super(CRN, self).__init__()
        self.F0_channel = F0_channel
        nh = 512
        nm = [32, 64, 128, 256, 512, nh]

        ds = ['timepreserve', 'timepreserve',
              'timepreserve', 'timepreserve', 'none', 'none']
        dnm, dds = nm[::-1], ds[::-1]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.to_out = nn.Sequential(
            InstanceNorm2d(nm[0], affine=False, track_running_stats=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nm[0], 1, 1, 1, 0))

        for i in range(len(nm)):
            nIn = 1 if i == 0 else nm[i-1]
            nOut = nm[i]
            self.encoder.append(
                ResBlk(nIn, nOut, normalize=True, downsample=ds[i]))

        for i in range(len(dnm)):
            nIn = nh if i == 0 else dnm[i-1]
            nOut = dnm[i]
            self.decoder.append(AdainResBlk(
                nIn, nOut, style_dim, upsample=dds[i]))

        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2),
                       normalize=True, downsample="timepreserve"),
            )

        self.convlstm = ConvLSTM(input_dim=dnm[0]+int(self.F0_channel/2),
                                 hidden_dim=dnm[0], kernel_size=(1, 1), num_layers=1, batch_first=True)

        self.kernel_size = 3
        self.depth = 6
        self.stride = 1

    def Encode(self, x):
        """
        Encode the input image.

        Args:
        - x (torch.Tensor): Input image tensor.

        Returns:
        - torch.Tensor: Encoded image tensor.
        """
        for layer in self.encoder:
            x = layer(x)
        return x

    def Decode(self, c, s, F0=None, rnn_state=None):
        """
        Decode the encoded image.

        Args:
        - c (torch.Tensor): Encoded image tensor.
        - s (torch.Tensor): Style vector.
        - F0 (torch.Tensor): F0 input tensor.
        - rnn_state (torch.Tensor): RNN state tensor.

        Returns:
        - torch.Tensor: Decoded image tensor.
        - torch.Tensor: RNN state tensor.
        """
        output = c
        F0 = self.F0_conv(F0)
        output = torch.cat([output, F0], axis=1)

        output = output.permute(0, 3, 1, 2).unsqueeze(-1)
        output, rnn_state = self.convlstm(
            input_tensor=output, hidden_state=rnn_state)
        output = output[0].squeeze(-1).permute(0, 2, 3, 1)

        for layer in self.decoder:
            output = layer(output, s)
        output = self.to_out(output)
        return output, rnn_state

    def forward(self, x, s, F0=None, rnn_state=None):
        """
        Forward pass of the CRN model.

        Args:
        - x (torch.Tensor): Input image tensor.
        - s (torch.Tensor): Style vector.
        - F0 (torch.Tensor): F0 input tensor.
        - rnn_state (torch.Tensor): RNN state tensor.

        Returns:
        - torch.Tensor: Decoded image tensor.
        - torch.Tensor: Encoded image tensor.
        - torch.Tensor: RNN state tensor.
        """
        c = self.Encode(x)
        output, rnn_state = self.Decode(c, s, F0, rnn_state)
        return output, c, rnn_state

    def feed(self, x, ref, F0, conv_state, norm_state, lstm_state, use_input_stats=True, update_mean_var=False, is_speech=False):
        """
        Feed the input through the CRN model.

        Args:
        - x (torch.Tensor): Input tensor.
        - ref (torch.Tensor): Reference tensor.
        - F0 (torch.Tensor): F0 tensor.
        - conv_state (list): List of convolutional states.
        - norm_state (list): List of normalization states.
        - lstm_state (torch.Tensor): LSTM state tensor.
        - use_input_stats (bool): Whether to use input statistics.
        - update_mean_var (bool): Whether to update mean and variance.
        - is_speech (bool): Whether the input is speech.

        Returns:
        - torch.Tensor: Output tensor.
        - list: Updated convolutional states.
        - list: Updated normalization states.
        - torch.Tensor: Updated LSTM state tensor.
        """
        device = next(self.parameters()).device

        x = x.to(device)
        ref = ref.to(device)
        F0 = F0.to(device)

        next_state = []
        next_state_norm = []
        first = conv_state is None
        kernel_size = 5

        for idx, encode in enumerate(self.encoder):
            next_state.append(x[...,-(kernel_size-1):])
            if first:
                prev = torch.zeros_like(x)[...,-(kernel_size-1):].to(device)
                prev_stats = None
            else:
                prev = conv_state.pop(0)
                prev_stats = norm_state.pop(0)
            x = torch.cat([prev, x], axis=-1)
            x, stats = encode(x, live=True, prev_stats=prev_stats, use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
            next_state_norm.append(stats)

        F0 = self.F0_conv(F0)
        x = torch.cat([x, F0], axis=1)

        x = x.permute(0, 3, 1, 2).unsqueeze(-1)

        x, lstm_state = self.convlstm(
            input_tensor=x, hidden_state=lstm_state)

        x = x[0].squeeze(-1).permute(0, 2, 3, 1)

        for idx, decode in enumerate(self.decoder):
            next_state.append(x[...,-(kernel_size-1):])
            if first:
                prev = torch.zeros_like(x)[...,-(kernel_size-1):].to(device)
                prev_stats = None
            if not first:
                prev = conv_state.pop(0)
                prev_stats = norm_state.pop(0)
            x = torch.cat([prev, x], axis=-1)
            x, stats = decode(x, ref, live=True, prev_stats=prev_stats, use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
            next_state_norm.append(stats)
        if not first:
            prev_stats = norm_state.pop(0)
        else: 
            prev_stats = None
        x, stats = self.to_out[0](x, live=True, prev_stats=prev_stats, use_input_stats=use_input_stats, update_mean_var=update_mean_var)
        x = self.to_out[1](x)
        x = self.to_out[2](x)
        next_state_norm.append(stats)
        norm_state = next_state_norm
        conv_state = next_state
        return x, conv_state, norm_state, lstm_state


class MappingNetwork(nn.Module):
    """
    Mapping network for style vector generation.
    """

    def __init__(self, latent_dim=16, style_dim=48, num_domains=2, hidden_dim=384):
        """
        Initialize the mapping network.

        Args:
        - latent_dim (int): Dimension of the latent vector.
        - style_dim (int): Dimension of the style vector.
        - num_domains (int): Number of domains.
        - hidden_dim (int): Dimension of the hidden layers.
        """
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        """
        Forward pass of the mapping network.

        Args:
        - z (torch.Tensor): Latent vector.
        - y (torch.Tensor): Domain code.

        Returns:
        - torch.Tensor: Style vector.
        """
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    """
    Style encoder network.
    """

    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        """
        Initialize the style encoder.

        Args:
        - dim_in (int): Input dimension.
        - style_dim (int): Dimension of the style vector.
        - num_domains (int): Number of domains.
        - max_conv_dim (int): Maximum convolutional dimension.
        """
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        """
        Forward pass of the style encoder.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Domain code.

        Returns:
        - torch.Tensor: Style vector.
        """
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    """
    Discriminator network for domain classification.
    """

    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        """
        Initialize the discriminator.

        Args:
        - dim_in (int): Input dimension.
        - num_domains (int): Number of domains.
        - max_conv_dim (int): Maximum convolutional dimension.
        - repeat_num (int): Number of repeat blocks.
        """
        super().__init__()

        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                   max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                   max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        self.num_domains = num_domains

    def forward(self, x, y):
        """
        Forward pass of the discriminator.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Domain code.

        Returns:
        - torch.Tensor: Discriminator output.
        """
        return self.dis(x, y)

    def classifier(self, x):
        """
        Get the feature representation for domain classification.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Feature representation.
        """
        return self.cls.get_feature(x)


class Discriminator2d(nn.Module):
    """
    Discriminator network for 2D domain classification.
    """

    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        """
        Initialize the 2D discriminator.

        Args:
        - dim_in (int): Input dimension.
        - num_domains (int): Number of domains.
        - max_conv_dim (int): Maximum convolutional dimension.
        - repeat_num (int): Number of repeat blocks.
        """
        super().__init__()
        pre_blocks = []
        pre_blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            pre_blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out
        self.pre_blocks = nn.Sequential(*pre_blocks)

        back_blocks = []
        back_blocks += [nn.LeakyReLU(0.2)]
        back_blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        back_blocks += [nn.LeakyReLU(0.2)]
        back_blocks += [nn.AdaptiveAvgPool2d(1)]
        back_blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.back_blocks = nn.Sequential(*back_blocks)

    def get_feature(self, x):
        """
        Get the feature representation for domain classification.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Feature representation.
        """
        for l in self.pre_blocks:
            x = l(x)
        fm = x
        out = self.back_blocks(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, fm

    def forward(self, x, y):
        """
        Forward pass of the 2D discriminator.

        Args:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor): Domain code.

        Returns:
        - torch.Tensor: Discriminator output.
        """
        out, fm = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out, fm


def build_model(args, F0_model, ASR_model):
    """
    Build the CRN model.

    Args:
    - args (argparse.Namespace): Command-line arguments.
    - F0_model: F0 model.
    - ASR_model: ASR model.

    Returns:
    - Munch: Dictionary of model components.
    """
    generator = CRN(args.style_dim, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(
        args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(
        args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(
        args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)

    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     f0_model=F0_model,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema
