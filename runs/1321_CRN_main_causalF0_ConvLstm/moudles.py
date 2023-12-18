# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 0)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 0)
        self.pad1=torch.nn.ReplicationPad2d((2,0,1,1))
        self.pad2=torch.nn.ReplicationPad2d((0,0,1,1))
        if self.normalize:
            self.norm1 = InstanceNorm2d(dim_in, affine=False, track_running_stats=True)
            self.norm2 = InstanceNorm2d(dim_in, affine=False, track_running_stats=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x,live=False,prev_stats=[None,None], use_input_stats=True, update_mean_var=False, is_speech=False):
        if self.normalize:
            if not live:
                x = self.norm1(x)
            else:
                x, meanvar = self.norm1(x, live, prev_stats[0], use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
                stats=[meanvar]
        x = self.actv(x)
        x = self.pad2(x) if live else self.pad1(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            if not live:
                x = self.norm2(x)
            else:
                x, meanvar = self.norm2(x, live, prev_stats[1],use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
                stats.append(meanvar)
        x = self.actv(x)
        x = self.pad2(x) if live else self.pad1(x)
        x = self.conv2(x)
        if live==False:
            return x
        else:
            return x, stats

    def forward(self, x, live=False, prev_stats=None,use_input_stats=True, update_mean_var=False,is_speech=False):
        if not live:
            x = self._shortcut(x) + self._residual(x)
        else:
            if prev_stats==None:
                prev_stats=[None,None]
            x_res,stats = self._residual(x,live=live,prev_stats=prev_stats,use_input_stats=use_input_stats, update_mean_var=update_mean_var,is_speech=is_speech)
            x_shortcut=self._shortcut(x)[...,-x_res.shape[-1]:]
            x = x_res+x_shortcut
        if not live:
            return x / math.sqrt(2)  # unit variance
        else:
            return x / math.sqrt(2), stats
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = InstanceNorm2d(num_features, affine=False,track_running_stats=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s, live=False, prev_stats=None, use_input_stats=True, update_mean_var=False,is_speech=False):

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        if not live:
            x=self.norm(x)
            return (1 + gamma) * x + beta
        else:
            x, meanvar= self.norm(x, live, prev_stats=prev_stats, use_input_stats=use_input_stats, update_mean_var=update_mean_var,is_speech=is_speech)
            return (1 + gamma) * x + beta, meanvar

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.pad1=torch.nn.ReplicationPad2d((2,0,1,1))
        self.pad2=torch.nn.ReplicationPad2d((0,0,1,1))
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 0)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 0)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s ,live=False, prev_stats=[None,None], use_input_stats=True, update_mean_var=False, is_speech=False):
        if not live:
            x = self.norm1(x, s)
        else:
            x, meanvar = self.norm1(x,s,live,prev_stats[0], use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
            stats=[meanvar]
        x = self.actv(x)
        x = self.upsample(x)
        x = self.pad2(x) if live else self.pad1(x)
        x = self.conv1(x)
        if not live:
            x = self.norm2(x, s)
        else:
            x, meanvar = self.norm2(x,s,live,prev_stats[1],use_input_stats=use_input_stats, update_mean_var=update_mean_var, is_speech=is_speech)
            stats.append(meanvar)
        x = self.actv(x)
        x = self.pad2(x) if live else self.pad1(x)
        x = self.conv2(x)
        if not live:
            return x
        else:
            return x, stats

    def forward(self, x, s, live=False, prev_stats=None,use_input_stats=True, update_mean_var=False, is_speech=False):
        if not live:
            out = self._residual(x, s)
        else:
            if prev_stats==None:
                prev_stats=[None,None]
            out, stats = self._residual(x, s, live, prev_stats,use_input_stats=use_input_stats, update_mean_var=update_mean_var,is_speech=is_speech)
        if self.w_hpf == 0:
            x_shortcut=self._shortcut(x)[...,-out.shape[-1]:]
            out = (out+x_shortcut)/ math.sqrt(2)
            # out = (out + self._shortcut(x)) / math.sqrt(2)

        if not live:
            return out
        if live:
            return out, stats

class InstanceNorm2d(torch.nn.modules.batchnorm._NormBase):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.01,
            affine: bool = False,
            track_running_stats: bool = False,
            device=None,
            dtype=None
        ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(InstanceNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        assert (momentum>0)
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                    .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                            klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)
        super(InstanceNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)                                                                                                                                                  

    def calc_mean_var(self, feat):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.reshape(N, C, -1).var(dim=2) 
        feat_mean = feat.reshape(N, C, -1).mean(dim=2)
        return feat_mean.view(N, C, 1, 1), feat_var.view(N, C, 1, 1)


    def forward(self, content_feat, live=False, prev_stats=None, momentum=0.05, use_input_stats=True, update_mean_var=False, is_speech=False):
        size = content_feat.shape
        
        if not use_input_stats:
            mean=self.mean
            var=self.var
        else:
            mean, var = self.calc_mean_var(content_feat)

        if update_mean_var:
            self.mean=mean
            self.var=var
        
        if not live:
            normalized_feat = (content_feat - mean.expand(size)) / (var+1e-6).sqrt().expand(size)
            return normalized_feat

        else:
            if is_speech:
                if prev_stats:
                    mean=momentum*mean+(1-momentum)*prev_stats['mean']
                    var=momentum*var+(1-momentum)*prev_stats['var']
                else:
                    mean=momentum*mean+(1-momentum)*self.running_mean.reshape(1,-1,1,1)
                    var=momentum*var+(1-momentum)*self.running_var.reshape(1,-1,1,1)
            else:
                mean=prev_stats['mean'] if prev_stats else self.running_mean.reshape(1,-1,1,1)
                var=prev_stats['var'] if prev_stats else self.running_var.reshape(1,-1,1,1)


            normalized_feat = (content_feat - mean.reshape(1,-1,1,1)) / (var+1e-6).sqrt().reshape(1,-1,1,1)
            return normalized_feat, {'mean':mean, 'var':var, 'running_mean':self.running_mean, 'running_var':self.running_var}



class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

