import torch
from torch import nn

from modules.commons.layers import LayerNorm
import torch.nn.functional as F

# import math
# from modules.commons import stocpred_modules

class DurationPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=kernel_size // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = nn.Sequential(torch.nn.Linear(n_chans, 1), nn.Softplus())

    def forward(self, x, x_padding=None):
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x = f(x)  # (B, C, Tmax)
            if x_padding is not None:
                x = x * (1 - x_padding.float())[:, None, :]

        x = self.linear(x.transpose(1, -1))  # [B, T, C]
        x = x * (1 - x_padding.float())[:, :, None]  # (B, T, C)
        x = x[..., 0]  # (B, Tmax)
        return x


class LengthRegulator(torch.nn.Module):
    def __init__(self, pad_value=0.0):
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, dur, dur_padding=None, alpha=1.0):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        assert alpha > 0
        """
        dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2token = (token_idx * token_mask.long()).sum(1)
        return mel2token


class PitchPredictor(torch.nn.Module):
    def __init__(self, idim, n_layers=5, n_chans=384, odim=2, kernel_size=5, dropout_rate=0.1):
        super(PitchPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, padding=kernel_size // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x = f(x)  # (B, C, Tmax)
        x = self.linear(x.transpose(1, -1))  # (B, Tmax, H)
        return x
    
# class StochasticPitchPredictor(nn.Module):
#     """Borrowed from VITS"""
#     def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
#         super().__init__()
        
#         filter_channels = in_channels # it needs to be removed from future version.
#         self.in_channels = in_channels
#         self.filter_channels = filter_channels
#         self.kernel_size = kernel_size
#         self.p_dropout = p_dropout
#         self.n_flows = n_flows
#         self.gin_channels = gin_channels

#         self.log_flow = stocpred_modules.Log()
#         self.flows = nn.ModuleList()
#         self.flows.append(stocpred_modules.ElementwiseAffine(2))
#         for i in range(n_flows):
#             self.flows.append(stocpred_modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
#             self.flows.append(stocpred_modules.Flip())

#         self.pre = nn.Conv1d(in_channels, filter_channels, 1)
#         self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
#         self.convs = stocpred_modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
#         if gin_channels != 0:
#             self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

#     def forward(self, spect, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
#         x = torch.detach(spect)
#         x = self.pre(x)
#         if g is not None:
#             g = torch.detach(g)
#             a = self.cond(g)
#             x = x + self.cond(g)
#         x = self.convs(x, x_mask)
#         x = self.proj(x) * x_mask
#         if not reverse:
#             flows = self.flows
#             assert w is not None

#             e_q = torch.randn(w.size()).to(device=x.device, dtype=x.dtype) * x_mask

#             logdet_tot = 0
#             z = torch.cat([w, e_q], 1)
#             for flow in flows:
#                 z, logdet = flow(z, x_mask, g=x, reverse=reverse)
#                 logdet_tot = logdet_tot + logdet
#             nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            
#             stoch_pitch_loss = nll / torch.sum(x_mask)
#             stoch_pitch_loss = torch.sum(stoch_pitch_loss)
#             return stoch_pitch_loss, None # [b]
            
#         else:
#             flows = list(reversed(self.flows))
#             flows = flows[:-2] + [flows[-1]] # remove a useless vflow
#             z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
#             for flow in flows:
#                 z = flow(z, x_mask, g=x, reverse=reverse)
#             z0, z1 = torch.split(z, [1, 1], 1)
#             w = z0
#             return None, w

class EnergyPredictor(PitchPredictor):
    pass


