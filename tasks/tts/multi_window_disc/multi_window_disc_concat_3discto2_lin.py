import numpy as np
import torch
import torch.nn as nn

class SingleWindowDisc(nn.Module):
    def __init__(self, time_length, freq_length=80, kernel=(3, 3), c_in=1, hidden_size=128):
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.uncond_model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        self.emo_cond_model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        self.spk_cond_model = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(c_in, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(hidden_size, 0.8)
            ]),            
            nn.Sequential(*[
                nn.Conv2d(hidden_size, hidden_size, kernel, (2, 2), padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]),
        ])
        
        ds_size = (time_length // 2 ** 3, ((freq_length + 7) // 2 ** 3) + ((freq_length + 128 + 7) // 2 ** 3))
        self.emo_adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)
        self.spk_adv_layer = nn.Linear(hidden_size * ds_size[0] * ds_size[1], 1)
        self.emo_layer = nn.Linear(128, 128)
        self.spk_layer = nn.Linear(128, 128)


    def forward(self, x, emo_cond_embed, spk_cond_embed):
        """
        :param x: [B, C, T, n_bins]
        :condition_emb: [B, 1, H] 
        :return: validity: [B, 1], h: List of hiddens
        """
        _, _, T_i, _ = x.shape
        emo_cond_embed = self.emo_layer(emo_cond_embed.squeeze(1))
        emo_cond_embed = emo_cond_embed.unsqueeze(1)
        
        spk_cond_embed = self.spk_layer(spk_cond_embed.squeeze(1))
        spk_cond_embed = spk_cond_embed.unsqueeze(1)
        
        emo_embedding_expanded = emo_cond_embed.repeat(1, T_i, 1)
        emo_embedding_unsqueezed = emo_embedding_expanded.unsqueeze(1)
        x_emo_cond = torch.cat([x, emo_embedding_unsqueezed], dim=-1)

        spk_embedding_expanded = spk_cond_embed.repeat(1, T_i, 1)
        spk_embedding_unsqueezed = spk_embedding_expanded.unsqueeze(1)
        x_spk_cond = torch.cat([x, spk_embedding_unsqueezed], dim=-1)

        for l in self.uncond_model:
            x = l(x)
        for l in self.emo_cond_model:
            x_emo_cond = l(x_emo_cond)
        for l in self.emo_cond_model:
            x_spk_cond = l(x_spk_cond)
        
        x = x.view(x.shape[0], -1)
        x_emo_cond = x_emo_cond.view(x_emo_cond.shape[0], -1)
        x_spk_cond = x_spk_cond.view(x_spk_cond.shape[0], -1)
        
        x_emo = torch.cat([x, x_emo_cond], dim=1)
        x_spk = torch.cat([x, x_spk_cond], dim=1)
        emo_validity = self.emo_adv_layer(x_emo)  # [B, 1]
        spk_validity = self.spk_adv_layer(x_spk)  # [B, 1]
        return emo_validity, spk_validity


class MultiWindowDiscriminator(nn.Module):
    def __init__(self, time_lengths, freq_length=336, kernel=(3, 3), c_in=1, hidden_size=128):
        super(MultiWindowDiscriminator, self).__init__()
        self.win_lengths = time_lengths
        self.discriminators = nn.ModuleList()

        for time_length in time_lengths:
            self.discriminators += [SingleWindowDisc(time_length, freq_length, kernel, c_in=c_in, hidden_size=hidden_size)]

    def forward(self, x, x_len, emo_cond_embed, spk_cond_embed, start_frames_wins=None):
        '''
        Args:
            x (tensor): input mel, (B, c_in, T, n_bins).
            x_length (tensor): len of per mel. (B,).

        Returns:
            tensor : (B).
        '''
        e_cond_validity = []
        s_cond_validity = []
        if start_frames_wins is None:
            start_frames_wins = [None] * len(self.discriminators)
        for i, start_frames in zip(range(len(self.discriminators)), start_frames_wins):
            x_clip, start_frames = self.clip(x, x_len, self.win_lengths[i], start_frames)
            start_frames_wins[i] = start_frames
            if x_clip is None:
                continue
            e_cond, s_cond = self.discriminators[i](x_clip, emo_cond_embed, spk_cond_embed)
            e_cond_validity.append(e_cond)
            s_cond_validity.append(s_cond)
        if len(e_cond_validity) != len(self.discriminators):
            return None, start_frames_wins
        e_cond_validity = sum(e_cond_validity)  # [B]
        s_cond_validity = sum(s_cond_validity)  # [B]
        return e_cond_validity, s_cond_validity

    def clip(self, x, x_len, win_length, start_frames=None):
        '''Ramdom clip x to win_length.
        Args:
            x (tensor) : (B, c_in, T, n_bins).
            cond (tensor) : (B, T, H).
            x_len (tensor) : (B,).
            win_length (int): target clip length

        Returns:
            (tensor) : (B, c_in, win_length, n_bins).

        '''
        T_start = 0
        T_end = x_len.max() - win_length
        if T_end < 0:
            return None, None, start_frames
        T_end = T_end.item()
        if start_frames is None:
            start_frame = np.random.randint(low=T_start, high=T_end + 1)
            start_frames = [start_frame] * x.size(0)
        else:
            start_frame = start_frames[0]
        x_batch = x[:, :, start_frame: start_frame + win_length]
        return x_batch, start_frames


class Discriminator(nn.Module):
    def __init__(self, time_lengths=[32, 64, 128], freq_length=336, kernel=(3, 3), c_in=1,
                 hidden_size=128):
        super(Discriminator, self).__init__()
        self.time_lengths = time_lengths
        self.discriminator = MultiWindowDiscriminator(
            freq_length=freq_length,
            time_lengths=time_lengths,
            kernel=kernel,
            c_in=c_in, hidden_size=hidden_size
        )


    def forward(self, x, emo_cond_embed, spk_cond_embed, start_frames_wins=None):
        """
        :param x: [B, T, 80]
        :param return_y_only:
        :return:
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :] # [B,1,T,80]
        x_len = x.sum([1, -1]).ne(0).int().sum([-1])
        ret = {'e_y_cond': None, 's_y_cond': None}
        ret['e_y_cond'], ret['s_y_cond'] = self.discriminator(
            x, x_len, emo_cond_embed, spk_cond_embed, start_frames_wins=start_frames_wins)
        return ret