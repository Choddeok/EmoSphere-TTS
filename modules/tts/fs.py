from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from modules.commons.conv import TextConvEncoder, ConvBlocks
from modules.commons.layers import Embedding
from modules.commons.nar_tts_modules import PitchPredictor, DurationPredictor, LengthRegulator
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.rnn import TacotronEncoder, RNNEncoder, DecoderRNN
from modules.commons.transformer import FastSpeechEncoder, FastSpeechDecoder
from modules.commons.wavenet import WN
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
from tasks.tts.GRL import VQEmbeddingEMA


FS_ENCODERS = {
    'fft': lambda hp, dict_size: FastSpeechEncoder(
        dict_size, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, dict_size: TacotronEncoder(
        hp['hidden_size'], dict_size, hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, dict_size: RNNEncoder(dict_size, hp['hidden_size']),
    'conv': lambda hp, dict_size: TextConvEncoder(dict_size, hp['hidden_size'], hp['hidden_size'],
                                                  hp['enc_dilations'], hp['enc_kernel_size'],
                                                  layers_in_block=hp['layers_in_block'],
                                                  norm_type=hp['enc_dec_norm'],
                                                  post_net_kernel=hp.get('enc_post_net_kernel', 3)),
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

FS_DECODERS = {
    'fft': lambda hp: FastSpeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'],
                                  hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'],
                                  norm_type=hp['enc_dec_norm'], dropout=hp['dropout'],
                                  post_net_kernel=hp.get('dec_post_net_kernel', 3)),
    'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, dilation_rate=1, n_layers=hp['dec_layers'],
                        is_BTC=True),
}


class FastSpeech(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.hidden_size = hparams['hidden_size']
        self.encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.out_dims = hparams['audio_num_mel_bins'] if out_dims is None else out_dims
        self.mel_out = nn.Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_spk_lookup']:
            self.spk_id_proj = Embedding(hparams['num_spk'], self.hidden_size // 2)
        if hparams['use_emo_lookup']:
            self.emo_id_proj = Embedding(hparams['num_emo'], self.hidden_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size // 2, bias=True)
            self.vqvae = VQEmbeddingEMA(n_embeddings=hparams['vq_n_emb'], embedding_dim = self.hidden_size // 2, commitment_cost=0.25)
        if hparams['use_emo_embed']:
            self.emo_embed_proj = nn.Linear(256, self.hidden_size, bias=True)
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=5, dropout_rate=0.1, odim=2,
                kernel_size=hparams['predictor_kernel'])
        if hparams['dec_inp_add_noise']:
            self.z_channels = hparams['z_channels']
            self.dec_inp_noise_proj = nn.Linear(self.hidden_size + self.z_channels, self.hidden_size)

    def forward(self, txt_tokens, mels=None, mel2ph=None, spk_id=None, emo_id=None, f0=None, uv=None, emo_phVAD = None, infer=False, **kwargs):
        ret = {}
        #####################
        #   Text Encoder    #
        #####################
        encoder_out = self.encoder(txt_tokens)  # [B, T, C]
        
        #########################
        #   Reference Encoder   #
        #########################
        spks_embed=0
        if self.hparams['use_spk_lookup']:
            spks_embed = self.forward_style_embed(spk_id)
        
        emos_embed = 0
        inten_embed = 0
        if self.hparams['use_emo_embed']:
            enc_out = self.DINO_emb(mels)
            emos_embed = emos_embed + self.emo_embed_proj(enc_out)[:, None, :]
        
        if self.hparams['use_emo_lookup']:
            emos_embed = emos_embed + self.emo_id_proj(emo_id)[:, None, :]
        
        dur_embed = spks_embed + emos_embed + inten_embed
        out_embed = spks_embed + emos_embed
        
        #########################
        #   Variance Adaptor    #
        #########################
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        # add dur
        dur_inp = (encoder_out + dur_embed) * src_nonpadding
        mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        
        decoder_inp = expand_states(encoder_out, mel2ph) + expand_states(inten_embed, mel2ph)

        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = (decoder_inp + out_embed) * tgt_nonpadding
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp, f0, uv, mel2ph, ret, encoder_out)

        #################
        #   Deocoder    #
        #################
        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp) * tgt_nonpadding
        # ret['decoder_inp'] = decoder_inp = (decoder_inp + out_embed) * tgt_nonpadding
        if self.hparams['dec_inp_add_noise']:
            B, T, _ = decoder_inp.shape
            z = kwargs.get('adv_z', torch.randn([B, T, self.z_channels])).to(decoder_inp.device)
            ret['adv_z'] = z
            decoder_inp = torch.cat([decoder_inp, z], -1)
            decoder_inp = self.dec_inp_noise_proj(decoder_inp) * tgt_nonpadding
        ret['mel_out'] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        return ret

    def forward_style_embed(self, spk_id=None, vq=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_lookup']:
            style_embed = style_embed + self.spk_id_proj(spk_id)
        return style_embed

    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        return x * tgt_nonpadding


    def compute_perpendicular_foot(self, A, P, B):
        """Compute the foot of the perpendicular from P onto the line AB."""
        AP = P - A
        AB = B - A
        t = torch.dot(AP, AB) / torch.dot(AB, AB)
        D = A + t * AB
        return D

    def compute_adjusted_coordinate(self, P, D, scale):
        """Compute the adjusted coordinate based on the scale."""
        return P + (scale - 1) * (D - P)
    
    def adjust_coordinates(self, tensor, scale):
        B, L, _ = tensor.shape
        # Initialize the adjusted tensor with zeros
        adjusted = torch.full_like(tensor, float('nan'))
        
        # Iterate for each batch
        for b in range(B):
            # First and last coordinates are kept unchanged for each batch
            adjusted[b, 0] = tensor[b, 0]

            # Find the last non-zero coordinate for each batch
            last_nan_idx = L
            for i in range(0, L-1):
                if torch.any(torch.isnan(tensor[b, i])):
                    last_nan_idx = i - 1
                    break

            # Iterate from 2nd to the one before the last non-zero coordinate for each batch
            for i in range(1, last_nan_idx):
                A = tensor[b, i-1]
                P = tensor[b, i]
                B = tensor[b, i+1]

                D = self.compute_perpendicular_foot(A, P, B)
                adjusted_coord = self.compute_adjusted_coordinate(P, D, scale)

                adjusted[b, i] = adjusted_coord

            # Keep the last non-zero coordinate unchanged for each batch
            adjusted[b, last_nan_idx] = tensor[b, last_nan_idx]
        
        return adjusted