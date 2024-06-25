import torch
import torch.nn.functional as F
from modules.tts.EmoSphere import FastSpeech2Orig
from tasks.tts.dataset_utils import FastSpeechDataset
from tasks.tts.fs import FastSpeechTask
from utils.commons.dataset_utils import collate_1d, collate_2d
from utils.commons.hparams import hparams
from utils.plot.plot import spec_to_figure
import numpy as np
from utils.commons.tensor_utils import tensors_to_scalars
import torch.nn as nn
from utils.nn.model_utils import num_params
from tasks.tts.multi_window_disc.multi_window_disc_concat_3discto2_lin import Discriminator

class FastSpeech2OrigDataset(FastSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix, shuffle, items, data_dir)
        self.pitch_type = hparams.get('pitch_type')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        sample['energy'] = (mel.exp() ** 2).sum(-1).sqrt()
        if hparams['use_pitch_embed'] and self.pitch_type == 'cwt':
            cwt_spec = torch.Tensor(item['cwt_spec'])[:T]
            f0_mean = item.get('f0_mean', item.get('cwt_mean'))
            f0_std = item.get('f0_std', item.get('cwt_std'))
            sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super().collater(samples)
        if hparams['use_pitch_embed']:
            energy = collate_1d([s['energy'] for s in samples], 0.0)
        else:
            energy = None
        batch.update({'energy': energy})
        if self.pitch_type == 'cwt':
            cwt_spec = collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        return batch


class FastSpeech2OrigTask(FastSpeechTask):
    def __init__(self):
        super(FastSpeech2OrigTask, self).__init__()
        self.dataset_cls = FastSpeech2OrigDataset
        self.build_disc_model()
        self.mse_loss_fn = torch.nn.MSELoss()
        
    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = FastSpeech2Orig(dict_size, hparams)

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 96][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())
        
    def on_train_start(self):
        super().on_train_start()
        for n, m in self.model.named_children():
            num_params(m, model_name=n)
        num_params(self.mel_disc, model_name='disc')
    
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                emo_cond_embed = model_out['emo_embed']
                spk_cond_embed = model_out['spk_embed']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                    
                o_ = self.mel_disc(mel_p, emo_cond_embed, spk_cond_embed)
                e_p_cond, s_p_cond = o_['e_y_cond'], o_['s_y_cond']
                loss_output['a_e'] = self.mse_loss_fn(e_p_cond, e_p_cond.new_ones(e_p_cond.size()))
                loss_output['a_s'] = self.mse_loss_fn(s_p_cond, s_p_cond.new_ones(s_p_cond.size()))
                loss_weights['a_e'] = hparams['lambda_mel_adv']
                loss_weights['a_s'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                emo_cond_embed = model_out['emo_embed']
                spk_cond_embed = model_out['spk_embed']
                
                o = self.mel_disc(mel_g, emo_cond_embed, spk_cond_embed)
                e_g_cond, s_g_cond = o['e_y_cond'], o['s_y_cond']
                o_ = self.mel_disc(mel_p, emo_cond_embed, spk_cond_embed)
                e_p_cond, s_p_cond = o_['e_y_cond'], o_['s_y_cond']
                
                loss_output["r"] = (self.mse_loss_fn(e_g_cond, e_g_cond.new_ones(e_g_cond.size())) + self.mse_loss_fn(s_g_cond, s_g_cond.new_ones(s_g_cond.size()))) * 0.5
                loss_output["f"] = (self.mse_loss_fn(e_p_cond, e_p_cond.new_zeros(e_p_cond.size())) + self.mse_loss_fn(s_p_cond, s_p_cond.new_zeros(s_p_cond.size()))) * 0.5

        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['txt_tokens'].size()[0]
        return total_loss, loss_output
    
    def run_model(self, sample, infer=False, *args, **kwargs):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        spk_embed = sample.get('spk_embed')
        emo_id = sample.get('emo_ids')
        # VAD value
        emo_VAD_style = sample['emo_VAD_style']
        emo_VAD_style = emo_VAD_style.squeeze(1)
                
        if hparams['emo_style'] == "I":
            new_tensor = torch.tensor([np.pi/4, np.pi/4])
            emo_VAD_style[0] = new_tensor
        elif hparams['emo_style'] == "III":
            new_tensor = torch.tensor([np.pi/4, -3*np.pi/4])
            emo_VAD_style[0] = new_tensor
        elif hparams['emo_style'] == "V":
            new_tensor = torch.tensor([3*np.pi/4, np.pi/4])
            emo_VAD_style[0] = new_tensor
        elif hparams['emo_style'] == "VII":
            new_tensor = torch.tensor([3*np.pi/4, -3*np.pi/4])
            emo_VAD_style[0] = new_tensor
        
        emo_VAD_inten = sample['emo_VAD_inten']
        emo_VAD_inten = emo_VAD_inten.squeeze(1)
        
        if not infer:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph = sample['mel2ph']  # [B, T_s]
            f0 = sample.get('f0')
            uv = sample.get('uv')
            energy = sample.get('energy')
            output = self.model(txt_tokens, mels=target,mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id, emo_id=emo_id,
            f0=f0, uv=uv, energy=energy, emo_VAD_style=emo_VAD_style, emo_VAD_inten=emo_VAD_inten, infer=False)
            losses = {}
            self.add_mel_loss(output['mel_out'], sample['mels'], losses)
            self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            if hparams['use_energy_embed']:
                self.add_energy_loss(output, sample, losses)
            return losses, output
        else:
            target = sample['mels']  # [B, T_s, 80]
            mel2ph, uv, f0, energy = None, None, None, None
            use_gt_dur = kwargs.get('infer_use_gt_dur', hparams['use_gt_dur'])
            use_gt_f0 = kwargs.get('infer_use_gt_f0', hparams['use_gt_f0'])
            use_gt_energy = kwargs.get('infer_use_gt_energy', hparams['use_gt_energy'])
            if use_gt_dur:
                mel2ph = sample['mel2ph']
            if use_gt_f0:
                f0 = sample['f0']
                uv = sample['uv']
            if use_gt_energy:
                energy = sample['energy']
            output = self.model(txt_tokens, mels=target, mel2ph=mel2ph, spk_embed=spk_embed, spk_id=spk_id, emo_id=emo_id,
            f0=f0, uv=uv, energy=energy, emo_VAD_style=emo_VAD_style, emo_VAD_inten=emo_VAD_inten, infer=True)
            return output

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]
    
    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]
    
    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
    
    def save_valid_result(self, sample, batch_idx, model_out):
        super(FastSpeech2OrigTask, self).save_valid_result(sample, batch_idx, model_out)
        self.plot_cwt(batch_idx, model_out['cwt'], sample['cwt_spec'])

    def plot_cwt(self, batch_idx, cwt_out, cwt_gt=None):
        if len(cwt_out.shape) == 3:
            cwt_out = cwt_out[0]
        if isinstance(cwt_out, torch.Tensor):
            cwt_out = cwt_out.cpu().numpy()
        if cwt_gt is not None:
            if len(cwt_gt.shape) == 3:
                cwt_gt = cwt_gt[0]
            if isinstance(cwt_gt, torch.Tensor):
                cwt_gt = cwt_gt.cpu().numpy()
            cwt_out = np.concatenate([cwt_out, cwt_gt], -1)
        name = f'cwt_val_{batch_idx}'
        self.logger.add_figure(name, spec_to_figure(cwt_out), self.global_step)
        
    def add_pitch_loss(self, output, sample, losses):
        if hparams['pitch_type'] == 'cwt':
            cwt_spec = sample[f'cwt_spec']
            f0_mean = sample['f0_mean']
            uv = sample['uv']
            mel2ph = sample['mel2ph']
            f0_std = sample['f0_std']
            cwt_pred = output['cwt'][:, :, :10]
            f0_mean_pred = output['f0_mean']
            f0_std_pred = output['f0_std']
            nonpadding = (mel2ph != 0).float()
            losses['C'] = F.l1_loss(cwt_pred, cwt_spec) * hparams['lambda_f0']
            if hparams['use_uv']:
                assert output['cwt'].shape[-1] == 11
                uv_pred = output['cwt'][:, :, -1]
                losses['uv'] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction='none')
                                * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
            losses['f0_mean'] = F.l1_loss(f0_mean_pred, f0_mean) * hparams['lambda_f0']
            losses['f0_std'] = F.l1_loss(f0_std_pred, f0_std) * hparams['lambda_f0']
        else:
            super(FastSpeech2OrigTask, self).add_pitch_loss(output, sample, losses)

    def add_energy_loss(self, output, sample, losses):
        energy_pred, energy = output['energy_pred'], sample['energy']
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        losses['e'] = loss