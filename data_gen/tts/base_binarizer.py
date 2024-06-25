import json
import os
import random
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm

import utils.commons.single_thread_env  # NOQA
from utils.audio import librosa_wav2spec_bigvgan
from utils.audio.align import get_mel2ph, mel2token_to_dur
from utils.audio.cwt import get_lf0_cwt, get_cont_lf0
from utils.audio.pitch.utils import f0_to_coarse
from utils.audio.pitch_extractors import extract_pitch_simple
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file
from torch.nn.functional import pairwise_distance

np.seterr(divide='ignore', invalid='ignore')

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states_all = outputs[0]
        hidden_states = torch.mean(hidden_states_all, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states_all, logits

class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []
        self.num_list = []
        self.train_item_name = []
        self.valid_item_name = []
        self.test_item_name = []
        self.test_valid_item_name = []
        self.VAD_values = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.VAD_values2 = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.max_distance = {}
        
    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list_ = json.load(open(f"{processed_data_dir}/metadata.json"))
        items_list = sorted(items_list_, key=lambda d: d['item_name']) ##
        
        num_id = []
        for r in items_list:
            i = r['spk_id']
            num_id.append(i)    # 0~9번 스피커까지 스피커 넘버 정렬
        for i in range(10):
            num = num_id.count(i)
            self.num_list.append(num)   # 스피커 별로 음성 num 정립 [1750, 1749, 1747, 1749, 1737, 1750, 1747, 1743, 1749, 1750] 
        
        
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)

    @property
    def all_item_names(self):
        self.all_item_name = [i for i in self.item_names]
        return self.all_item_name

    @property
    def train_item_names(self):
        self.train_item_name = [i for i in self.item_names if i not in self.test_valid_item_name]
        
        return self.train_item_name

    @property
    def test_item_names(self):
        num = 0
        nums = []
        nums.append(num)

        for i in self.num_list:
            num += i
            nums.append(num)    # [0, 1750, 3500, 5250, 7000, 8750, 10500, 12250, 14000, 15750, 17500]
        # nums[:len(nums)-1]: [0, 1750, 3500, 5250, 7000, 8750, 10500, 12250, 14000, 15750]  
        for i in nums[:len(nums)-1]:
            self.test_item_name += self.item_names[i+20:i+50]
            self.test_item_name += self.item_names[i+370:i+400]
            self.test_item_name += self.item_names[i+720:i+750]
            self.test_item_name += self.item_names[i+1070:i+1100]
            self.test_item_name += self.item_names[i+1420:i+1450]
        
        return self.test_item_name

    @property
    def valid_item_names(self):
        num = 0
        nums = []
        nums.append(num)
        
        for i in self.num_list:
            num += i
            nums.append(num)    
             
        for i in nums[:len(nums)-1]:
            self.valid_item_name += self.item_names[i:i+20]
            self.valid_item_name += self.item_names[i+350:i+370]
            self.valid_item_name += self.item_names[i+700:i+720]
            self.valid_item_name += self.item_names[i+1050:i+1070]
            self.valid_item_name += self.item_names[i+1400:i+1420]
            
        for i in nums[:len(nums)-1]:
            self.test_valid_item_name += self.item_names[i:i+50]
            self.test_valid_item_name += self.item_names[i+350:i+400]
            self.test_valid_item_name += self.item_names[i+700:i+750]
            self.test_valid_item_name += self.item_names[i+1050:i+1100]
            self.test_valid_item_name += self.item_names[i+1400:i+1450]
            
        return self.valid_item_name

    def meta_data(self, prefix):
        if prefix == 'all':
            item_names = self.all_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        elif prefix == 'valid':
            item_names = self.valid_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        for fn in ['phone_set.json', 'word_set.json', 'spk_map.json']:
            remove_file(f"{hparams['binary_data_dir']}/{fn}")
            copy_file(f"{hparams['processed_data_dir']}/{fn}", f"{hparams['binary_data_dir']}/{fn}")
        
        self.process_data('all')
        self.process_data('test')
        self.process_data('valid')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item, binarization_args=self.binarization_args, prefix=prefix)
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is not None:
                items.append(item)
                
        device = "cpu"
        if self.binarization_args['with_spk_embed']:
            args = [{'wav': item['wav']} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                    self.get_spk_embed, args,
                    init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().to(device)}, num_workers=4,
                    desc='Extracting spk embed'):
                items[item_id]['spk_embed'] = spk_embed
        
        #######################
        #   평균, 분산 고려   #
        #######################
        device = "cpu"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = EmotionModel.from_pretrained(model_name)
        model.to(device)
        print("start")
        if prefix == "all":
            VAD_Neu_center = torch.tensor([0.4135, 0.5169, 0.3620]).to(device)
            all_distances = []
            for item in tqdm(items, desc="Processing items"):
                emo_id = item['emo_id']
                wav_gt = torch.from_numpy(item['wav'])
                wav_gt = wav_gt.unsqueeze(0)
                pro_wav = processor(wav_gt, sampling_rate=16000)
                pro_wav = pro_wav['input_values'][0]
                pro_wav = torch.from_numpy(pro_wav)
        
                with torch.no_grad():
                    _, emo_VAD = model(pro_wav.to(device))
                    self.VAD_values[emo_id].append(emo_VAD)
                    self.VAD_values2[emo_id].append((emo_VAD, item['item_name']))
                    
            for emo_id in tqdm(self.VAD_values):
                print(f"Calculating statistics for emo_id: {emo_id}")
                
                all_distances = []
                for emo_vad, item_name in self.VAD_values2[emo_id]:
                    distance_to_neutral = pairwise_distance(emo_vad.unsqueeze(0), VAD_Neu_center, p=2)
                    all_distances.append(distance_to_neutral.item())
                self.max_distance[emo_id] = max(all_distances)
            
        else:
            VAD_Neu_center = torch.tensor([0.4135, 0.5169, 0.3620]).to(device)
            for item in tqdm(items, desc="Processing items"):
                emo_id = item['emo_id']
                wav_gt = torch.from_numpy(item['wav'])
                wav_gt = wav_gt.unsqueeze(0)
                pro_wav = processor(wav_gt, sampling_rate=16000)
                pro_wav = pro_wav['input_values'][0]
                pro_wav = torch.from_numpy(pro_wav)
        
                with torch.no_grad():
                    item['hidden_emotion_vector'], emo_VAD = model(pro_wav.to(device))

                max_distance = self.max_distance[emo_id]
                re_emo_VAD = emo_VAD - VAD_Neu_center
                item['spherical_emotion_vector'] = self.convert_tensor_to_polar_coordinates(re_emo_VAD, max_distance).to("cpu")
                
        for item in items:
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args, prefix):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        if wav.shape[0] < 32000:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        item['wav_fn'] = item['wav_fn'].replace("/hd0/choddeok/", "/workspace/choddeok/hd0/")
        # if max(item['mel2ph']*320) != item['wav'].shape[1]
        
        try:
            n_bos_frames, n_eos_frames = 0, 0
            if binarization_args['with_align']:
                tg_fn = f"{hparams['processed_data_dir']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item)
                if binarization_args['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * hparams['hop_size']:len(wav) - n_eos_frames * hparams['hop_size']]
            if binarization_args['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def convert_tensor_to_polar_coordinates(cls, tensor,max_distance):
        r = torch.sqrt(torch.sum(tensor**2, dim=1))
        r_norm = torch.sqrt(torch.sum(tensor**2, dim=1))/ max_distance  # 반지름 계산
        theta = torch.acos(tensor[:, 2] / r)  # 경사각 계산
        phi = torch.atan2(tensor[:, 1], tensor[:, 0])  # 방위각 계산

        polar_coordinates_tensor = torch.stack((r_norm, theta, phi), dim=1)  # 변환된 좌표 텐서
        return polar_coordinates_tensor
       
    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = librosa_wav2spec_bigvgan(wav_fn, fft_size=hparams['fft_size'],
                                         hop_size=hparams['hop_size'],
                                         win_length=hparams['win_size'],
                                         num_mels=hparams['audio_num_mel_bins'],
                                         fmin=hparams['fmin'],
                                         fmax=hparams['fmax'],
                                         sample_rate=hparams['audio_sample_rate'],
                                         loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        if binarization_args['with_linear']:
            res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel

    @staticmethod
    def process_align(tg_fn, item):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams['hop_size'], hparams['audio_sample_rate'],
                                     hparams['binarization_args']['min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur
        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch_simple(item['wav'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        if hparams['binarization_args']['with_f0cwt']:
            uv, cont_lf0_lpf = get_cont_lf0(f0)
            logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
            cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
            item['cwt_spec'] = cwt_spec
            item['cwt_mean'] = logf0s_mean_org
            item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))