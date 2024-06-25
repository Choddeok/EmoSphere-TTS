import torch
from modules.vocoder.bigvgan.models import BigVGAN as model
from tasks.tts.vocoder_infer.base_vocoder import register_vocoder, BaseVocoder
from utils.commons.hparams import hparams
from utils.commons.meters import Timer
import json
import os
import pyloudnorm as pyln

total_time = 0
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@register_vocoder("BigVGAN")
class BigVGAN(BaseVocoder):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(hparams["vocoder_ckpt"], "config.json"), "r") as f:
            data = f.read()
        config = json.loads(data)
        h = AttrDict(config)
        # Generator(h)
        self.model = model(h)
        checkpoint_dict = torch.load(
            os.path.join(hparams["vocoder_ckpt"], "g_05050000"),
            map_location=self.device,
        )
        print("##################### Load BigVGAN")
        self.model.load_state_dict(checkpoint_dict["generator"])
        self.model.to(self.device)
        self.model.eval()
        self.model.remove_weight_norm()

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).to(device)
            c = c.transpose(2, 1)
            with Timer("bigvgan", enable=hparams["profile_infer"]):
                y = self.model(c)

        audio = y.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")
        audio = 0.95 * (audio / audio.max())

        meter = pyln.Meter(hparams["audio_sample_rate"])  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, -20.0)
        # print(audio)
        # wav_out = y.squeeze().cpu().numpy()
        # wav_out = wav_out * MAX_WAV_VALUE
        return audio
